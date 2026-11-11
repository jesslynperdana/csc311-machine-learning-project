"""Explore feeling words with an explicit training-derived vocabulary.

This is a rule-based extraction of adjectives and equivalent feeling phrases,
not a general grammatical parser or a complete sentiment model. Each category
is counted once per answer; multiple or mixed feelings are allowed.
"""

from pathlib import Path
import re
import unicodedata

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data_splits" / "train.csv"
OUTPUT_DIR = PROJECT_DIR / "distribution_results" / "feeling"
COLUMN = "How do you feel when you watch this sport?"
COLORS = ["#367a9c", "#ce8133", "#48865c"]

# Adjectives and their corresponding nouns/phrases share an interpretable group.
# Observed spelling errors are listed explicitly; no unrestricted fuzzy matching.
FEELINGS = {
    "Excited": "excited|exciting|excitement|excitements|excitments|exictement|thrilled|thrilling|thrill|exhilarated|exhilarating|hyped|ecstatic|euphoric|euphoria|giddy|兴奋",
    "Bored": "bored|boring|boredom|monotonous",
    "Interested / engaged": "interested|interesting|intrigued|intriguing|engaged|engagement|immersed|invested|captivated|captivating|fascinating|fascinated|engrossed|enthralled|absorbed|involved|curious|glued to the game",
    "Uninterested": "uninterested|unattentive|inattentive|not my type",
    "Neutral / indifferent": "neutral|indifferent|unfazed|detached|ordinary",
    "Happy / joyful": "happy|happiness|joy|joyful|joyed|cheerful",
    "Entertained": "entertained|entertaining|amused|amusing|fun|enjoy|enjoyed|enjoyable|satisfying|satisfied|love it",
    "Energetic / intense": "energetic|energized|energised|energetized|invigorated|electric|high energy|full of energy|energy|intense|frenetic|adrenaline|adrenaline rush|dopamine release",
    "Calm / relaxed": "calm|relaxed|relaxing|relaxation|peaceful|comfortable|comfort|chill|meditative|patient|patience",
    "Tense / anxious": "tense|tension|anxious|anxiety|nervous|nervousness|stressed|stressful|stress|stressed out|jittery|on edge|on the edge of my seat|restless",
    "Anticipatory / suspenseful": "anticipation|anticipatory|anticipating|suspense|suspenseful|in suspense",
    "Focused / analytical": "focused|focussed|foucsed|concentrated|analytical|paying attention|locked into",
    "Confused": "confused|confusing|confusion|confusions|confuse",
    "Passionate": "passionate|passion|passinate|enthusiasm|enthusiastic|enthused",
    "Angry / frustrated": "angry|anger|annoyed|frustrated|frustration|contempt",
    "Sad / disappointed": "sad|sadness|sorrow|disappointed|disappointment|heartbreak|heartbroken|regretfulness|pity",
    "Motivated / inspired": "motivated|motivating|motivation|inspired|inspiring|moved",
    "Impressed / amazed": "impressed|amazed|amazing|awe|admiring|admiration|appreciation|surprised",
    "Tired / sleepy": "tired|sleepy|exhausted|fatigued",
    "Scared / overwhelmed": "scared|fearful|afraid|overwhelmed",
    "Connected": "connected|social connection|sense of belonging|united|a part of something bigger",
    "Proud": "proud|pride",
    "Nostalgic": "nostalgic|nostalgia",
    "Competitive": "competitive",
    "Emotional": "emotional",
}


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    # Preserve clause punctuation so negation does not spread across unrelated clauses.
    text = text.replace("\n", "; ")
    text = re.sub(r"[^\w\s.,;:!?]", " ", text)
    return " ".join(text.split())


def plain(text):
    return " ".join(re.sub(r"[^\w\s]", " ", normalize(text)).split())


MISSING = {"", "none", "nothing", "no sense", "unknown", "unsure", "not sure", "idk", "n a", "na", "no idea", "don t know", "i don t know", "i do not know", "i don t feel much in particular"}
# Full-answer exceptions were reviewed in training data; no label is used.
UNRELATED = {
    plain("I feel as though it is tense about checkpoints security all or nothing."): "Off-topic statement about checkpoint security",
}
NO_WATCH = re.compile(r"\b(?:don t|dont|do not)(?:\s+\w+){0,2}\s+watch\b|\bnever(?:\s+\w+){0,3}\s+watch(?:ed)?\b")
NEGATION = re.compile(r"\b(?:not|never|no|don t|dont|doesn t|didn t|isn t|aren t|wasn t|weren t|can t|cannot)\b")


def make_matcher():
    lookup = {}
    for category, variants in FEELINGS.items():
        for variant in variants.split("|"):
            assert variant not in lookup
            lookup[variant] = category
    terms = sorted(lookup, key=lambda term: (-len(term), term))
    patterns = [re.escape(t) if re.search(r"[\u4e00-\u9fff]", t)
                else r"(?<!\w)" + re.escape(t) + r"(?!\w)" for t in terms]
    return re.compile("|".join(patterns)), lookup


def extract_answer(raw, matcher, lookup):
    text = normalize(raw)
    simplified = plain(raw)
    if not simplified:
        return {}, "Blank answer", []
    if simplified in MISSING:
        return {}, "No specific feeling / does not know", []
    if simplified in UNRELATED:
        return {}, UNRELATED[simplified], []
    if NO_WATCH.search(simplified):
        return {}, "Explicitly does not watch / has never watched the sport", []

    categories, suppressed = {}, []
    for match in matcher.finditer(text):
        phrase = match.group()
        category = lookup[phrase]
        # Negation applies locally, ending at punctuation or a contrast/explanation clause.
        clause = re.split(r"[.,;:!?]|\b(?:but|however|although|yet|because|so|then)\b", text[:match.start()])[-1]
        prefix = " ".join(clause.split()[-5:])
        if re.search(r"\bused to\b", clause) and "no longer" in text[match.end():]:
            suppressed.append(phrase + " (past feeling explicitly no longer present)")
            continue
        if NEGATION.search(prefix) and not re.search(r"\bnot only\b", prefix):
            if category == "Interested / engaged":
                categories.setdefault("Uninterested", []).append("negated " + phrase)
            else:
                suppressed.append(phrase + " (negated)")
            continue
        categories.setdefault(category, []).append(phrase)

    # 'Good' is accepted only as a direct feeling, not in phrases such as 'good game'.
    if simplified in {"good", "neutral good", "good motivated excited"}:
        categories.setdefault("Happy / joyful", []).append("good (direct feeling)")
    # Explicit loss of interest is not a positive interest match.
    if re.search(r"\bnever really had an interest\b", simplified):
        categories.setdefault("Uninterested", []).append("never really had an interest")
    reason = "" if categories else "No recognized feeling expression; unrelated, vague or unmapped response"
    return categories, reason, suppressed


def analyze(df):
    matcher, lookup = make_matcher()
    parsed_rows, evidence, excluded = [], [], []
    categories_by_row = []
    included = []
    for index, row in df.iterrows():
        categories, reason, suppressed = extract_answer(row[COLUMN], matcher, lookup)
        categories_by_row.append(set(categories))
        included.append(not reason)
        record = {"csv_row": index+2, "id": row["id"], "label": row["label"], "raw_answer": row[COLUMN]}
        parsed_rows.append({**record, "included": not reason, "matched_feelings": " | ".join(categories),
                            "suppressed_phrases": " | ".join(suppressed), "exclusion_reason": reason})
        if reason:
            excluded.append({**record, "reason": reason, "suppressed_phrases": " | ".join(suppressed)})
        for category, phrases in categories.items():
            evidence.append({**record, "feeling": category, "matched_phrases": " | ".join(sorted(set(phrases)))})
    selected = pd.Series(categories_by_row, index=df.index)
    included = pd.Series(included, index=df.index)
    for filename, rows, columns in [
        ("parsed_answers.csv", parsed_rows, ["csv_row", "id", "label", "raw_answer", "included", "matched_feelings", "suppressed_phrases", "exclusion_reason"]),
        ("excluded_answers.csv", excluded, ["csv_row", "id", "label", "raw_answer", "reason", "suppressed_phrases"]),
        ("matched_feeling_evidence.csv", evidence, ["csv_row", "id", "label", "raw_answer", "feeling", "matched_phrases"]),
    ]:
        pd.DataFrame(rows, columns=columns).to_csv(OUTPUT_DIR / filename, index=False, encoding="utf-8-sig")
    return selected, included


def summarize(df, selected, included, sports):
    scopes = {"Overall": pd.Series(True, index=df.index)}
    scopes.update({sport: df.label.eq(sport) for sport in sports})
    rows = []
    for scope, mask in scopes.items():
        n = int((mask & included).sum())
        for feeling in FEELINGS:
            count = int(selected.loc[mask & included].map(lambda values: feeling in values).sum())
            rows.append({"scope": scope, "feeling": feeling, "selected_count": count,
                         "total_response_count": int(mask.sum()), "included_response_count": n,
                         "excluded_count": int((mask & ~included).sum()),
                         "percent_of_included_answers": 100*count/n if n else float("nan")})
    summary = pd.DataFrame(rows)
    summary.loc[summary.scope.eq("Overall")].to_csv(OUTPUT_DIR / "overall_summary.csv", index=False, encoding="utf-8-sig")
    summary.loc[summary.scope.ne("Overall")].to_csv(OUTPUT_DIR / "summary_by_sport.csv", index=False, encoding="utf-8-sig")
    dictionary = pd.DataFrame([{"feeling": name, "aliases": aliases} for name, aliases in FEELINGS.items()])
    dictionary.to_csv(OUTPUT_DIR / "feeling_dictionary.csv", index=False, encoding="utf-8-sig")
    return summary


def plot_summaries(summary, sports):
    overall = summary.loc[summary.scope.eq("Overall")].sort_values("selected_count", ascending=False, kind="stable")
    names = overall.feeling.tolist()
    fig, ax = plt.subplots(figsize=(12, 11))
    bars = ax.barh(overall.feeling, overall.percent_of_included_answers, color=COLORS[0])
    ax.bar_label(bars, labels=[f"{r.percent_of_included_answers:.1f}% ({r.selected_count})" for r in overall.itertuples()], padding=4, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("Percentage of included training answers")
    ax.set_title(f"Feelings while watching sports - overall (n={int(overall.included_response_count.iloc[0])})")
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    fig.text(.5, .015, "Rule-based grouping of feeling words and phrases. Each group counted once per answer; mixed feelings allowed.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(OUTPUT_DIR / "feelings_overall.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(21, 12))
    width = .8/len(sports)
    midpoint = (len(names)+1)//2
    for ax, subset in zip(axes, [names[:midpoint], names[midpoint:]]):
        for i, sport in enumerate(sports):
            table = summary.loc[summary.scope.eq(sport)].set_index("feeling").loc[subset]
            positions = [j-.4+width/2+i*width for j in range(len(subset))]
            bars = ax.barh(positions, table.percent_of_included_answers, height=width,
                           color=COLORS[i], label=f"{sport} (n={int(table.included_response_count.iloc[0])})")
            ax.bar_label(bars, labels=[f"{r.percent_of_included_answers:.1f}% ({r.selected_count})" for r in table.itertuples()], padding=3, fontsize=9)
        ax.set_yticks(range(len(subset)), subset)
        ax.invert_yaxis()
        ax.set_xlim(0, 108)
        ax.set_xticks(range(0, 101, 20))
        ax.set_xlabel("Percentage of included answers within sport")
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Feelings by sport - training only", fontsize=16)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .955), ncol=len(sports), frameon=False)
    fig.text(.5, .015, "Each group counted once per answer. Mixed feelings retained; blank, no-viewing and unrecognized answers excluded.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, .91), w_pad=3)
    fig.savefig(OUTPUT_DIR / "feelings_by_sport.png", dpi=160)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_FILE, dtype=str, keep_default_na=False)
    sports = sorted(df.label.unique())
    selected, included = analyze(df)
    summary = summarize(df, selected, included, sports)
    plot_summaries(summary, sports)
    print(summary[["scope", "total_response_count", "included_response_count", "excluded_count"]].drop_duplicates().to_string(index=False))
    print(summary.pivot(index="feeling", columns="scope", values="percent_of_included_answers").round(1).to_string())
    print(f"\nSaved all outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
