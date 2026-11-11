"""Explore color associations using only the saved training split.

Normalize spelling and common shades, count each color once per answer,
and exclude vague responses according to the documented project rule.
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
OUTPUT_DIR = PROJECT_DIR / "distribution_results" / "color"
COLUMN = "What colour do you associate with this sport?"
SPORT_COLORS = ["#367a9c", "#ce8133", "#48865c"]

# Shades are grouped explicitly; gold and pale neutral colors stay separate.
# Format: canonical color, matching aliases, display color, interpretation.
COLOR_RULES = [
    ("Red", "red|reds|reddish|maroon|burgundy", "#c74440", "Includes dark red, maroon and burgundy."),
    ("Orange", "orange|oranges|orangish|orangeish", "#e58c32", "Includes light/dark orange through the base color word."),
    ("Yellow", "yellow|yellows|yellowish", "#e1c843", "Includes yellowish; gold remains separate."),
    ("Green", "green|greens|greenish|绿色|绿", "#478c61", "Includes observed Chinese green wording and light/dark green."),
    ("Blue", "blue|blues|bluish|blueish|navy|navy blue|navy blues", "#447faf", "Includes navy and navy blues."),
    ("Purple", "purple|purples|purplish|violet|lavender", "#8c68aa", "Includes common purple shades."),
    ("Pink", "pink|pinks|pinkish", "#d790ae", "Common palette entry; may have zero observed answers."),
    ("Brown", "brown|browns|brownish", "#96724e", "Includes light/dark brown; wood alone is not guessed to be brown."),
    ("Black", "black|blacks|blackish", "#333333", "Specific color word."),
    ("White", "white|whites|whitish", "#fafafa", "Cream remains separate."),
    ("Gray", "gray|grey|grays|greys|grayish|greyish", "#999999", "American and British spellings are merged."),
    ("Gold", "gold|golden", "#bc9b31", "Gold and golden are merged; not grouped into yellow."),
    ("Silver", "silver|silvery", "#c0c7ce", "Common palette entry; may have zero observed answers."),
    ("Beige", "beige", "#cfba98", "Kept separate from brown and cream."),
    ("Cream", "cream", "#f0e4bd", "Yellowish-cream counts as Yellow and Cream."),
    ("Tan", "tan", "#bc946b", "Kept separate from brown and beige."),
]


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


# These are whole words or phrases, not substring searches. The user's rule
# excludes the entire response even when it also contains recognized colors.
# In particular, a complete word 'all' anywhere triggers exclusion.
EXCLUSION_RULES = [
    (r"\ball\b", "Contains all"),
    (r"\bany\b", "Contains any"),
    (r"\bunknown\b|\bunsure\b", "Unknown or unsure"),
    (r"\bnone\b|\bnothing\b", "None or nothing"),
    (r"\brainbow\b|\bvarious\b|\bevery\b|\beverything\b", "Unspecified collection of colors"),
    (r"\bno idea\b|\bno particular\b|\bno specific\b|\bno colou?rs?\b", "No specific color association"),
    (r"\bdon t know\b|\bdo not know\b|\bnot sure\b", "Does not know"),
    (r"\bn a\b|^na$|^name$", "Missing/error expression"),
]


def make_matcher():
    lookup = {}
    for name, aliases, _, _ in COLOR_RULES:
        for alias in aliases.split("|"):
            key = normalize(alias)
            assert key not in lookup
            lookup[key] = name
    ordered = sorted(lookup, key=lambda term: (-len(term), term))
    pieces = [re.escape(term) if re.search(r"[\u4e00-\u9fff]", term)
              else r"(?<!\w)" + re.escape(term) + r"(?!\w)" for term in ordered]
    return re.compile("|".join(pieces)), lookup


def parse_answers(df):
    # 1. Match concrete colors, then apply whole-answer exclusion rules.
    matcher, lookup = make_matcher()
    selections, statuses, excluded, evidence = [], [], [], []
    for index, row in df.iterrows():
        text = normalize(row[COLUMN])
        matches = list(matcher.finditer(text))
        names = {lookup[m.group()] for m in matches}
        reasons = [reason for pattern, reason in EXCLUSION_RULES if re.search(pattern, text)]
        if not text:
            status, reasons = "blank", ["Blank answer"]
        elif reasons:
            status = "nonspecific"
        elif not names:
            status, reasons = "unmatched", ["No recognized concrete color; no color inferred from the wording"]
        else:
            status = "included"
        selections.append(names if status == "included" else set())
        statuses.append(status)
        if status != "included":
            excluded.append({"csv_row": index+2, "id": row["id"], "label": row["label"],
                             "raw_answer": row[COLUMN], "reason": "; ".join(reasons),
                             "recognized_colors_not_counted": " | ".join(sorted(names))})
        else:
            for name in sorted(names):
                evidence.append({"csv_row": index+2, "id": row["id"], "label": row["label"],
                                 "color": name, "matched_phrases": " | ".join(sorted({m.group() for m in matches if lookup[m.group()] == name})),
                                 "raw_answer": row[COLUMN]})
    pd.DataFrame(excluded, columns=["csv_row", "id", "label", "raw_answer", "reason", "recognized_colors_not_counted"]).to_csv(
        OUTPUT_DIR / "excluded_answers.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(evidence, columns=["csv_row", "id", "label", "color", "matched_phrases", "raw_answer"]).to_csv(
        OUTPUT_DIR / "matched_color_evidence.csv", index=False, encoding="utf-8-sig")
    return pd.Series(selections, index=df.index), pd.Series(statuses, index=df.index)


def summarize(df, selections, statuses, sports):
    # 2. Include zero-count palette colors and report exclusions in the same summary.
    scopes = {"Overall": pd.Series(True, index=df.index)}
    scopes.update({sport: df.label.eq(sport) for sport in sports})
    rows = []
    for scope, mask in scopes.items():
        included = mask & statuses.eq("included")
        n = int(included.sum())
        for color, _, _, _ in COLOR_RULES:
            count = int(selections.loc[included].map(lambda answer: color in answer).sum())
            rows.append({"scope": scope, "color": color, "selected_count": count,
                         "total_response_count": int(mask.sum()), "included_response_count": n,
                         "blank_count": int((mask & statuses.eq("blank")).sum()),
                         "nonspecific_count": int((mask & statuses.eq("nonspecific")).sum()),
                         "unmatched_count": int((mask & statuses.eq("unmatched")).sum()),
                         "excluded_count": int((mask & statuses.ne("included")).sum()),
                         "percent_of_included_answers": count / n * 100 if n else float("nan")})
    summary = pd.DataFrame(rows)
    summary.loc[summary.scope.eq("Overall")].to_csv(OUTPUT_DIR / "overall_summary.csv", index=False, encoding="utf-8-sig")
    summary.loc[summary.scope.ne("Overall")].to_csv(OUTPUT_DIR / "summary_by_sport.csv", index=False, encoding="utf-8-sig")
    overall = summary.loc[summary.scope.eq("Overall")].set_index("color")
    pd.DataFrame([{"color": name, "aliases": aliases, "interpretation_note": note,
                   "included_answer_count": int(overall.loc[name, "selected_count"])}
                  for name, aliases, _, note in COLOR_RULES]).to_csv(
        OUTPUT_DIR / "color_dictionary.csv", index=False, encoding="utf-8-sig")
    return summary


def plot_summaries(summary, sports):
    # 3. Overall bars use the named colors; sport comparisons use consistent sport colors.
    names = [rule[0] for rule in COLOR_RULES]
    overall = summary.loc[summary.scope.eq("Overall")].set_index("color").loc[names]
    fig, ax = plt.subplots(figsize=(11, 8))
    bars = ax.barh(names, overall.percent_of_included_answers, color=[rule[2] for rule in COLOR_RULES], edgecolor="#777777", linewidth=.6)
    ax.bar_label(bars, labels=[f"{r.percent_of_included_answers:.1f}% ({r.selected_count})" for r in overall.itertuples()], padding=4, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("Percentage of included training answers")
    ax.set_title(f"Color associations - overall (n={int(overall.included_response_count.iloc[0])})")
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    fig.text(.5, .015, "One answer may name several colors. Blank, nonspecific and unmatched answers excluded.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(OUTPUT_DIR / "colors_overall.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    height = .8 / len(sports)
    for ax, subset in zip(axes, [names[:8], names[8:]]):
        for i, sport in enumerate(sports):
            table = summary.loc[summary.scope.eq(sport)].set_index("color").loc[subset]
            positions = [j - .4 + height / 2 + i*height for j in range(len(subset))]
            bars = ax.barh(positions, table.percent_of_included_answers, height=height,
                           color=SPORT_COLORS[i % len(SPORT_COLORS)],
                           label=f"{sport} (n={int(table.included_response_count.iloc[0])})")
            ax.bar_label(bars, labels=[f"{r.percent_of_included_answers:.1f}% ({r.selected_count})" for r in table.itertuples()], padding=3, fontsize=9)
        ax.set_yticks(range(len(subset)), subset)
        ax.invert_yaxis()
        ax.set_xlim(0, 108)
        ax.set_xticks(range(0, 101, 20))
        ax.set_xlabel("Percentage of included answers within sport")
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Color associations by sport - training only", fontsize=16)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .955), ncol=len(sports), frameon=False)
    fig.text(.5, .015, "Each color counted once per answer. Blank, nonspecific and unmatched answers excluded.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, .91), w_pad=3)
    fig.savefig(OUTPUT_DIR / "colors_by_sport.png", dpi=160)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_FILE, dtype=str, keep_default_na=False)
    sports = sorted(df.label.unique())
    selections, statuses = parse_answers(df)
    summary = summarize(df, selections, statuses, sports)
    plot_summaries(summary, sports)
    print(summary[["scope", "included_response_count", "blank_count", "nonspecific_count", "unmatched_count"]].drop_duplicates().to_string(index=False))
    print(summary.pivot(index="color", columns="scope", values="percent_of_included_answers").round(1).to_string())
    print(f"\nSaved all outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
