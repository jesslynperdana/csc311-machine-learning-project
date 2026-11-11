"""Explore common song-title associations in the saved training split only.

Candidate titles and explicit variants were reviewed in training answers.
Keep titles mentioned in at least MIN_SONG_COUNT training answers, across all
sports. Selection uses frequency, not labels. Count a song once per answer.
This is title-level matching: artists/recordings with the same title are not
disambiguated. Do not infer a title from an artist, genre or redacted wording.

The plots describe answers matching at least one selected common title.
Summary CSVs also report percentages of ALL training answers in each group,
so unequal coverage between sports remains visible. Exclusion affects this
question's summary only; the original rows and split files are never changed.
"""

from collections import Counter
from pathlib import Path
import re
import unicodedata

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data_splits" / "train.csv"
OUTPUT_DIR = PROJECT_DIR / "distribution_results" / "song"
COLUMN = "Which song do you associate with this sport?"
MIN_SONG_COUNT = 5
COLORS = ["#367a9c", "#ce8133", "#48865c"]

# This short candidate list covers repeated, identifiable titles in training.
# Explicit observed spelling variants are merged; no fuzzy matching is used.
# Artist-only answers and anonymized song names are not reconstructed.
SONG_RULES = {
    "Waka Waka": ["waka waka", "waka wake", "wake waka", "this time for africa", "it's time for africa"],
    "Take Me Out to the Ball Game": ["take me out to the ball game", "take me out to the ballgame", "take me out to the ball gam"],
    "We Will Rock You": ["we will rock you"],
    "Wavin' Flag": ["wavin flag", "wavin flags", "waving flag", "waving flags"],
    "Sirius": ["sirius"],
    "Space Jam": ["space jam"],
    "We Are One": ["we are one"],
    "We Are the Champions": ["we are the champions"],
    "Roundball Rock": ["roundball rock"],
    "See You Again": ["see you again"],
    "Sweet Caroline": ["sweet caroline"],
    "Centerfield": ["centerfield"],
    "Sicko Mode": ["sicko mode"],
    "Forever": ["forever"],
    "Seven Nation Army": ["seven nation army"],
    "Remember the Name": ["remember the name"],
    "Hall of Fame": ["hall of fame", "hall of the fame"],
    "Dai Dai": ["dai dai", "daidai"],
    "Love Me Again": ["love me again"],
    "God's Plan": ["god's plan"],
    "Wonderwall": ["wonderwall"],
}


def normalize(raw):
    """Normalize case, spacing and punctuation without guessing song titles."""
    text = unicodedata.normalize("NFKC", raw).casefold()
    # A URL alone is not a title, and URL characters must not create matches.
    text = re.sub(r"https?://\S+", " ", text)
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


# Match only complete answers, so 'no idea, maybe Waka Waka' retains its title.
NO_SPECIFIC_SONG = {normalize(text) for text in [
    "none", "n/a", "na", "nil", "no", "nothing", "unsure", "unknown", "idk",
    "no idea", "i have no idea", "really no idea", "no song", "no choice", "-", "#NAME?", "emm",
    "no particular song", "no perticular song", "no song in particular",
    "no particular song comes to mind", "none in particular", "nothing really", "nothing specific",
    "nothing in particular, something with a good beat?", "don't have", "cant think of",
    "i don't know", "i dont know", "i don't no", "sorry i don't know", "i don't know any songs",
    "none i do not know of any", "n/a since i know nothing about it at all",
    "none i basically know nothing about the sport", "i do not associate a song with this sport",
]}


# 1. Reusable parsing: no labels, summaries, file writes or plots are needed.
def make_matcher():
    lookup = {}
    for title, variants in SONG_RULES.items():
        for variant in variants:
            alias = normalize(variant)
            if alias in lookup and lookup[alias] != title:
                raise ValueError(f"Ambiguous song alias: {alias}")
            lookup[alias] = title
    ordered = sorted(lookup, key=lambda term: (-len(term), term))
    pattern = "|".join(r"(?<!\w)" + re.escape(term) + r"(?!\w)" for term in ordered)
    return re.compile(pattern), lookup


def parse_answer(raw, matcher, lookup):
    """Return candidate titles and their matched phrases, without a cutoff."""
    matches = {}
    for match in matcher.finditer(normalize(raw)):
        matches.setdefault(lookup[match.group()], set()).add(match.group())
    return matches


# 2. Select common titles and prepare inspectable row-level results.
def analyze(df):
    matcher, lookup = make_matcher()
    parsed = [parse_answer(raw, matcher, lookup) for raw in df[COLUMN]]
    counts = Counter(title for answer in parsed for title in answer)
    selected = sorted((title for title in SONG_RULES if counts[title] >= MIN_SONG_COUNT),
                      key=lambda title: (-counts[title], title))
    rows = []
    for (index, row), matched in zip(df.iterrows(), parsed):
        raw = row[COLUMN]
        retained = [title for title in selected if title in matched]
        if retained:
            status, reason = "included", ""
        elif not raw.strip():
            status, reason = "blank", "Blank answer"
        elif normalize(raw) in NO_SPECIFIC_SONG:
            status, reason = "no_specific_song", "No specific song / does not know / missing or error expression"
        elif matched:
            status = "below_cutoff"
            reason = f"Recognized candidate title(s), but none reaches {MIN_SONG_COUNT} training answers"
        elif re.search(r"\[(?:a |an |this |Team Name)", raw, re.I):
            status, reason = "obscured_title", "No selected title identified; wording contains an anonymized placeholder"
        else:
            status, reason = "no_selected_title", "No selected common title identified; may be a rare song, vague answer or unrecognized text"
        rows.append({"csv_row": index + 2, "id": row["id"], "label": row["label"],
                     "raw_answer": raw, "included": bool(retained), "status": status,
                     "matched_candidate_songs": " | ".join(sorted(matched)),
                     "selected_songs": " | ".join(retained),
                     "matched_phrases": " | ".join(sorted(phrase for title in retained for phrase in matched[title])),
                     "reason": reason})
    audit = pd.DataFrame(rows)
    dictionary = pd.DataFrame([
        {"song": title, "aliases": " | ".join(SONG_RULES[title]), "training_answer_count": counts[title],
         "selected": title in selected, "minimum_training_count": MIN_SONG_COUNT,
         "observed_answers": " | ".join(sorted(set(df.loc[[title in answer for answer in parsed], COLUMN])))}
        for title in sorted(SONG_RULES, key=lambda title: (-counts[title], title))
    ])
    return parsed, selected, audit, dictionary


def summarize(df, parsed, selected, audit):
    sports = sorted(df.label.unique())
    rows = []
    for scope in ["Overall", *sports]:
        mask = pd.Series(True, index=df.index) if scope == "Overall" else df.label.eq(scope)
        # Audit and input rows stay in the same order throughout this analysis.
        positions = [i for i, keep in enumerate(mask) if keep]
        subset = audit.iloc[positions]
        total, included = len(subset), int(subset.included.sum())
        for title in selected:
            count = sum(title in parsed[i] for i in positions)
            rows.append({"scope": scope, "song": title, "answer_count": count,
                         "total_count": total, "included_count": included,
                         "excluded_count": total - included,
                         "included_percent": 100 * included / total if total else 0,
                         "blank_count": int(subset.status.eq("blank").sum()),
                         "no_specific_song_count": int(subset.status.eq("no_specific_song").sum()),
                         "below_cutoff_count": int(subset.status.eq("below_cutoff").sum()),
                         "obscured_title_count": int(subset.status.eq("obscured_title").sum()),
                         "no_selected_title_count": int(subset.status.eq("no_selected_title").sum()),
                         "percent_of_included": 100 * count / included if included else 0,
                         "percent_of_all_answers": 100 * count / total if total else 0})
    return pd.DataFrame(rows), sports


# 3. Plot the summaries; matching and exclusion rules stay outside plotting.
def plot_summaries(summary, selected, sports):
    overall = summary.loc[summary.scope.eq("Overall")].set_index("song").loc[selected]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(selected, overall.percent_of_included, color=COLORS[0])
    ax.invert_yaxis()
    upper = max(10, overall.percent_of_included.max() * 1.22)
    ax.set_xlim(0, upper)
    for y, row in enumerate(overall.itertuples()):
        ax.text(row.percent_of_included + upper * .012, y,
                f"{row.percent_of_included:.1f}% ({row.answer_count})", va="center", fontsize=10)
    retained = int(overall.iloc[0].included_count)
    total = int(overall.iloc[0].total_count)
    ax.set_title(f"Common song associations - training only\n{retained}/{total} answers match at least one selected title")
    ax.set_xlabel("Percentage of answers matching a selected common song")
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    fig.text(.5, .02, f"Titles mentioned in at least {MIN_SONG_COUNT} training answers. Multiple songs per answer are allowed; percentages may sum above 100%.",
             ha="center", fontsize=9)
    fig.subplots_adjust(left=.27, right=.97, bottom=.14, top=.88)
    fig.savefig(OUTPUT_DIR / "songs_overall.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(sports), figsize=(18, 7), sharey=True, squeeze=False)
    upper = max(10, summary.loc[summary.scope.ne("Overall"), "percent_of_included"].max() * 1.4)
    for ax, sport, fill in zip(axes[0], sports, COLORS):
        group = summary.loc[summary.scope.eq(sport)].set_index("song").loc[selected]
        ax.barh(selected, group.percent_of_included, color=fill)
        ax.set_xlim(0, upper)
        ax.set_title(f"{sport}\n{int(group.iloc[0].included_count)}/{int(group.iloc[0].total_count)} answers retained")
        ax.set_xlabel("% of retained answers within sport")
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
        for y, row in enumerate(group.itertuples()):
            ax.text(row.percent_of_included + upper * .012, y,
                    f"{row.percent_of_included:.1f}% ({row.answer_count})", va="center", fontsize=9)
    axes[0, 0].invert_yaxis()
    fig.suptitle("Common song associations by sport - training only", fontsize=15)
    fig.text(.5, .025, "Percentages describe retained answers, not every student. CSV summaries also give percentages of all answers. Multiple titles per answer may count.",
             ha="center", fontsize=10)
    fig.subplots_adjust(left=.19, right=.98, bottom=.13, top=.85, wspace=.12)
    fig.savefig(OUTPUT_DIR / "songs_by_sport.png", dpi=160)
    plt.close(fig)


def main():
    df = pd.read_csv(DATA_FILE, dtype=str, keep_default_na=False)
    parsed, selected, audit, dictionary = analyze(df)
    if not selected:
        raise ValueError("No candidate songs meet MIN_SONG_COUNT; inspect the training answers and threshold.")
    summary, sports = summarize(df, parsed, selected, audit)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, table in [
        ("song_dictionary.csv", dictionary),
        ("parsed_answers.csv", audit),
        ("excluded_answers.csv", audit.loc[~audit.included]),
        ("overall_summary.csv", summary.loc[summary.scope.eq("Overall")]),
        ("summary_by_sport.csv", summary.loc[summary.scope.ne("Overall")]),
    ]:
        table.to_csv(OUTPUT_DIR / filename, index=False, encoding="utf-8-sig")
    plot_summaries(summary, selected, sports)
    print(f"Selected {len(selected)} titles with at least {MIN_SONG_COUNT} training answers.")
    print(summary.drop_duplicates("scope")[["scope", "total_count", "included_count", "excluded_count"]].to_string(index=False))
    print(summary.loc[summary.scope.eq("Overall"), ["song", "answer_count"]].to_string(index=False))
    print(f"Outputs: {OUTPUT_DIR}")
    print("Only train.csv was read. Exclusion does not delete source rows.")


if __name__ == "__main__":
    main()
