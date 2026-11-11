"""Explore fixed-choice music associations using training responses only.

Primary summaries retain ambiguous answers as recorded. A sensitivity table
shows the effect of excluding No particular type combined with other options.
Blank cells are excluded; there are no free-text missing-marker rules.
"""

from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data_splits" / "train.csv"
OUTPUT_DIR = PROJECT_DIR / "distribution_results" / "music"
COLUMN = "What types of music do you most associate with this sport?"
OPTIONS = ["Pop", "Rock", "Hip-hop or rap", "Electronic or dance", "Country",
           "Latin", "Classical or jazz", "Folk or traditional", "Other", "No particular type"]
COLORS = ["#367a9c", "#ce8133", "#48865c"]


def prepare_answers(df):
    # 1. Keep a blank response distinct from an explicit No particular type answer.
    blank = df[COLUMN].str.strip().eq("")
    answers = df[COLUMN].map(lambda text: {item.strip() for item in text.split(",")} if text.strip() else set())
    unexpected = set().union(*answers) - set(OPTIONS)
    if unexpected:
        raise ValueError(f"Unexpected music options: {sorted(unexpected)}")
    ambiguous = answers.map(lambda selected: "No particular type" in selected and len(selected) > 1)

    # 2. Save excluded blanks separately from ambiguous answers retained for exploration.
    for mask, filename, reason in [
        (blank, "excluded_answers.csv", "Blank answer; excluded from music percentages"),
        (ambiguous, "flagged_answers.csv", "No particular type with another option; retained in primary summaries"),
    ]:
        audit = df.loc[mask, ["id", "label"]].copy()
        audit.insert(0, "csv_row", audit.index + 2)
        audit["raw_answer"] = df.loc[mask, COLUMN]
        audit["reason"] = reason
        audit.to_csv(OUTPUT_DIR / filename, index=False, encoding="utf-8-sig")
    return answers, blank, ambiguous


def summarize(df, answers, blank, ambiguous, sports):
    # 3. Count each option once per answer, overall and within each sport.
    scopes = {"Overall": pd.Series(True, index=df.index)}
    scopes.update({sport: df["label"].eq(sport) for sport in sports})
    rows, comparisons = [], []
    for scope, mask in scopes.items():
        primary = answers.loc[mask & ~blank]
        without_ambiguous = answers.loc[mask & ~blank & ~ambiguous]
        n, clean_n = len(primary), len(without_ambiguous)
        for option in OPTIONS:
            count = int(primary.map(lambda selected: option in selected).sum())
            clean_count = int(without_ambiguous.map(lambda selected: option in selected).sum())
            percent = count / n * 100 if n else float("nan")
            clean_percent = clean_count / clean_n * 100 if clean_n else float("nan")
            rows.append({
                "scope": scope, "music_type": option, "selected_count": count,
                "total_response_count": int(mask.sum()), "answered_count": n,
                "missing_count": int((mask & blank).sum()),
                "missing_percent": (mask & blank).sum() / mask.sum() * 100,
                "ambiguous_count": int((mask & ambiguous).sum()),
                "ambiguous_percent_of_answered": (mask & ambiguous).sum() / n * 100 if n else float("nan"),
                "percent_of_answered": percent,
            })
            # Both the selected count and denominator change when answers are excluded.
            comparisons.append({
                "scope": scope, "music_type": option,
                "with_ambiguous_count": count, "with_ambiguous_denominator": n,
                "with_ambiguous_percent": percent,
                "without_ambiguous_count": clean_count, "without_ambiguous_denominator": clean_n,
                "without_ambiguous_percent": clean_percent,
                "change_percentage_points": clean_percent - percent,
            })
    summary = pd.DataFrame(rows)
    summary.loc[summary.scope.eq("Overall")].to_csv(
        OUTPUT_DIR / "overall_summary.csv", index=False, encoding="utf-8-sig")
    summary.loc[summary.scope.ne("Overall")].to_csv(
        OUTPUT_DIR / "summary_by_sport.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(comparisons).to_csv(
        OUTPUT_DIR / "ambiguity_sensitivity.csv", index=False, encoding="utf-8-sig")
    return summary


def plot_summaries(summary, sports):
    # 4. Show percentages and counts; use nonblank responses as the denominator.
    overall = summary.loc[summary.scope.eq("Overall")].set_index("music_type").loc[OPTIONS]
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(OPTIONS, overall.percent_of_answered, color=COLORS[0])
    ax.bar_label(bars, labels=[f"{row.percent_of_answered:.1f}% ({row.selected_count})"
                             for row in overall.itertuples()], padding=4, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("Percentage of nonblank training responses")
    ax.set_title(f"Music associated with sports - overall (n={int(overall.answered_count.iloc[0])})")
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    fig.text(.5, .015, "Multiple selections allowed. Blank answers excluded;  ambiguous answers retained as recorded.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(OUTPUT_DIR / "music_overall.png", dpi=160)
    plt.close(fig)

    # 5. Keep the same option order, sport colors, and percentage scale for comparison.
    fig, ax = plt.subplots(figsize=(13, 10))
    height = .8 / len(sports)
    for i, sport in enumerate(sports):
        table = summary.loc[summary.scope.eq(sport)].set_index("music_type").loc[OPTIONS]
        positions = [j - .4 + height / 2 + i * height for j in range(len(OPTIONS))]
        bars = ax.barh(positions, table.percent_of_answered, height=height,
                       color=COLORS[i % len(COLORS)],
                       label=f"{sport} (n={int(table.answered_count.iloc[0])})")
        ax.bar_label(bars, labels=[f"{row.percent_of_answered:.1f}% ({row.selected_count})"
                                  for row in table.itertuples()], padding=3, fontsize=9)
    ax.set_yticks(range(len(OPTIONS)), OPTIONS)
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("Percentage of nonblank responses within each sport")
    ax.set_title("Music associations by sport - training only", pad=48)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=len(sports), frameon=False)
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    fig.text(.5, .015, "Multiple selections allowed. Blank answers excluded; ambiguous answers retained as recorded.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(OUTPUT_DIR / "music_by_sport.png", dpi=160)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_FILE, dtype=str, keep_default_na=False)
    sports = sorted(df["label"].unique())
    answers, blank, ambiguous = prepare_answers(df)
    summary = summarize(df, answers, blank, ambiguous, sports)
    plot_summaries(summary, sports)
    print(summary[["scope", "answered_count", "missing_count", "ambiguous_count"]].drop_duplicates().to_string(index=False))
    print("\nSelection percentages (ambiguous answers retained):")
    print(summary.pivot(index="music_type", columns="scope", values="percent_of_answered").loc[OPTIONS].round(1).to_string())
    print(f"\nSaved results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
