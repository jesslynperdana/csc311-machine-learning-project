"""Explore season selections in the training split only.

The fixed-choice question has no free-text missing markers.
Actual blank CSV cells are counted separately and excluded from percentages.
"""

from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data_splits" / "train.csv"
OUTPUT_DIR = PROJECT_DIR / "distribution_results" / "season"
COLUMN = "During which seasons would you be most likely to play this sport recreationally in Toronto?"
SEASONS = ["Spring", "Summer", "Fall", "Winter"]
COLORS = ["#367a9c", "#ce8133", "#48865c"]


def summarize(df):
    # 1. Split multiple selections; keep blank answers out of the denominator.
    text = df[COLUMN].str.strip()
    blank = text.eq("")
    answers = text.loc[~blank].map(lambda value: {item.strip() for item in value.split(",")})
    # Fixed options are a data-format requirement, not free-text missing markers.
    unexpected = set().union(*answers) - set(SEASONS)
    if unexpected:
        raise ValueError(f"Unexpected season options: {sorted(unexpected)}")
    df.loc[blank, ["id", "label", COLUMN]].to_csv(
        OUTPUT_DIR / "excluded_answers.csv", index=False, encoding="utf-8-sig")

    # 2. Count each season once per response, overall and within each sport.
    sports = sorted(df["label"].unique())
    scopes = {"Overall": pd.Series(True, index=df.index)}
    scopes.update({sport: df["label"].eq(sport) for sport in sports})
    summary_rows, combination_rows, number_rows = [], [], []
    for scope, mask in scopes.items():
        selected = answers.loc[mask.loc[answers.index]]
        n = len(selected)
        counts = selected.map(len)
        for season in SEASONS:
            count = int(selected.map(lambda response: season in response).sum())
            summary_rows.append({"scope": scope, "season": season, "selected_count": count,
                                 "total_response_count": int(mask.sum()),
                                 "missing_count": int((mask & blank).sum()),
                                 "missing_percent": (mask & blank).sum() / mask.sum() * 100,
                                 "valid_response_count": n,
                                 "percent_of_valid": count / n * 100 if n else float("nan")})
        # Save complete combinations as well as individual season frequencies.
        combinations = selected.map(lambda response: ",".join(s for s in SEASONS if s in response))
        for combination, count in combinations.value_counts().items():
            combination_rows.append({"scope": scope, "combination": combination,
                                     "count": count, "percent_of_valid": count / n * 100})
        for number in range(1, 5):
            count = int(counts.eq(number).sum())
            number_rows.append({"scope": scope, "seasons_selected": number, "count": count,
                                "percent_of_valid": count / n * 100 if n else float("nan")})

    summary = pd.DataFrame(summary_rows)
    summary.loc[summary.scope.eq("Overall")].to_csv(
        OUTPUT_DIR / "overall_summary.csv", index=False, encoding="utf-8-sig")
    summary.loc[summary.scope.ne("Overall")].to_csv(
        OUTPUT_DIR / "summary_by_sport.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(combination_rows).to_csv(
        OUTPUT_DIR / "season_combinations.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(number_rows).to_csv(
        OUTPUT_DIR / "number_of_seasons_selected.csv", index=False, encoding="utf-8-sig")
    return summary, sports


def plot_summaries(summary, sports):
    # 3. Show seasons in calendar order on a consistent percentage scale.
    overall = summary.loc[summary.scope.eq("Overall")].set_index("season").loc[SEASONS]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(SEASONS, overall.percent_of_valid, color=COLORS[0], width=.6)
    ax.bar_label(bars, labels=[f"{row.percent_of_valid:.1f}% ({row.selected_count})"
                             for row in overall.itertuples()], padding=4)
    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Percentage of valid training responses")
    ax.set_title(f"Seasons for recreational play in Toronto (n={int(overall.valid_response_count.iloc[0])})")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    fig.text(.5, .015, "Multiple selections allowed; blank answers excluded.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(OUTPUT_DIR / "seasons_overall.png", dpi=160)
    plt.close(fig)

    # 4. Compare all sports within each season using their valid-answer totals.
    fig, ax = plt.subplots(figsize=(14, 6))
    width = .8 / len(sports)
    for i, sport in enumerate(sports):
        table = summary.loc[summary.scope.eq(sport)].set_index("season").loc[SEASONS]
        positions = [j - .4 + width / 2 + i * width for j in range(len(SEASONS))]
        bars = ax.bar(positions, table.percent_of_valid, width=width, color=COLORS[i % len(COLORS)],
                      label=f"{sport} (n={int(table.valid_response_count.iloc[0])})")
        ax.bar_label(bars, labels=[f"{row.percent_of_valid:.1f}% ({row.selected_count})"
                                   for row in table.itertuples()], padding=4, fontsize=9)
    ax.set_xticks(range(len(SEASONS)), SEASONS)
    ax.set_ylim(0, 108)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Percentage of valid responses within each sport")
    ax.set_title("Seasons for recreational play in Toronto - training only", pad=46)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=len(sports), frameon=False)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    fig.text(.5, .015, "Multiple selections allowed; percentages do not sum to 100%. Blank answers excluded.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(OUTPUT_DIR / "seasons_by_sport.png", dpi=160)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_FILE, dtype=str, keep_default_na=False)
    summary, sports = summarize(df)
    plot_summaries(summary, sports)
    print("\nSeason selection percentages:")
    print(summary.pivot(index="season", columns="scope", values="percent_of_valid").loc[SEASONS].round(1).to_string())
    print(f"\nSaved results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
