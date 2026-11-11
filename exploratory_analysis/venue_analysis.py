"""Explore the multiple-choice venue question using the training split only.

Percentages use valid responses, not the number of selected options.
Missing answers remain missing; the input CSV is never changed.
"""

from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data_splits" / "train.csv"
OUTPUT_DIR = PROJECT_DIR / "distribution_results" / "venue"
COLUMN = "Where might you play this sport?"
OPTIONS = [
    "Indoor facility", "Outdoor public space", "School or university facility",
    "Community recreation facility", "Private sports facility",
    "At home or nearby", "Others",
]
COLORS = ["#367a9c", "#ce8133", "#48865c"]


def parse_answers(df):
    # 1. Split each multiple-choice answer into a set of distinct venues.
    # A missing response is not the same as selecting none of the venues.
    lookup = {option.casefold(): option for option in OPTIONS}
    selections = []
    statuses = []
    audits = []
    for row_number, row in df.iterrows():
        raw = row[COLUMN]
        text = " ".join(raw.split())
        status = "valid"
        selected = None
        reason = ""
        if text == "":
            status = "missing"
            reason = "Missing answer"
        else:
            tokens = [" ".join(token.split()).casefold() for token in raw.split(",")]
            unknown = [token for token in tokens if token not in lookup]
            if unknown:
                status = "unrecognized"
                reason = "Unrecognized option; entire answer excluded: " + repr(unknown)
            else:
                selected = {lookup[token] for token in tokens}
                if len(selected) < len(tokens):
                    reason = "Repeated option counted once; answer retained"
        selections.append(selected)
        statuses.append(status)
        if status != "valid":
            audits.append({"csv_row": row_number + 2, "id": row["id"],
                           "label": row["label"], "raw_answer": raw, "reason": reason})
    pd.DataFrame(audits, columns=["csv_row", "id", "label", "raw_answer", "reason"]).to_csv(
        OUTPUT_DIR / "excluded_answers.csv", index=False, encoding="utf-8-sig")
    return pd.Series(selections, index=df.index), pd.Series(statuses, index=df.index)


def summarize(df, selections, statuses, sports):
    # 2. Summarize each option separately, overall and within each sport.
    # Multi-select percentages can add up to more than 100%.
    option_rows, number_rows = [], []
    scopes = {"Overall": pd.Series(True, index=df.index)}
    scopes.update({sport: df["label"].eq(sport) for sport in sports})
    for scope, mask in scopes.items():
        answers = selections.loc[mask & statuses.eq("valid")]
        number_selected = answers.map(len)
        valid_count = len(answers)
        for option in OPTIONS:
            count = int(answers.map(lambda answer: option in answer).sum())
            option_rows.append({
                "scope": scope, "venue": option, "selected_count": count,
                "total_response_count": int(mask.sum()),
                "missing_count": int((mask & statuses.eq("missing")).sum()),
                "missing_percent": (mask & statuses.eq("missing")).sum() / mask.sum() * 100,
                "excluded_count": int((mask & statuses.ne("valid")).sum()),
                "valid_response_count": valid_count,
                "percent_of_valid": count / valid_count * 100 if valid_count else float("nan"),
            })
        for number in range(1, len(OPTIONS) + 1):
            count = int(number_selected.eq(number).sum())
            number_rows.append({"scope": scope, "venues_selected": number, "count": count,
                                "percent_of_valid": count / valid_count * 100 if valid_count else float("nan")})
    summary = pd.DataFrame(option_rows)
    summary.loc[summary.scope.eq("Overall")].to_csv(
        OUTPUT_DIR / "overall_summary.csv", index=False, encoding="utf-8-sig")
    summary.loc[summary.scope.ne("Overall")].to_csv(
        OUTPUT_DIR / "summary_by_sport.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(number_rows).to_csv(
        OUTPUT_DIR / "number_of_venues_selected.csv", index=False, encoding="utf-8-sig")
    return summary


def plot_summaries(summary, sports):
    # 3. Use horizontal bars to keep the venue names readable.
    overall = summary.loc[summary.scope.eq("Overall")].set_index("venue").loc[OPTIONS]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.barh(OPTIONS, overall.percent_of_valid, color=COLORS[0])
    ax.bar_label(bars, labels=[f"{row.percent_of_valid:.1f}% ({row.selected_count})"
                             for row in overall.itertuples()], padding=4, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("Percentage of valid training responses")
    ax.set_title(f"Where might you play this sport? Overall (n={int(overall.valid_response_count.iloc[0])})")
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    fig.text(.5, .015, "Multiple selections allowed; percentages do not sum to 100%. Missing answers excluded.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(OUTPUT_DIR / "venues_overall.png", dpi=160)
    plt.close(fig)

    # 4. Compare sports using percentages with sport-specific denominators.
    fig, ax = plt.subplots(figsize=(12, 7))
    height = .8 / len(sports)
    for i, sport in enumerate(sports):
        table = summary.loc[summary.scope.eq(sport)].set_index("venue").loc[OPTIONS]
        positions = [j - .4 + height / 2 + i * height for j in range(len(OPTIONS))]
        bars = ax.barh(positions, table.percent_of_valid, height=height,
                       color=COLORS[i % len(COLORS)],
                       label=f"{sport} (n={int(table.valid_response_count.iloc[0])})")
        ax.bar_label(bars, labels=[f"{row.percent_of_valid:.1f}% ({row.selected_count})"
                                   for row in table.itertuples()], padding=3, fontsize=8)
    ax.set_yticks(range(len(OPTIONS)), OPTIONS)
    ax.invert_yaxis()
    ax.set_xlim(0, 116)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("Percentage of valid responses within each sport")
    ax.set_title("Where might you play this sport? By sport - training only", pad=48)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=len(sports), frameon=False)
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    fig.text(.5, .015, "Each response may select several venues. Missing answers excluded; no imputation.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(OUTPUT_DIR / "venues_by_sport.png", dpi=160)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_FILE, dtype=str, keep_default_na=False)
    sports = sorted(df["label"].unique())
    selections, statuses = parse_answers(df)
    summary = summarize(df, selections, statuses, sports)
    plot_summaries(summary, sports)
    print("\nVenue selection percentages:")
    print(summary.pivot(index="venue", columns="scope", values="percent_of_valid").round(1).to_string())
    print(f"\nSaved results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
