"""Explore class balance and the ease-rating distribution in the training split.

"""

from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data_splits" / "train.csv"
OUTPUT_DIR = PROJECT_DIR / "distribution_results" / "ease"

EASE_COL = (
    "How easy do you think it would be for someone with no prior experience "
    "to start playing this sport recreationally? Rate it on a scale from 0 to 10, "
    "where 0 means extremely difficult and 10 means extremely easy."
)


def compare_sports(df, ratings, valid_mask, score_values, class_counts):
    # 7. Group valid ease ratings by sport. Keep class sizes to audit exclusions.
    ease_data = pd.DataFrame({"sport": df["label"], "ease": ratings})
    ease_data = ease_data.loc[valid_mask]
    grouped = ease_data.groupby("sport")["ease"]
    summary = grouped.agg(["count", "mean", "median", "std", "min", "max"])
    summary = summary.reindex(class_counts.index)
    summary = summary.rename(columns={"count": "valid_count"})
    summary["total_count"] = class_counts
    summary["valid_count"] = summary["valid_count"].fillna(0).astype(int)
    summary["excluded_count"] = summary["total_count"] - summary["valid_count"]
    summary["q25"] = grouped.quantile(0.25)
    summary["q75"] = grouped.quantile(0.75)
    summary = summary[["total_count", "valid_count", "excluded_count",
                       "mean", "median", "std", "q25", "q75", "min", "max"]]
    summary.index.name = "sport"
    summary.to_csv(OUTPUT_DIR / "ease_by_sport_summary.csv", encoding="utf-8-sig")
    print("\nEase statistics by sport (valid responses only):")
    print(summary.round(2).to_string())

    # 8. Each column is one sport; each row is an exact score, including decimals.
    counts = pd.crosstab(ease_data["ease"], ease_data["sport"])
    counts = counts.reindex(index=score_values, columns=summary.index, fill_value=0)
    counts.index.name = "ease_rating"
    counts.to_csv(OUTPUT_DIR / "ease_by_sport_counts.csv", encoding="utf-8-sig")
    percentages = counts.div(summary["valid_count"].replace(0, float("nan")), axis=1) * 100
    percentages.to_csv(OUTPUT_DIR / "ease_by_sport_percentages.csv", encoding="utf-8-sig")

    # 9. Use the same axes and within-sport percentages for fair visual comparison.
    sports = summary.index.tolist()
    colors = ["#367a9c", "#ce8133", "#48865c"]
    fig, axes = plt.subplots(1, len(sports), figsize=(4 * len(sports), 4.2),
                             sharex=True, sharey=True, squeeze=False)
    for i, sport in enumerate(sports):
        ax = axes[0, i]
        ax.bar(score_values, percentages[sport], width=0.4,
               color=colors[i % len(colors)], edgecolor="white")
        ax.set_xticks(range(11))
        ax.set_title(f"{sport} (n={summary.loc[sport, 'valid_count']})")
        ax.set_xlabel("Ease rating")
        ax.set_xlim(-0.6, 10.6)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        if summary.loc[sport, "valid_count"] == 0:
            ax.text(0.5, 0.5, "No valid ratings", transform=ax.transAxes, ha="center")
    axes[0, 0].set_ylabel("Percentage of valid responses within sport")
    fig.suptitle("Training ease-rating distributions by sport")
    fig.text(0.5, 0.01, "0 = extremely difficult; 10 = extremely easy. Blank/invalid ratings excluded; decimals retained.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(OUTPUT_DIR / "ease_distribution_by_sport.png", dpi=160)
    plt.close(fig)

    # 10. Boxes show the middle 50%; median lines and mean dots show location.
    # Whiskers use the 1.5-IQR rule. Points outside them are not automatically errors.
    fig, ax = plt.subplots(figsize=(7, 4.8))
    box = ax.boxplot([ease_data.loc[ease_data["sport"].eq(s), "ease"] for s in sports],
                     patch_artist=True, showmeans=True,
                     meanprops={"marker": "D", "markerfacecolor": "#a32939",
                                "markeredgecolor": "#a32939", "markersize": 5},
                     medianprops={"color": "#171717", "linewidth": 2})
    for i, patch in enumerate(box["boxes"]):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.45)
    ax.set_xticks(range(1, len(sports) + 1), sports)
    ax.set(xlabel="Sport", ylabel="Ease rating", ylim=(-0.5, 10.5),
           title="Ease by sport: median, spread, and mean")
    ax.plot([], [], color="#171717", linewidth=2, label="Median")
    ax.plot([], [], marker="D", color="#a32939", linestyle="None", label="Mean")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ease_boxplot_by_sport.png", dpi=160)
    plt.close(fig)


def main():
    # 1. Read training responses.
    df = pd.read_csv(DATA_FILE, dtype=str, keep_default_na=False)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Training data: {len(df)} rows from {df['id'].nunique()} students.")

    class_counts = df["label"].value_counts().sort_index()

    # 3. Convert rating text to numbers for counting, keeping the raw answers.
    raw_ratings = df[EASE_COL].str.strip()
    ratings = pd.to_numeric(raw_ratings, errors="coerce")
    valid_mask = ratings.between(0, 10)
    valid_ratings = ratings[valid_mask]
    if valid_ratings.empty:
        raise ValueError("No valid numeric ease ratings between 0 and 10.")

    # 4. Record answers excluded from this plot, rather than silently hiding them.
    excluded = df.loc[~valid_mask, ["id", EASE_COL]].copy()
    excluded["reason"] = "Not a numeric rating from 0 to 10"
    excluded.loc[raw_ratings.loc[excluded.index].eq(""), "reason"] = "Blank response"
    excluded.to_csv(OUTPUT_DIR / "excluded_answers.csv",
                    index=False, encoding="utf-8-sig")

    # 5. Count observed scores, retaining decimals and integer scores with zero counts.
    score_values = sorted(set(range(11)) | set(valid_ratings))
    counts = valid_ratings.value_counts().reindex(score_values, fill_value=0)
    counts.index.name = "ease_rating"
    distribution = counts.to_frame("count")
    distribution["percent_of_valid"] = counts / len(valid_ratings) * 100
    distribution.to_csv(OUTPUT_DIR / "ease_distribution.csv", encoding="utf-8-sig")
    summary = valid_ratings.describe()
    summary.to_csv(OUTPUT_DIR / "ease_summary.csv", header=["value"])
    print(f"\nEase: {len(valid_ratings)} valid ratings, {len(excluded)} excluded answers.")
    print(distribution.round(2).to_string())
    print("\nEase summary:")
    print(summary.round(2).to_string())

    # 6. Plot exact score frequencies without rounding decimal responses.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(counts.index, counts.values, width=0.4, color="#367a9c", edgecolor="white")
    ax.set_xticks(score_values, [f"{x:g}" for x in score_values])
    ax.set(xlabel="Ease rating (0 = extremely difficult, 10 = extremely easy)",
           ylabel="Number of training responses",
           title=f"Ease-rating distribution: {len(valid_ratings)} valid responses")
    ax.bar_label(ax.containers[0], fontsize=9)
    ax.margins(y=0.15)
    fig.text(0.5, 0.01, f"Excluded from this plot: {len(excluded)} blank or invalid answers.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUTPUT_DIR / "ease_distribution.png", dpi=160)
    plt.close(fig)
    compare_sports(df, ratings, valid_mask, score_values, class_counts)
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
