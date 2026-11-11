"""Describe three Likert questions using training responses only.

Scores 1-5 preserve answer order. Means/SDs are descriptive summaries under
an equal-spacing assumption; inspect the category percentages as well.
Missing/unrecognized answers are audited, not imputed. Correlations are
descriptive Spearman coefficients, not significance tests or causal claims.
"""

from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data_splits" / "train.csv"
OUTPUT_DIR = PROJECT_DIR / "distribution_results" / "likert"

QUESTIONS = {
    "strategy": "Success in this sport depends heavily on strategy and planning.",
    "injury": "Players must be comfortable with a high risk of injury.",
    "individual": "Individual brilliance can outweigh teamwork in this sport.",
}
ANSWERS = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]
SCORE_MAP = {answer.casefold(): score for score, answer in enumerate(ANSWERS, 1)}
MISSING_MARKERS = {"", "n/a", "na", "nan", "null", "-", "--", "[]"}


def prepare_scores(df):
    # 1. Normalize spelling case/whitespace and map only the five valid answers.
    scores = pd.DataFrame(index=df.index)
    missing = pd.DataFrame(index=df.index)
    unknown = pd.DataFrame(index=df.index)
    audits = []
    for name, column in QUESTIONS.items():
        text = df[column].str.strip().str.casefold().str.replace(r"\s+", " ", regex=True)
        missing[name] = text.isin(MISSING_MARKERS)
        scores[name] = text.map(SCORE_MAP)
        unknown[name] = scores[name].isna() & ~missing[name]
        invalid = scores[name].isna()
        audit = df.loc[invalid, ["id", "label"]].copy()
        audit["question"] = name
        audit["raw_answer"] = df.loc[invalid, column]
        audit["reason"] = "Unrecognized answer"
        audit.loc[missing.loc[invalid, name], "reason"] = "Missing answer"
        audits.append(audit)
    pd.concat(audits, ignore_index=True).to_csv(
        OUTPUT_DIR / "excluded_answers.csv", index=False, encoding="utf-8-sig")
    return scores, missing, unknown


def summarize(scores, missing, unknown, mask, scope):
    # 2. Summarize each question separately; never average different questions together.
    rows = []
    for name in QUESTIONS:
        values = scores.loc[mask, name].dropna()
        rows.append({
            "scope": scope, "question": name, "total_count": int(mask.sum()),
            "valid_count": len(values), "missing_count": int(missing.loc[mask, name].sum()),
            "unrecognized_count": int(unknown.loc[mask, name].sum()),
            "missing_percent": missing.loc[mask, name].mean() * 100,
            "mean": values.mean(), "median": values.median(), "std": values.std(),
            "q25": values.quantile(.25), "q75": values.quantile(.75),
            "min": values.min(), "max": values.max(),
        })
    return pd.DataFrame(rows)


def response_distribution(scores, mask, scope):
    # 3. Use valid responses for the percentage denominator of each question.
    rows = []
    for name in QUESTIONS:
        values = scores.loc[mask, name].dropna()
        counts = values.value_counts().reindex(range(1, 6), fill_value=0)
        for score, answer in enumerate(ANSWERS, 1):
            rows.append({"scope": scope, "question": name, "score": score,
                         "answer": answer, "count": int(counts[score]),
                         "percent_of_valid": counts[score] / len(values) * 100 if len(values) else float("nan")})
    return pd.DataFrame(rows)


def plot_distributions(distributions, sports):
    # 4. Compare sports on identical answer categories and percentage axes.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    colors = ["#367a9c", "#ce8133", "#48865c"]
    width = 0.8 / len(sports)
    for ax, name in zip(axes, QUESTIONS):
        for i, sport in enumerate(sports):
            table = distributions.loc[(distributions.scope == sport) &
                                      (distributions.question == name)].sort_values("score")
            positions = [score - .4 + width / 2 + i * width for score in range(1, 6)]
            ax.bar(positions, table.percent_of_valid, width=width,
                   color=colors[i % len(colors)], label=sport)
        ax.set_xticks(range(1, 6), ["Strongly\ndisagree", "Disagree", "Neutral", "Agree", "Strongly\nagree"], fontsize=9)
        ax.set_title(name.capitalize())
        ax.set_xlabel("Response (ordered from 1 to 5)")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Percentage of valid responses within sport")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .94), ncol=len(sports), frameon=False)
    fig.suptitle("Likert response distributions by sport - training only")
    fig.text(.5, .01, "Missing/unrecognized answers excluded separately for each question. No imputation.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .85))
    fig.savefig(OUTPUT_DIR / "distributions_by_sport.png", dpi=160)
    plt.close(fig)


def correlations(scores, scopes):
    # 5. Pairwise-complete Spearman correlations use ranks, including tied ranks.
    # Each pair can have a different sample size; save that matrix alongside rho.
    # Overall rows repeat students across sports. Do not treat these as independent
    # observations for a significance test; no p-values are calculated here.
    results = {}
    pair_rows = []
    names = list(QUESTIONS)
    for scope, mask in scopes.items():
        subset = scores.loc[mask]
        corr = subset.corr(method="spearman", min_periods=3)
        present = subset.notna().astype(int)
        pair_n = present.T.dot(present)
        corr.to_csv(OUTPUT_DIR / f"spearman_{scope.lower()}.csv", encoding="utf-8-sig")
        pair_n.to_csv(OUTPUT_DIR / f"pair_counts_{scope.lower()}.csv", encoding="utf-8-sig")
        results[scope] = corr
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pair_rows.append({"scope": scope, "question_1": names[i], "question_2": names[j],
                                  "spearman_rho": corr.loc[names[i], names[j]],
                                  "paired_responses": pair_n.loc[names[i], names[j]]})
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(OUTPUT_DIR / "correlation_pairs.csv", index=False, encoding="utf-8-sig")

    # 6. Fixed -1 to +1 colors allow comparisons across all four panels.
    nrows = (len(results) + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(9, 4 * nrows), squeeze=False, layout="constrained")
    for ax, (scope, corr) in zip(axes.flat, results.items()):
        picture = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(3), names)
        ax.set_yticks(range(3), names)
        ax.set_title(scope)
        for i in range(3):
            for j in range(3):
                value = corr.iloc[i, j]
                ax.text(j, i, "N/A" if pd.isna(value) else f"{value:.2f}",
                        ha="center", va="center", color="white" if abs(value) > .6 else "black")
    for ax in list(axes.flat)[len(results):]:
        ax.set_visible(False)
    fig.colorbar(picture, ax=list(axes.flat), shrink=.7, label="Spearman correlation")
    fig.suptitle("Question correlations - overall and within each sport\nTraining only; pairwise available responses")
    fig.savefig(OUTPUT_DIR / "correlations.png", dpi=160)
    plt.close(fig)
    return pairs


def main():
    df = pd.read_csv(DATA_FILE, dtype=str, keep_default_na=False)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(QUESTIONS.items(), columns=["question", "original_column"]).to_csv(
        OUTPUT_DIR / "question_key.csv", index=False, encoding="utf-8-sig")
    scores, missing, unknown = prepare_scores(df)
    sports = sorted(df["label"].unique())
    scopes = {"Overall": pd.Series(True, index=df.index)}
    scopes.update({sport: df["label"].eq(sport) for sport in sports})
    summary = pd.concat([summarize(scores, missing, unknown, mask, scope)
                         for scope, mask in scopes.items()], ignore_index=True)
    summary[summary.scope.eq("Overall")].to_csv(OUTPUT_DIR / "overall_summary.csv", index=False, encoding="utf-8-sig")
    by_sport = summary[~summary.scope.eq("Overall")]
    by_sport.to_csv(OUTPUT_DIR / "summary_by_sport.csv", index=False, encoding="utf-8-sig")
    distributions = pd.concat([response_distribution(scores, mask, scope)
                               for scope, mask in scopes.items()], ignore_index=True)
    distributions.to_csv(OUTPUT_DIR / "response_distributions.csv", index=False, encoding="utf-8-sig")
    plot_distributions(distributions, sports)
    pairs = correlations(scores, scopes)
    print(f"Training data: {len(df)} rows from {df['id'].nunique()} students.")
    print("\nSummary by question and sport:")
    print(by_sport[["scope", "question", "valid_count", "missing_count", "unrecognized_count", "mean", "median", "std"]].round(3).to_string(index=False))
    print("\nDescriptive correlations (not significance tests):")
    print(pairs.round(3).to_string(index=False))
    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
