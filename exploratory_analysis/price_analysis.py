"""Explore reported ticket willingness-to-pay using the training split only.

A single numeric amount is retained. A sole two-number 'to' or hyphen range
is replaced by its midpoint. No digits, rating scales, multiple scenarios,
foreign currencies and invalid amounts are excluded from the CAD summary.
Unspecified dollars/amounts are assumed CAD; no exchange conversion is done.
Written number words are not converted under the current numeric-only rule.
"""

from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data_splits" / "train.csv"
OUTPUT_DIR = PROJECT_DIR / "distribution_results" / "price"
COLUMN = "How much would you be willing to pay for a decent seat at a professional competition of this sport?"
COLORS = ["#367a9c", "#ce8133", "#48865c"]
NUMBER = re.compile(r"\d+(?:\.\d+)?\s*k?", re.I)
BIN_EDGES = [-.5, 0, 50, 100, 200, 300, 500, 1000, 2000, 5000, np.inf]
BIN_LABELS = ["0", "(0, 50]", "(50, 100]", "(100, 200]", "(200, 300]",
              "(300, 500]", "(500, 1000]", "(1000, 2000]", "(2000, 5000]", "> 5000"]


def normalize(raw):
    text = unicodedata.normalize("NFKC", raw).casefold()
    text = re.sub(r"[–—−]", "-", text)
    # Remove valid thousands separators without merging arbitrary adjacent amounts.
    text = re.sub(r"(?<!\d)\d{1,3}(?:,\d{3})+(?!\d)", lambda m: m.group().replace(",", ""), text)
    text = re.sub(r"(?<!\d)\d{1,3}(?:[ \u00a0]\d{3})+(?!\d)", lambda m: re.sub(r"\s", "", m.group()), text)
    return text


def parse_answer(raw):
    text = normalize(raw)
    has_digits = bool(re.search(r"\d", text))
    result = {"has_digits": has_digits, "included": False, "amount": float("nan"),
              "currency": "CAD (assumed)", "method": "", "reason": "", "notes": "",
              "range_lower": float("nan"), "range_upper": float("nan")}
    if not has_digits:
        result["reason"] = "Blank answer" if not text.strip() else "No numeric digits; no amount inferred"
        return result

    # A numeric rating is not a monetary answer, even if it contains a range.
    if re.search(r"\bscale\b|\bout of\s*10\b|\b[1-9]\s*/\s*10\b", text):
        result["reason"] = "Rating scale rather than a ticket amount"
        return result
    if re.search(r"€|\beur\b|\beuros?\b|£|\bgbp\b|\busd\b|us\$|\bus dollars?\b|\brmb\b|\bcny\b|¥", text):
        result["currency"] = "Explicit non-CAD currency"
        result["reason"] = "Foreign currency; excluded from the CAD summary without conversion"
        return result
    if re.search(r"\bcad\b|cad(?=\s*\d)|(?<=\d)cad\b|\bcanadian\b", text):
        result["currency"] = "CAD (explicit)"

    matches = list(NUMBER.finditer(text))
    values = [float(m.group().strip().rstrip("k").strip()) * (1000 if m.group().strip().endswith("k") else 1)
              for m in matches]
    notes = []
    if any(m.group().strip().endswith("k") for m in matches):
        notes.append("k expanded to thousands")
    if matches and re.search(r"-\s*\$?\s*$", text[:matches[0].start()]):
        result["reason"] = "Negative amount"
        return result

    if len(values) == 1:
        result["amount"] = values[0]
        result["method"] = "Single amount"
    elif len(values) == 2:
        separator = text[matches[0].end():matches[1].start()]
        separator = re.sub(r"\bcad\b|\bcanadian\b|\bdollars?\b|\bbucks?\b|\$", "", separator).strip()
        if separator not in {"-", "to"}:
            result["reason"] = "Multiple amounts without one unambiguous range"
            return result
        result["range_lower"], result["range_upper"] = sorted(values)
        result["amount"] = sum(values) / 2
        result["method"] = "Range midpoint"
        if values[0] > values[1]:
            notes.append("Reversed range endpoints; midpoint retained")
    else:
        result["reason"] = "Multiple amounts/scenarios; no single price chosen"
        return result

    if re.search(r"\bat most\b|\bmaximum\b|\bmax\b|\bup to\b|\bno more than\b|<=|=<|<", text):
        notes.append("Upper limit wording; stated number retained")
    if re.search(r"\d\s*\+|\bat least\b|\bminimum\b|>=|=>|>", text):
        notes.append("Lower limit wording; stated number retained")
    if re.search(r"\baround\b|\babout\b|\bperhaps\b|\bmaybe\b", text):
        notes.append("Approximate or conditional amount")
    result["included"] = True
    result["notes"] = "; ".join(notes)
    return result


def prepare(df):
    # 1. Preserve raw text alongside each value, method, exclusion and assumption.
    rows = []
    for index, row in df.iterrows():
        rows.append({"csv_row": index+2, "id": row["id"], "label": row["label"],
                     "raw_answer": row[COLUMN], **parse_answer(row[COLUMN])})
    parsed = pd.DataFrame(rows)
    parsed["iqr_outlier_within_sport"] = False
    # Outliers are review flags, not automatic exclusions or replacements.
    for sport in sorted(df.label.unique()):
        mask = parsed.included & parsed.label.eq(sport)
        values = parsed.loc[mask, "amount"]
        q1, q3 = values.quantile([.25, .75])
        lower, upper = q1-1.5*(q3-q1), q3+1.5*(q3-q1)
        parsed.loc[mask, "iqr_outlier_within_sport"] = (values < lower) | (values > upper)
    parsed.to_csv(OUTPUT_DIR / "parsed_answers.csv", index=False, encoding="utf-8-sig")
    parsed.loc[~parsed.included].to_csv(OUTPUT_DIR / "excluded_answers.csv", index=False, encoding="utf-8-sig")
    flagged = parsed.included & (parsed.notes.ne("") | parsed.iqr_outlier_within_sport)
    parsed.loc[flagged].to_csv(OUTPUT_DIR / "flagged_answers.csv", index=False, encoding="utf-8-sig")
    return parsed


def summarize(parsed, sports):
    # 2. Distinguish no-digit answers from numeric but unusable answers.
    summaries, distributions = [], []
    scopes = {"Overall": pd.Series(True, index=parsed.index)}
    scopes.update({sport: parsed.label.eq(sport) for sport in sports})
    for scope, mask in scopes.items():
        subset = parsed.loc[mask]
        values = subset.loc[subset.included, "amount"]
        q1, q3 = values.quantile([.25, .75])
        summaries.append({
            "scope": scope, "total_count": len(subset), "included_count": len(values),
            "excluded_count": int((~subset.included).sum()),
            "no_digit_count": int((~subset.has_digits).sum()),
            "no_digit_percent": 100*(~subset.has_digits).mean(),
            "blank_count": int(subset.raw_answer.str.strip().eq("").sum()),
            "numeric_but_excluded_count": int((subset.has_digits & ~subset.included).sum()),
            "range_midpoint_count": int((subset.included & subset.method.eq("Range midpoint")).sum()),
            "zero_count": int(values.eq(0).sum()), "zero_percent_of_included": 100*values.eq(0).mean(),
            "mean": values.mean(), "median": values.median(), "std": values.std(),
            "q25": q1, "q75": q3, "min": values.min(), "max": values.max(),
            "iqr_lower_fence": q1-1.5*(q3-q1), "iqr_upper_fence": q3+1.5*(q3-q1),
            "iqr_outlier_count": int(((values < q1-1.5*(q3-q1)) | (values > q3+1.5*(q3-q1))).sum()),
        })
        counts = pd.cut(values, bins=BIN_EDGES, labels=BIN_LABELS, right=True).value_counts(sort=False)
        for label, count in counts.items():
            distributions.append({"scope": scope, "amount_band": label, "count": int(count),
                                  "included_count": len(values),
                                  "percent_of_included": 100*count/len(values) if len(values) else float("nan")})
    summary = pd.DataFrame(summaries)
    summary.loc[summary.scope.eq("Overall")].to_csv(OUTPUT_DIR / "overall_summary.csv", index=False, encoding="utf-8-sig")
    summary.loc[summary.scope.ne("Overall")].to_csv(OUTPUT_DIR / "summary_by_sport.csv", index=False, encoding="utf-8-sig")
    distribution = pd.DataFrame(distributions)
    distribution.to_csv(OUTPUT_DIR / "amount_distributions.csv", index=False, encoding="utf-8-sig")
    return summary, distribution


def plot_results(parsed, summary, distribution, sports):
    # 3. Categorical bands make the long right tail visible without dropping high values.
    overall = distribution.loc[distribution.scope.eq("Overall")]
    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(overall.amount_band, overall.percent_of_included, color=COLORS[0])
    ax.bar_label(bars, labels=[f"{r.percent_of_included:.1f}% ({r.count})" for r in overall.itertuples()], padding=3, fontsize=9)
    ax.set_ylabel("Percentage of included answers")
    ax.set_xlabel("Reported amount band (assumed CAD); (a, b] means greater than a and up to b")
    ax.set_title("Ticket willingness-to-pay - training only")
    ax.tick_params(axis="x", labelrotation=25)
    ax.margins(y=.15)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "price_distribution.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for i, (ax, sport) in enumerate(zip(axes, sports)):
        table = distribution.loc[distribution.scope.eq(sport)]
        bars = ax.bar(range(len(BIN_LABELS)), table.percent_of_included, color=COLORS[i])
        ax.bar_label(bars, labels=[f"{r.percent_of_included:.1f}%\n({r.count})" for r in table.itertuples()], padding=3, fontsize=8)
        ax.set_xticks(range(len(BIN_LABELS)), BIN_LABELS, rotation=50, ha="right", fontsize=8)
        ax.set_title(f"{sport} (n={int(table.included_count.iloc[0])})")
        ax.set_xlabel("Reported amount band (assumed CAD)")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Percentage of included answers within sport")
    axes[0].set_ylim(0, max(distribution.percent_of_included)*1.3)
    fig.suptitle("Ticket willingness-to-pay by sport - training only")
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(OUTPUT_DIR / "price_distribution_by_sport.png", dpi=160)
    plt.close(fig)

    # 4. Show full-range and compressed-axis views of the same unmodified values.
    data = [parsed.loc[parsed.included & parsed.label.eq(sport), "amount"].to_numpy() for sport in sports]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax in axes:
        boxes = ax.boxplot(data, tick_labels=sports, showmeans=True, patch_artist=True,
                           medianprops={"color": "black"},
                           meanprops={"marker": "D", "markerfacecolor": "#bd3030", "markeredgecolor": "#bd3030", "markersize": 5})
        for patch, color in zip(boxes["boxes"], COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(.6)
        ax.set_ylabel("Reported amount (assumed CAD)")
        ax.grid(axis="y", alpha=.2)
    axes[0].set_title("Full linear scale; all amounts retained")
    axes[1].set_yscale("symlog", linthresh=1)
    axes[1].set_title("Compressed scale; zero remains visible")
    axes[1].set_yticks([0, 1, 10, 100, 1000, 10000], ["0", "1", "10", "100", "1,000", "10,000"])
    fig.suptitle("Ticket willingness-to-pay by sport")
    fig.text(.5, .015, "Black line = median; red diamond = mean. Points beyond 1.5 x IQR are flagged, not removed.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .06, 1, .94))
    fig.savefig(OUTPUT_DIR / "price_boxplot_by_sport.png", dpi=160)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_FILE, dtype=str, keep_default_na=False)
    sports = sorted(df.label.unique())
    parsed = prepare(df)
    summary, distribution = summarize(parsed, sports)
    plot_results(parsed, summary, distribution, sports)
    print(summary[["scope", "total_count", "included_count", "no_digit_count", "no_digit_percent", "numeric_but_excluded_count", "mean", "median", "min", "max", "zero_count", "range_midpoint_count"]].round(2).to_string(index=False))
    print("\nExcluded reasons:")
    print(parsed.loc[~parsed.included, "reason"].value_counts().to_string())
    print(f"\nSaved all outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
