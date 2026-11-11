"""Compare the 21 selected food/drink categories using training responses only.

Mention percentages include every nonmissing answer in the denominator,
including answers that mention only foods outside the selected dictionary.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

"""Build a first food/drink dictionary from inspected training answers only.

The vocabulary below was curated from train.csv, not validation or test data.
Each category is counted at most once per answer. Outputs are a reviewable
draft for exploration, not a final model vocabulary or a general NLP parser.
"""

from collections import Counter, defaultdict
from pathlib import Path
import re
import unicodedata

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data_splits" / "train.csv"
OUTPUT_DIR = PROJECT_DIR / "distribution_results" / "food_and_drinks"
COLUMN = "What foods and drinks do you associate with watching this sport in the place or culture you know best?"

# Alias groups were inspected in the training answers. Brand/category merges
# and spelling corrections are explicit so they can be reviewed and revised.
# Format: (canonical name, type, pipe-separated aliases, interpretation note).
RULES = [
    ('Soft drinks', 'Drink', 'soda|sodas|soft drink|soft drinks|softdrink|pop|pops|cola|coke|coca cola|coke cola|diet coke|pepsi|sprite|fanta|root beer|fizzy drinks', 'Broad category: generic soft drinks and named brands are merged; original text is retained.'),
    ('Beer', 'Drink', 'beer|beers|bier', 'Includes bier; excludes root beer. Ambiguous bear is left for review.'),
    ('Hot dog', 'Food', 'hot dog|hot dogs|hotdog|hotdogs', 'Spacing and plural variants; hyphens normalized.'),
    ('Popcorn', 'Food', 'popcorn|popcorns|pop corn|popocorn', 'Includes observed popocorn typo; pop corn is matched before pop.'),
    ('Pizza', 'Food', 'pizza|pizzas', 'Singular and plural.'),
    ('Fries', 'Food', 'fries|french fries', 'Does not infer that bare chips means fries.'),
    ('Chips', 'Food', 'chips|potato chips|crisps|cheetos|hot cheetos', 'Includes bare chips, potato chips and crisps. Bare chips is not inferred to mean fries; fish and chips is outside this selected list.'),
    ('Burger', 'Food', 'burger|burgers|hamburger|hamburgers|slider|sliders|bergers', 'Groups sliders with burgers; includes observed bergers typo.'),
    ('Nachos', 'Food', 'nacho|nachos|natchos', 'Includes observed natchos typo.'),
    ('Chicken', 'Food', 'chicken wings|chickenwings|wings|buffalo wings|fried chicken|fry chicken|chicken tenders|tenders|chicken', 'Includes wings, fried chicken, tenders and unspecified chicken; each answer counts once even if it names several parts.'),
    ('Peanuts', 'Food', 'peanuts|peanut|fried peanuts', 'Kept separate from unspecified nuts.'),
    ('Energy / electrolyte drinks', 'Drink', 'energy drink|energy drinks|red bull|redbull|monster drink|sports drinks|sports drink|gatorade|gaderade|powerade|electrolyte drinks|电解质饮料|东鹏补水水', 'User-selected analysis group combining energy, sports and electrolyte drinks; these are not nutritionally equivalent.'),
    ('Water', 'Drink', 'water|watwr', 'Includes observed watwr typo; word boundaries prevent substring matches.'),
    ('Sandwich', 'Food', 'sandwich|sandwiches|meatball subs|uncrustables', 'Groups named sandwich products with sandwiches.'),
    ('Poutine', 'Food', 'poutine', 'Kept separate from fries.'),
    ('Pretzel', 'Food', 'pretzel|pretzels|soft pretzels', 'Singular and plural.'),
    ('Sunflower seeds', 'Food', 'sunflower seeds', 'Specific seed snack.'),
    ('Barbecue', 'Food', 'bbq|barbecue', 'Generic barbecue wording; specific skewers remain separate.'),
    ('Gum', 'Food', 'gum|gums|chewing gum|bubblegum', 'Chewing gum is retained; tobacco and nicotine products are not foods.'),
    ('Tea', 'Drink', 'tea|black tea|iced tea|bubble tea|milk tea|teh tarik', 'Broad tea category, including milk/bubble tea; aliases remain visible.'),
    ('Ice cream / frozen treats', 'Food', 'ice cream|icecream|popsicle|freezies|frosty', 'Broad frozen-treat category.'),
]


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


# Only full-answer matches are excluded. For example, "none. maybe hotdogs"
# remains usable because it includes a specific food despite the word none.
MISSING_ANSWERS = {normalize(value) for value in [
    "", "none", "nothing", "no", "unsure", "n/a", "na", "none in particular",
    "nothing in particular", "no food or drink", "no food or drink in particular",
    "N/A, since I know nothing about it at all", "I haven't ever watched it before.",
    "None, the sport is not popular",
    "Perhaps some refreshments for the athletes but I do not associate any food or drink with the viewing experience.",
]}


def make_matcher():
    alias_to_name = {}
    for name, kind, aliases, note in RULES:
        for alias in aliases.split("|"):
            key = normalize(alias)
            assert key not in alias_to_name, key
            alias_to_name[key] = name
    # Block a removed compound dish so its component word does not become a new chips match.
    alias_to_name["fish and chips"] = None
    # Longest matching phrases take precedence at a given location. This avoids
    # counting root beer as beer and pop corn as both popcorn and a soft drink.
    ordered = sorted(alias_to_name, key=lambda x: (-len(x), x))
    pieces = [re.escape(a) if re.search(r"[\u4e00-\u9fff]", a)
              else r"(?<!\w)" + re.escape(a) + r"(?!\w)" for a in ordered]
    return re.compile("|".join(pieces)), alias_to_name


def build_tables(df):
    matcher, lookup = make_matcher()
    normalized = df[COLUMN].map(normalize)
    missing = normalized.isin(MISSING_ANSWERS)
    parsed = []
    observed = defaultdict(Counter)
    evidence = []
    candidate_counts = Counter()
    for index, row in df.iterrows():
        text = normalized.loc[index]
        matches = [] if missing.loc[index] else list(matcher.finditer(text))
        names = {lookup[match.group()] for match in matches if lookup[match.group()] is not None}
        parsed.append(names)
        for alias in {match.group() for match in matches}:
            if lookup[alias] is not None:
                observed[lookup[alias]][alias] += 1
        if not missing.loc[index]:
            # Candidate frequencies are answer counts, not repetitions in a long answer.
            words = text.split()
            candidate_counts.update({" ".join(words[j:j+n]) for n in (1, 2, 3)
                                     for j in range(len(words)-n+1)})
        for name in sorted(names):
            evidence.append({"csv_row": index + 2, "id": row["id"], "label": row["label"],
                             "food_name": name,
                             "matched_phrases": " | ".join(sorted({m.group() for m in matches if lookup[m.group()] == name})),
                             "raw_answer": row[COLUMN]})
    parsed = pd.Series(parsed, index=df.index)
    rows = []
    sports = sorted(df["label"].unique())
    for name, kind, aliases, note in RULES:
        present = parsed.map(lambda names: name in names)
        count = int(present.sum())
        record = {"food_name": name, "type": kind,
                  "tier": "Selected",
                  "observed_aliases": " | ".join(sorted(observed[name])),
                  "answer_count": count, "usable_answer_count": int((~missing).sum()),
                  "percent_of_usable_answers": 100 * count / (~missing).sum(),
                  "interpretation_note": note}
        for sport in sports:
            mask = df.label.eq(sport)
            n = int((mask & ~missing).sum())
            number = int((mask & present).sum())
            record[f"{sport}_count"] = number
            record[f"{sport}_usable_answers"] = n
            record[f"{sport}_percent"] = 100 * number / n if n else float("nan")
        rows.append(record)
    dictionary = pd.DataFrame(rows).sort_values(["answer_count", "food_name"], ascending=[False, True])
    core_names = set(dictionary.loc[dictionary.tier.eq("Selected"), "food_name"])

    # Exclusions, unmatched answers, and uncertain spellings remain inspectable.
    excluded = df.loc[missing, ["id", "label"]].copy()
    excluded.insert(0, "csv_row", excluded.index + 2)
    excluded["raw_answer"] = df.loc[missing, COLUMN]
    excluded["reason"] = ["Blank answer" if normalized.loc[i] == "" else "No association / unsure; treated as missing by project rule" for i in excluded.index]
    reviews = []
    for index in df.index[~missing]:
        reasons = []
        names = parsed.loc[index]
        if not names:
            reasons.append("No selected-category match; retained in the percentage denominator, not treated as missing")
        if re.search(r"\bbear\b|\bpower drinks\b", normalized.loc[index]):
            reasons.append("Ambiguous wording; no automatic correction")
        if re.search(r"\bcigarettes\b|\btobacco\b|\bzonnic\b|\bshisha\b", normalized.loc[index]):
            reasons.append("Non-food items present; those items are not mapped to foods")
        if reasons:
            reviews.append({"csv_row": index+2, "id": df.loc[index, "id"], "label": df.loc[index, "label"],
                            "raw_answer": df.loc[index, COLUMN], "matched_foods": " | ".join(sorted(names)),
                            "reason": "; ".join(reasons)})
    coverage_rows = []
    scopes = {"Overall": pd.Series(True, index=df.index)}
    scopes.update({sport: df.label.eq(sport) for sport in sports})
    for scope, mask in scopes.items():
        usable = mask & ~missing
        coverage_rows.append({"scope": scope, "total_count": int(mask.sum()),
            "blank_count": int((mask & normalized.eq("")).sum()),
            "nonblank_missing_count": int((mask & missing & normalized.ne("")).sum()),
            "usable_count": int(usable.sum()),
            "selected_matched_count": int((usable & parsed.map(lambda names: bool(names & core_names))).sum()),
            "unmatched_count": int((usable & ~parsed.map(bool)).sum())})
    return dictionary, excluded, pd.DataFrame(reviews), pd.DataFrame(evidence), pd.DataFrame(coverage_rows), candidate_counts


COLORS = ["#367a9c", "#ce8133", "#48865c"]


def summarize(dictionary, coverage, sports):
    # 1. Use one denominator per sport; missing responses are excluded only for this question.
    rows = []
    coverage = coverage.set_index("scope")
    for scope in ["Overall", *sports]:
        meta = coverage.loc[scope]
        for row in dictionary.itertuples():
            count = row.answer_count if scope == "Overall" else getattr(row, f"{scope}_count")
            n = int(meta.usable_count)
            rows.append({
                "scope": scope, "food_name": row.food_name, "selected_count": int(count),
                "total_response_count": int(meta.total_count), "usable_response_count": n,
                "missing_count": int(meta.blank_count + meta.nonblank_missing_count),
                "missing_percent": 100 * (meta.blank_count + meta.nonblank_missing_count) / meta.total_count,
                "no_selected_food_count": int(meta.unmatched_count),
                "percent_of_usable_answers": 100 * count / n if n else float("nan"),
            })
    return pd.DataFrame(rows)


def plot_summary(summary, sports):
    # 2. Preserve the selected category order; the same answer may appear in several bars.
    order = [rule[0] for rule in RULES]
    overall = summary.loc[summary.scope.eq("Overall")].set_index("food_name").loc[order]
    fig, ax = plt.subplots(figsize=(12, 10))
    bars = ax.barh(order, overall.percent_of_usable_answers, color=COLORS[0])
    ax.bar_label(bars, labels=[f"{r.percent_of_usable_answers:.1f}% ({r.selected_count})"
                             for r in overall.itertuples()], padding=4, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("Percentage of nonmissing training answers")
    ax.set_title(f"Selected food and drink associations - overall (n={int(overall.usable_response_count.iloc[0])})")
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    fig.text(.5, .015, "Each category counted once per answer. Answers with no selected-category match remain in the denominator.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(OUTPUT_DIR / "foods_overall.png", dpi=160)
    plt.close(fig)

    # 3. Use two panels so all 21 categories and their counts remain legible.
    fig, axes = plt.subplots(1, 2, figsize=(20, 11))
    height = .8 / len(sports)
    for ax, names in zip(axes, [order[:11], order[11:]]):
        for i, sport in enumerate(sports):
            table = summary.loc[summary.scope.eq(sport)].set_index("food_name").loc[names]
            positions = [j - .4 + height / 2 + i * height for j in range(len(names))]
            bars = ax.barh(positions, table.percent_of_usable_answers, height=height,
                           color=COLORS[i % len(COLORS)],
                           label=f"{sport} (n={int(table.usable_response_count.iloc[0])})")
            ax.bar_label(bars, labels=[f"{r.percent_of_usable_answers:.1f}% ({r.selected_count})"
                                      for r in table.itertuples()], padding=3, fontsize=9)
        ax.set_yticks(range(len(names)), names)
        ax.set_ylim(-.7, 10.6)
        ax.invert_yaxis()
        ax.set_xlim(0, 108)
        ax.set_xticks(range(0, 101, 20))
        ax.set_xlabel("Percentage of nonmissing answers within sport")
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Selected food and drink associations by sport - training only", fontsize=16)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .955), ncol=len(sports), frameon=False)
    fig.text(.5, .015, "Chicken parts are combined; energy, sports and electrolyte drinks are combined. Counts are unique answers, not word repetitions.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .045, 1, .91), w_pad=3)
    fig.savefig(OUTPUT_DIR / "foods_by_sport.png", dpi=160)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_FILE, dtype=str, keep_default_na=False)
    dictionary, excluded, reviews, evidence, coverage, _ = build_tables(df)
    sports = sorted(df.label.unique())
    summary = summarize(dictionary, coverage, sports)
    for filename, table in [
        ("food_dictionary.csv", dictionary), ("excluded_answers.csv", excluded),
        ("answers_to_review.csv", reviews), ("matched_food_evidence.csv", evidence),
        ("coverage_summary.csv", coverage),
        ("overall_summary.csv", summary.loc[summary.scope.eq("Overall")]),
        ("summary_by_sport.csv", summary.loc[summary.scope.ne("Overall")]),
    ]:
        table.to_csv(OUTPUT_DIR / filename, index=False, encoding="utf-8-sig")
    plot_summary(summary, sports)
    print(coverage.to_string(index=False))
    print(summary.pivot(index="food_name", columns="scope", values="percent_of_usable_answers").round(1).to_string())
    print(f"\nSaved all results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
