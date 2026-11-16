"""
Methodology: decision trees.
Training and validation
"""
import datetime as dt
import getpass
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.tree import DecisionTreeClassifier, export_text

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preprocessing import NUMERIC, SurveyPreprocessor

SEED = 42
CLASSES = ["Baseball", "Basketball", "Soccer"]
TIE_MARGIN = 0.02
DEPTHS = [3, 5, 8, 12]
MIN_LEAF = [2, 5, 10]
DEPTH_EXTEND_LOW, DEPTH_EXTEND_HIGH = [1, 2], [15, 20, 25]
LEAF_EXTEND_LOW, LEAF_EXTEND_HIGH = [1], [20, 40]
OUT = ROOT / "results" / "decision_tree"
LABEL = "label"


#data
def load_split(name):
    return pd.read_csv(ROOT / "data_splits" / f"{name}.csv", dtype=str,
                       keep_default_na=False, encoding="utf-8-sig")


def tree_features(prep, df):
    """Shared features, with the 5 numeric columns put back on their original
    scale (ease 0-10, Likert 1-5, log price). Standardizing does not change
    which split a tree picks, but original units make the splits readable,
    and the proposal says the tree uses unstandardized features."""
    X = prep.transform(df).copy()
    for c in NUMERIC:
        X[c] = X[c] * prep.stds_[c] + prep.means_[c]
    return X


#metrics
def evaluate(model, X, y):
    pred = model.predict(X)
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=CLASSES, zero_division=0)
    out = {"accuracy": accuracy_score(y, pred),
           "macro_f1": f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0)}
    for i, c in enumerate(CLASSES):
        out[f"precision_{c}"], out[f"recall_{c}"], out[f"f1_{c}"] = p[i], r[i], f[i]
    return out


def fit_one(depth, leaf, X_tr, y_tr, X_va, y_va):
    model = DecisionTreeClassifier(criterion="entropy", max_depth=depth,
                                   min_samples_leaf=leaf, random_state=SEED)
    model.fit(X_tr, y_tr)
    tr, va = evaluate(model, X_tr, y_tr), evaluate(model, X_va, y_va)
    row = {"max_depth": depth, "min_samples_leaf": leaf,
           "actual_depth": model.get_depth(), "n_leaves": model.get_n_leaves(),
           "train_accuracy": tr["accuracy"], "train_macro_f1": tr["macro_f1"]}
    row.update({f"val_{k}": v for k, v in va.items()})
    return model, row


def grid(depths, leaves, data, done):
    rows = []
    for d in depths:
        for m in leaves:
            if (d, m) not in done:
                _, row = fit_one(d, m, *data)
                done[(d, m)] = row
                rows.append(row)
    return rows


def choose(results):
    """Best validation macro-F1; anything within TIE_MARGIN is a tie, and among
    ties we keep the simplest tree (fewest leaves, then smallest depth, then
    the larger min_samples_leaf)."""
    best = results["val_macro_f1"].max()
    ties = results[results["val_macro_f1"] >= best - TIE_MARGIN]
    ties = ties.sort_values(["n_leaves", "actual_depth", "min_samples_leaf"],
                            ascending=[True, True, False])
    return ties.iloc[0], best, ties


#check for pred.py
def tree_to_dict(model, columns):
    t = model.tree_
    return {"feature": [columns[i] if i >= 0 else None for i in t.feature],
            "threshold": t.threshold.tolist(),
            "left": t.children_left.tolist(), "right": t.children_right.tolist(),
            "leaf_class": [model.classes_[int(np.argmax(v))] for v in t.value[:, 0, :]],
            "columns": list(columns), "classes": model.classes_.tolist()}


def numpy_predict(tree, X):
    """What pred.py would do: walk the saved split rules with plain numpy."""
    col = {c: i for i, c in enumerate(tree["columns"])}
    out = []
    for x in np.asarray(X, dtype=float):
        node = 0
        while tree["left"][node] != -1:
            go_left = x[col[tree["feature"][node]]] <= tree["threshold"][node]
            node = tree["left"][node] if go_left else tree["right"][node]
        out.append(tree["leaf_class"][node])
    return np.array(out)


#plots
BLUE, ORANGE = "#2a78d6", "#eb6834"


def tuning_plot(results, best, path):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    a = results[results.min_samples_leaf == best.min_samples_leaf].sort_values("max_depth")
    b = results[results.max_depth == best.max_depth].sort_values("min_samples_leaf")
    for axis, df, x, title in [(ax[0], a, "max_depth", f"min_samples_leaf = {int(best.min_samples_leaf)}"),
                               (ax[1], b, "min_samples_leaf", f"max_depth = {int(best.max_depth)}")]:
        axis.plot(df[x], df["train_macro_f1"], "-o", color=BLUE, lw=2, ms=6, label="training")
        axis.plot(df[x], df["val_macro_f1"], "-o", color=ORANGE, lw=2, ms=6, label="validation")
        axis.axvline(best[x], color="0.6", ls="--", lw=1)
        axis.set(xlabel=x, title=title, xticks=df[x].tolist())
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="0.9")
    ax[0].set_ylabel("Macro-F1")
    ax[0].legend(frameon=False)
    fig.suptitle("Decision tree: training vs validation macro-F1 (dashed = chosen value)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
  

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train, val = load_split("train"), load_split("validation")
    prep = SurveyPreprocessor(ROOT).fit(train)          # training data only
    X_tr, X_va = tree_features(prep, train), tree_features(prep, val)
    y_tr, y_va = train[LABEL].to_numpy(), val[LABEL].to_numpy()
    assert X_tr["log_price__missing"].mean() < 0.5, "Price feature mostly missing: check preprocessing.py"
    print(f"Train {X_tr.shape}, validation {X_va.shape}; test set not read.")

    data, done = (X_tr, y_tr, X_va, y_va), {}
    depths, leaves = list(DEPTHS), list(MIN_LEAF)
    grid(depths, leaves, data, done)
    # Extend a range whenever the chosen value sits on its edge (proposal).
    for _ in range(5):
        best, _, _ = choose(pd.DataFrame(done.values()))
        add_d = (DEPTH_EXTEND_LOW if best.max_depth == min(depths) and min(depths) > 1 else
                 DEPTH_EXTEND_HIGH if best.max_depth == max(depths) and max(depths) < 25 else [])
        add_m = (LEAF_EXTEND_LOW if best.min_samples_leaf == min(leaves) and min(leaves) > 1 else
                 LEAF_EXTEND_HIGH if best.min_samples_leaf == max(leaves) and max(leaves) < 40 else [])
        if not add_d and not add_m:
            break
        print(f"Best value on a range edge -> extending depths {add_d}, min_samples_leaf {add_m}")
        depths = sorted(set(depths + add_d))
        leaves = sorted(set(leaves + add_m))
        grid(depths, leaves, data, done)

    results = pd.DataFrame(done.values()).sort_values(["max_depth", "min_samples_leaf"])
    best, best_f1, ties = choose(results)
    model, _ = fit_one(int(best.max_depth), int(best.min_samples_leaf), *data)

    # Experiment log (one row per setting), as promised in the proposal.
    log = results.copy()
    log.insert(0, "model", "decision_tree")
    log.insert(0, "person", getpass.getuser())
    log.insert(0, "date", dt.date.today().isoformat())
    log["criterion"], log["seed"], log["sklearn_version"] = "entropy", SEED, sklearn.__version__
    log.round(4).to_csv(OUT / "experiments.csv", index=False)

    tuning_plot(results, best, OUT / "tuning_curves.png")
    y_hat = model.predict(X_va)
    cm = confusion_plot(y_va, y_hat, OUT / "confusion_matrix.png")
    pd.DataFrame(cm, index=[f"true_{c}" for c in CLASSES],
                 columns=[f"pred_{c}" for c in CLASSES]).to_csv(OUT / "confusion_matrix.csv")

    tree = tree_to_dict(model, X_tr.columns)
    same = (numpy_predict(tree, X_va) == y_hat).all()
    (OUT / "best_tree.json").write_text(json.dumps(tree))
    (OUT / "best_tree_rules.txt").write_text(export_text(model, feature_names=list(X_tr.columns)))
    importance = pd.Series(model.feature_importances_, index=X_tr.columns)
    importance[importance > 0].sort_values(ascending=False).round(4).to_csv(OUT / "feature_importance.csv")

    summary = {"chosen": {"max_depth": int(best.max_depth), "min_samples_leaf": int(best.min_samples_leaf)},
               "actual_depth": int(best.actual_depth), "n_leaves": int(best.n_leaves),
               "train_accuracy": round(float(best.train_accuracy), 4),
               "train_macro_f1": round(float(best.train_macro_f1), 4),
               "val_accuracy": round(float(best.val_accuracy), 4),
               "val_macro_f1": round(float(best.val_macro_f1), 4),
               "best_val_macro_f1_in_grid": round(float(best_f1), 4),
               "settings_tied_within_2pp": int(len(ties)),
               "val_per_class": {c: {m: round(float(best[f"val_{m}_{c}"]), 4)
                                     for m in ("precision", "recall", "f1")} for c in CLASSES},
               "grid_searched": {"max_depth": depths, "min_samples_leaf": leaves},
               "numpy_only_prediction_matches_sklearn": bool(same),
               "saved_tree_json_kb": round((OUT / "best_tree.json").stat().st_size / 1024, 1),
               "sklearn_version": sklearn.__version__}
    (OUT / "best_config.json").write_text(json.dumps(summary, indent=2))

    show = ["max_depth", "min_samples_leaf", "actual_depth", "n_leaves",
            "train_accuracy", "val_accuracy", "train_macro_f1", "val_macro_f1"]
    print(results[show].round(3).to_string(index=False))
    print(json.dumps(summary, indent=2))
    print("Most important features:\n", importance.sort_values(ascending=False).head(10).round(3).to_string())


if __name__ == "__main__":
    main()
