"""
Methodology: decision trees.
Training and validation
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from preprocessing import SurveyPreprocessor  # noqa: E402

SEED = 42
DEPTHS = [3, 5, 8, 12]          # proposal Table 4
MIN_LEAF = [2, 5, 10]           # proposal Table 4
DEPTH_LOWER, DEPTH_HIGHER = [1, 2], [16, 20, None]
LEAF_LOWER, LEAF_HIGHER = [1], [20, 40]
OUT = project_root / "results" / "decision_tree"


def run_search(X_train, y_train, groups, depths, leaves):
    grid_search = GridSearchCV(
        DecisionTreeClassifier(criterion="entropy", random_state=SEED),
        param_grid={"max_depth": depths, "min_samples_leaf": leaves},
        cv=StratifiedGroupKFold(n_splits=5),
        scoring="f1_macro",
        return_train_score=True,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train, groups=groups)
    return grid_search


def extension(best, values, lower, higher):
    """Values to add when the best value is at the edge of the searched range."""
    numeric = [v for v in values if v is not None]
    if best is None or best == max(numeric) and None not in values:
        return [v for v in higher if v not in values] if best is not None else []
    if best == min(numeric):
        return [v for v in lower if v not in values]
    return []


def tuning_plot(cv, best, path):
    """Training vs cross-validation macro-F1 for each hyperparameter, holding
    the other at its chosen value."""
    df = pd.DataFrame({"max_depth": cv["param_max_depth"],
                       "min_samples_leaf": cv["param_min_samples_leaf"],
                       "train": cv["mean_train_score"],
                       "cv": cv["mean_test_score"],
                       "cv_std": cv["std_test_score"]})
    df["depth_label"] = df["max_depth"].map(lambda d: "None" if d is None or pd.isna(d) else str(int(d)))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    panels = [
        (ax[0], df[df.min_samples_leaf == best["min_samples_leaf"]], "max_depth",
         f"min_samples_leaf = {best['min_samples_leaf']}"),
        (ax[1], df[df.depth_label == ("None" if best["max_depth"] is None else str(best["max_depth"]))],
         "min_samples_leaf", f"max_depth = {best['max_depth']}"),
    ]
    for axis, d, x, title in panels:
        if x == "max_depth":
            d = d.assign(order=d["max_depth"].fillna(10**6)).sort_values("order")
            xs, labels = range(len(d)), d["depth_label"].tolist()
            chosen = labels.index("None" if best["max_depth"] is None else str(best["max_depth"]))
        else:
            d = d.sort_values(x)
            xs, labels = range(len(d)), d[x].astype(int).astype(str).tolist()
            chosen = labels.index(str(best[x]))
        axis.plot(xs, d["train"], "-o", color="#2a78d6", lw=2, ms=6, label="training")
        axis.errorbar(xs, d["cv"], yerr=d["cv_std"], fmt="-o", color="#eb6834", lw=2, ms=6,
                      capsize=3, label="5-fold CV (± 1 std)")
        axis.axvline(chosen, color="0.6", ls="--", lw=1)
        axis.set(xlabel=x, title=title, xticks=list(xs), xticklabels=labels)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="0.9")
    ax[0].set_ylabel("Macro-F1")
    ax[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Decision tree: training vs cross-validation macro-F1 (dashed = chosen value)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    data_dir = project_root / "data_splits"
    print(f"Loading data from {data_dir}...")
    train_df = pd.read_csv(data_dir / "train.csv", dtype=str, keep_default_na=False)
    val_df = pd.read_csv(data_dir / "validation.csv", dtype=str, keep_default_na=False)
    y_train, y_val = train_df["label"].values, val_df["label"].values
    groups = train_df["id"].values

    print("Fitting preprocessor...")
    preprocessor = SurveyPreprocessor(project_dir=project_root)
    X_train = preprocessor.fit_transform(train_df)
    X_val = preprocessor.transform(val_df)
    print(f"Processed training shape: {X_train.shape}")
    print(f"Processed validation shape: {X_val.shape}")

    print("Training Decision Tree model...")
    depths, leaves = list(DEPTHS), list(MIN_LEAF)
    grid_search = run_search(X_train, y_train, groups, depths, leaves)
    for _ in range(4):  # extend a range whenever the best value sits on its edge
        best = grid_search.best_params_
        add_d = extension(best["max_depth"], depths, DEPTH_LOWER, DEPTH_HIGHER)
        add_m = extension(best["min_samples_leaf"], leaves, LEAF_LOWER, LEAF_HIGHER)
        if not add_d and not add_m:
            break
        print(f"Best value on a range edge: adding max_depth {add_d}, min_samples_leaf {add_m}")
        depths += add_d
        leaves += add_m
        grid_search = run_search(X_train, y_train, groups, depths, leaves)

    best = grid_search.best_params_
    model = grid_search.best_estimator_
    train_preds, val_preds = model.predict(X_train), model.predict(X_val)
    train_acc = accuracy_score(y_train, train_preds)
    val_acc = accuracy_score(y_val, val_preds)
    val_macro_f1 = f1_score(y_val, val_preds, average="macro")
    cv_index = grid_search.best_index_
    cv_std = grid_search.cv_results_["std_test_score"][cv_index]

    print("\n--- Decision Tree Results ---")
    print("Best Hyperparameters:", best)
    print(f"Best CV Macro-F1: {grid_search.best_score_:.4f} (std {cv_std:.4f})")
    print(f"Tree size: depth {model.get_depth()}, {model.get_n_leaves()} leaves")
    print(f"Training Accuracy: {train_acc * 100:.2f}%")
    print(f"Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"Validation Macro-F1: {val_macro_f1:.4f}")
    print("\nValidation Classification Report:")
    print(classification_report(y_val, val_preds, digits=4))

    OUT.mkdir(parents=True, exist_ok=True)
    cv = pd.DataFrame(grid_search.cv_results_)
    keep = ["param_max_depth", "param_min_samples_leaf", "mean_train_score",
            "mean_test_score", "std_test_score", "rank_test_score"]
    cv[keep].rename(columns={"param_max_depth": "max_depth",
                             "param_min_samples_leaf": "min_samples_leaf",
                             "mean_train_score": "train_macro_f1",
                             "mean_test_score": "cv_macro_f1",
                             "std_test_score": "cv_macro_f1_std",
                             "rank_test_score": "rank"}).round(4).fillna({"max_depth": "None"}).to_csv(
        OUT / "experiments.csv", index=False)
    summary = {"best_params": best,
               "cv_macro_f1": round(float(grid_search.best_score_), 4),
               "cv_macro_f1_std": round(float(cv_std), 4),
               "tree_depth": int(model.get_depth()), "n_leaves": int(model.get_n_leaves()),
               "train_accuracy": round(float(train_acc), 4),
               "val_accuracy": round(float(val_acc), 4),
               "val_macro_f1": round(float(val_macro_f1), 4),
               "val_correct": int((val_preds == y_val).sum()), "val_samples": int(len(y_val)),
               "grid_searched": {"max_depth": depths, "min_samples_leaf": leaves}}
    (OUT / "best_config.json").write_text(json.dumps(summary, indent=2))
    tuning_plot(grid_search.cv_results_, best, OUT / "tuning_curves.png")
    joblib.dump(model, OUT / "decision_tree_model.joblib")
    print(f"Results saved to {OUT}")


if __name__ == "__main__":
    main()
