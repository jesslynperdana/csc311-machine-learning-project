"""CSC311 tuned MLP.

Run from the project root:
    python3 models/tune_mlp.py           # Final model: train/validation scores
    python3 models/tune_mlp.py --search  # Reproduce the 96-config grouped CV
    python3 models/tune_mlp.py --export  # Train on train+validation, export weights

All operations use only train.csv and/or validation.csv, never test.csv.
"""

import argparse
import json
import sys
import warnings
import joblib
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preprocessing import SurveyPreprocessor

OUT = ROOT / "results/mlp"

# Frozen winner of the original 96-configuration grouped CV search.
BEST = dict(hidden_layer_sizes=(64,), alpha=0.1, learning_rate_init=0.001)

# Same four feature sets used in the original tuning search.
FEATURE_SETS = {
    "all_features": (),
    "without_feeling": ("feeling__",),
    "without_ease_season": ("ease", "season__"),
    "without_price": ("log_price",),
}


def load(name):
    df = pd.read_csv(ROOT / "data_splits" / name, dtype=str, keep_default_na=False)
    if not {"id", "label"}.issubset(df.columns):
        raise ValueError(f"{name} must contain 'id' and 'label'")
    return df


def selected_columns(columns, feature_set):
    prefixes = FEATURE_SETS[feature_set]
    return [col for col in columns if not str(col).startswith(prefixes)]


def make_model(hidden_layer_sizes, alpha, learning_rate_init):
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=800,
        n_iter_no_change=30,
        random_state=42,
    )


def evaluate():
    """Reproduce the frozen tuned model's train and validation results."""
    train, val = load("train.csv"), load("validation.csv")
    pp = SurveyPreprocessor(project_dir=ROOT)
    X_train = pp.fit_transform(train)
    X_val = pp.transform(val)
    cols = selected_columns(X_train.columns, "all_features")

    model = make_model(**BEST)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X_train[cols].to_numpy(dtype=np.float64), train["label"])
        OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "preprocessor": pp, "features": cols},
        OUT / "mlp_model.joblib"
    )
    print("Saved:", OUT / "mlp_model.joblib")

    print(f"Tuned MLP: {len(cols)} features, hidden=(64,), alpha=0.1, lr=0.001")
    for name, X, y in [
        ("Train", X_train, train["label"]),
        ("Validation", X_val, val["label"]),
    ]:
        pred = model.predict(X[cols].to_numpy(dtype=np.float64))
        print(f"{name}: Accuracy={accuracy_score(y, pred):.4f}, "
              f"Macro-F1={f1_score(y, pred, average='macro'):.4f}")
        if name == "Validation":
            print("Class order:", list(model.classes_))
            print("Validation confusion matrix:\n",
                  confusion_matrix(y, pred, labels=model.classes_))


def search():
    """Reproduce original 4 feature sets x 4 shapes x 3 alphas x 2 LRs."""
    train = load("train.csv")
    y = train["label"].to_numpy()
    ids = train["id"].to_numpy()

    folds = []
    for tr, va in GroupKFold(n_splits=5).split(train, y, ids):
        if set(ids[tr]) & set(ids[va]):
            raise AssertionError("Respondent leakage between CV folds")
        pp = SurveyPreprocessor(project_dir=ROOT)
        X_tr = pp.fit_transform(train.iloc[tr])
        X_va = pp.transform(train.iloc[va])
        folds.append((X_tr, X_va, y[tr], y[va]))

    rows = []
    configurations = list(product(
        FEATURE_SETS,
        [(16,), (32,), (64,), (32, 16)],
        [0.001, 0.01, 0.1],
        [0.0003, 0.001],
    ))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for i, (features, hidden, alpha, lr) in enumerate(configurations, 1):
            accuracies, macro_f1s = [], []
            for X_tr, X_va, y_tr, y_va in folds:
                cols = selected_columns(X_tr.columns, features)
                model = make_model(hidden, alpha, lr)
                model.fit(X_tr[cols].to_numpy(dtype=np.float64), y_tr)
                pred = model.predict(X_va[cols].to_numpy(dtype=np.float64))
                accuracies.append(accuracy_score(y_va, pred))
                macro_f1s.append(f1_score(y_va, pred, average="macro"))
            rows.append({
                "feature_set": features,
                "n_features": len(cols),
                "hidden_layers": str(hidden),
                "alpha": alpha,
                "learning_rate": lr,
                "mean_accuracy": float(np.mean(accuracies)),
                "std_accuracy": float(np.std(accuracies)),
                "mean_macro_f1": float(np.mean(macro_f1s)),
            })
            print(f"[{i:02d}/96] {features}, {hidden}, alpha={alpha}, "
                  f"lr={lr}: CV Accuracy={np.mean(accuracies):.4f}", flush=True)

    results = pd.DataFrame(rows).sort_values(
        ["mean_accuracy", "mean_macro_f1", "std_accuracy", "n_features"],
        ascending=[False, False, True, True],
    )
    OUT.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT / "tuning_results.csv", index=False)
    print("\nTop configuration:\n", results.head(1).to_string(index=False))
    print("Saved:", OUT / "tuning_results.csv")


def export():
    """Fit frozen best MLP on train+validation for pred.py (no local test)."""
    full = pd.concat([load("train.csv"), load("validation.csv")], ignore_index=True)
    pp = SurveyPreprocessor(project_dir=ROOT)
    X = pp.fit_transform(full)
    cols = selected_columns(X.columns, "all_features")

    model = make_model(**BEST)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X[cols].to_numpy(dtype=np.float64), full["label"])
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "preprocessor": pp, "features": cols},
        OUT / "mlp_model.joblib"
    )
    print("Saved:", OUT / "mlp_model.joblib")       

    numeric = list(pp.medians_.index)
    metadata = {
        "model_type": "sklearn_mlp_relu_multiclass",
        "classes": [str(x) for x in model.classes_],
        "n_layers": len(model.coefs_),
        "all_feature_names": list(X.columns),
        "selected_feature_names": cols,
        "feature_set": "all_features",
        "removed_groups": [],
        "text_names": {k: [str(v) for v in names]
                       for k, names in pp.text_names_.items()},
        "numeric_feature_names": numeric,
        "numeric_medians": [float(pp.medians_[k]) for k in numeric],
        "numeric_means": [float(pp.means_[k]) for k in numeric],
        "numeric_stds": [float(pp.stds_[k]) for k in numeric],
        "train_plus_validation_rows": len(full),
    }
    weights = {f"W{i}": w for i, w in enumerate(model.coefs_)}
    weights.update({f"b{i}": b for i, b in enumerate(model.intercepts_)})
    np.savez_compressed(ROOT / "model_params.npz", **weights)
    (ROOT / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Exported MLP: {len(cols)} features, {len(full)} training rows")
    print("Saved: model_params.npz, model_metadata.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--search", action="store_true", help="reproduce 96-config CV")
    group.add_argument("--export", action="store_true", help="export fitted MLP")
    args = parser.parse_args()
    if args.search:
        search()
    elif args.export:
        export()
    else:
        evaluate()
