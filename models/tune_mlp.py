
"""CSC311 Sports Classification - MLP Tuning and Final Training.

Run from repository root:

    python3 models/tune_mlp.py --compare
    python3 models/tune_mlp.py --test
    python3 models/tune_mlp.py --export

Optional: repeat hyperparameter search

    python3 models/tune_mlp.py --search
    python3 models/tune_mlp.py --quick

Only --test reads test.csv.
Only --export trains the final model on train + validation.

No dependency on mlp_common.py or pred.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocessing import SurveyPreprocessor


DATA_DIR = ROOT / "data_splits"
RESULTS_DIR = ROOT / "results" / "mlp"

SEED = 42
MAX_ITER = 800
CLASSES = ["Baseball", "Basketball", "Soccer"]


FEATURE_SETS = {
    "all_features": [],
    "without_feeling": ["feeling"],
    "without_ease_season": ["ease", "season"],
    "without_price": ["price"],
}

GROUP_PREFIXES = {
    "ease": ("ease",),
    "price": ("log_price",),
    "season": ("season__",),
    "feeling": ("feeling__",),
}


def selected_columns(columns, removed_groups):
    """Select feature columns while preserving original order."""
    prefixes = []

    for group in removed_groups:
        if group not in GROUP_PREFIXES:
            raise ValueError(f"Unknown feature group: {group}")
        prefixes.extend(GROUP_PREFIXES[group])

    if not prefixes:
        return list(columns)

    selected = [
        name for name in columns
        if not any(
            str(name).lower().startswith(prefix)
            for prefix in prefixes
        )
    ]

    if not selected:
        raise ValueError("No features remaining")

    return selected


def check_price_column(X):
    if "log_price__missing" in X.columns:
        missing_rate = float(X["log_price__missing"].mean())

        print(
            f"[check] Price missing rate: {missing_rate:.1%}"
        )

        if missing_rate > 0.99:
            raise ValueError(
                "Price extraction may be broken. "
                "Check the NUMBER regex in preprocessing.py."
            )


def load_data(filename):
    df = pd.read_csv(
        DATA_DIR / filename,
        dtype=str,
        keep_default_na=False,
    )

    if not {"id", "label"}.issubset(df.columns):
        raise ValueError(
            f"{filename} must contain id and label"
        )

    if not df["label"].isin(CLASSES).all():
        raise ValueError(f"Unexpected label in {filename}")

    return df


def check_disjoint(a, b):
    if not set(a["id"]).isdisjoint(set(b["id"])):
        raise ValueError("Respondent leakage between datasets")


def preprocess_train_eval(train, evaluation):
    """Fit preprocessing only on the training portion."""
    check_disjoint(train, evaluation)

    processor = SurveyPreprocessor(project_dir=ROOT)

    X_train = processor.fit_transform(train)
    X_eval = processor.transform(evaluation)

    if X_train.columns.tolist() != X_eval.columns.tolist():
        raise ValueError("Feature column/order mismatch")

    if not np.isfinite(
        X_train.to_numpy(dtype=np.float64)
    ).all():
        raise ValueError("Non-finite training values")

    if not np.isfinite(
        X_eval.to_numpy(dtype=np.float64)
    ).all():
        raise ValueError("Non-finite evaluation values")

    return processor, X_train, X_eval


def make_model(config):
    return MLPClassifier(
        hidden_layer_sizes=tuple(
            int(v) for v in config["hidden_layer_sizes"]
        ),
        activation="relu",
        solver="adam",
        alpha=float(config["alpha"]),
        learning_rate_init=float(
            config["learning_rate_init"]
        ),
        max_iter=int(config["max_iter"]),
        n_iter_no_change=30,
        random_state=int(config["random_state"]),
    )


def fit_model(model, X, y):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=ConvergenceWarning,
        )
        model.fit(X, y)

    return model


def get_best_config():
    path = RESULTS_DIR / "best_config.json"

    if not path.exists():
        raise FileNotFoundError(
            "Missing results/mlp/best_config.json"
        )

    return json.loads(path.read_text(encoding="utf-8"))


def feature_names_from_config(columns, config):
    selected = selected_columns(
        columns,
        config["removed_groups"],
    )

    expected = int(config["n_features"])

    if len(selected) != expected:
        raise ValueError(
            f"Expected {expected} features, "
            f"got {len(selected)}"
        )

    return selected


def report_predictions(y_true, y_pred, title):
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true, y_pred, average="macro"
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASSES,
    )

    print(f"\n{title}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print("Class order:", CLASSES)
    print("Confusion matrix:")
    print(cm)

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=CLASSES,
            zero_division=0,
        )
    )

    return accuracy, macro_f1, cm


def search_grid(quick=False):
    if quick:
        sizes = [(16,), (32,)]
        alphas = [0.001, 0.1]
        learning_rates = [0.001]
    else:
        sizes = [(16,), (32,), (64,), (32, 16)]
        alphas = [0.001, 0.01, 0.1]
        learning_rates = [0.0003, 0.001]

    for fs, hidden, alpha, lr in product(
        FEATURE_SETS,
        sizes,
        alphas,
        learning_rates,
    ):
        yield fs, hidden, alpha, lr


def build_cv_cache(train):
    cv = GroupKFold(n_splits=5)
    y = train["label"].to_numpy()
    groups = train["id"].to_numpy()

    cache = []

    for tr_idx, val_idx in cv.split(train, y, groups):
        fold_train = train.iloc[tr_idx]
        fold_val = train.iloc[val_idx]

        _, X_tr, X_val = preprocess_train_eval(
            fold_train, fold_val
        )

        cache.append((
            X_tr,
            X_val,
            y[tr_idx],
            y[val_idx],
        ))

    return cache


def run_search(quick=False):
    train = load_data("train.csv")
    cache = build_cv_cache(train)

    experiments = list(search_grid(quick))
    results = []

    print(
        f"Evaluating {len(experiments)} configs "
        f"x {len(cache)} folds"
    )

    for i, (fs, hidden, alpha, lr) in enumerate(
        experiments, 1
    ):
        accuracies = []
        macro_f1s = []
        n_features = None

        config = {
            "hidden_layer_sizes": list(hidden),
            "alpha": alpha,
            "learning_rate_init": lr,
            "max_iter": MAX_ITER,
            "random_state": SEED,
        }

        for X_tr, X_val, y_tr, y_val in cache:
            cols = selected_columns(
                X_tr.columns,
                FEATURE_SETS[fs],
            )
            n_features = len(cols)

            model = make_model(config)
            fit_model(
                model,
                X_tr[cols].to_numpy(dtype=np.float64),
                y_tr,
            )

            pred = model.predict(
                X_val[cols].to_numpy(dtype=np.float64)
            )

            accuracies.append(
                accuracy_score(y_val, pred)
            )
            macro_f1s.append(
                f1_score(y_val, pred, average="macro")
            )

        row = {
            "feature_set": fs,
            "n_features": n_features,
            "hidden_layers": "-".join(map(str, hidden)),
            "alpha": alpha,
            "learning_rate": lr,
            "max_iter": MAX_ITER,
            "mean_accuracy": float(np.mean(accuracies)),
            "std_accuracy": float(np.std(accuracies)),
            "mean_macro_f1": float(np.mean(macro_f1s)),
            "std_macro_f1": float(np.std(macro_f1s)),
        }

        for j, (acc, f1) in enumerate(
            zip(accuracies, macro_f1s), 1
        ):
            row[f"fold_{j}_accuracy"] = float(acc)
            row[f"fold_{j}_macro_f1"] = float(f1)

        results.append(row)

        print(
            f"[{i:3d}/{len(experiments)}] "
            f"{fs:21s} h={hidden} "
            f"alpha={alpha:g} lr={lr:g} "
            f"Accuracy={row['mean_accuracy']:.4f} "
            f"Macro-F1={row['mean_macro_f1']:.4f}",
            flush=True,
        )

    ranking = pd.DataFrame(results).sort_values(
        [
            "mean_accuracy",
            "mean_macro_f1",
            "std_accuracy",
            "n_features",
        ],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    suffix = "quick" if quick else "full"

    ranking.to_csv(
        RESULTS_DIR / f"tuning_{suffix}_results.csv",
        index=False,
    )

    best = ranking.iloc[0]

    config = {
        "feature_set": str(best["feature_set"]),
        "removed_groups": FEATURE_SETS[
            str(best["feature_set"])
        ],
        "hidden_layer_sizes": [
            int(x)
            for x in str(best["hidden_layers"]).split("-")
        ],
        "alpha": float(best["alpha"]),
        "learning_rate_init": float(best["learning_rate"]),
        "max_iter": MAX_ITER,
        "random_state": SEED,
        "cv_mean_accuracy": float(best["mean_accuracy"]),
        "cv_mean_macro_f1": float(best["mean_macro_f1"]),
        "cv_std_accuracy": float(best["std_accuracy"]),
        "n_features": int(best["n_features"]),
        "n_configs_searched": len(experiments),
        "quick_mode": quick,
    }

    # Save separately. Do not overwrite an already
    # selected best_config.json automatically.
    path = RESULTS_DIR / f"best_{suffix}_config.json"
    path.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print("\nSearch completed.")
    print("Best configuration:")
    print(json.dumps(config, indent=2))
    print("Saved:", path)
    print(
        "Existing best_config.json was not overwritten."
    )


def compare_top_models():
    path = RESULTS_DIR / "tuning_results.csv"

    if not path.exists():
        raise FileNotFoundError(
            "Missing tuning_results.csv"
        )

    ranking = pd.read_csv(path).sort_values(
        [
            "mean_accuracy",
            "mean_macro_f1",
            "std_accuracy",
            "n_features",
        ],
        ascending=[False, False, True, True],
    ).head(3)

    train = load_data("train.csv")
    val = load_data("validation.csv")

    _, X_train, X_val = preprocess_train_eval(
        train, val
    )

    y_train = train["label"].to_numpy()
    y_val = val["label"].to_numpy()

    rows = []

    for rank, (_, cfg) in enumerate(
        ranking.iterrows(), 1
    ):
        hidden = [
            int(x)
            for x in str(cfg["hidden_layers"]).split("-")
        ]

        config = {
            "hidden_layer_sizes": hidden,
            "alpha": float(cfg["alpha"]),
            "learning_rate_init": float(
                cfg["learning_rate"]
            ),
            "max_iter": MAX_ITER,
            "random_state": SEED,
        }

        removed = FEATURE_SETS[str(cfg["feature_set"])]
        cols = selected_columns(X_train.columns, removed)

        model = make_model(config)
        fit_model(
            model,
            X_train[cols].to_numpy(dtype=np.float64),
            y_train,
        )

        train_pred = model.predict(
            X_train[cols].to_numpy(dtype=np.float64)
        )
        val_pred = model.predict(
            X_val[cols].to_numpy(dtype=np.float64)
        )

        train_acc = accuracy_score(y_train, train_pred)

        val_acc, val_f1, cm = report_predictions(
            y_val,
            val_pred,
            f"MODEL {rank}: {cfg['feature_set']}",
        )

        print("Hidden layers:", hidden)
        print("Alpha:", config["alpha"])
        print("Learning rate:", config["learning_rate_init"])
        print("Train accuracy:", round(train_acc, 4))

        rows.append({
            "cv_rank": rank,
            "feature_set": cfg["feature_set"],
            "n_features": len(cols),
            "hidden_layers": str(tuple(hidden)),
            "alpha": config["alpha"],
            "learning_rate": config["learning_rate_init"],
            "cv_accuracy": float(cfg["mean_accuracy"]),
            "cv_macro_f1": float(cfg["mean_macro_f1"]),
            "train_accuracy": float(train_acc),
            "validation_accuracy": float(val_acc),
            "validation_macro_f1": float(val_f1),
            "validation_errors": int(
                np.sum(y_val != val_pred)
            ),
            "n_iterations": int(model.n_iter_),
        })

    summary = pd.DataFrame(rows).sort_values(
        ["validation_accuracy", "validation_macro_f1"],
        ascending=[False, False],
    )

    output = RESULTS_DIR / "top3_validation_comparison.csv"
    summary.to_csv(output, index=False)

    print("\nFINAL VALIDATION COMPARISON")
    print(summary.to_string(index=False))
    print("\nSaved:", output)

    print(
        "\nBest configuration remains unchanged."
    )


def evaluate_holdout_test():
    """Evaluate the frozen configuration on test.csv.

    Fit only on train.csv.
    Do not use the test result to tune parameters.
    """
    config = get_best_config()

    train = load_data("train.csv")
    test = load_data("test.csv")

    processor, X_train, X_test = preprocess_train_eval(
        train, test
    )

    cols = feature_names_from_config(
        X_train.columns,
        config,
    )

    y_train = train["label"].to_numpy()
    y_test = test["label"].to_numpy()

    model = make_model(config)

    fit_model(
        model,
        X_train[cols].to_numpy(dtype=np.float64),
        y_train,
    )

    predictions = model.predict(
        X_test[cols].to_numpy(dtype=np.float64)
    )

    acc, macro_f1, cm = report_predictions(
        y_test,
        predictions,
        "FINAL HOLDOUT TEST",
    )

    print("Feature set:", config["feature_set"])
    print("Number of features:", len(cols))
    print("Training rows:", len(train))
    print("Test rows:", len(test))

    result = {
        "feature_set": config["feature_set"],
        "n_features": len(cols),
        "hidden_layer_sizes": config["hidden_layer_sizes"],
        "alpha": config["alpha"],
        "learning_rate_init": config["learning_rate_init"],
        "cv_mean_accuracy": config["cv_mean_accuracy"],
        "cv_mean_macro_f1": config["cv_mean_macro_f1"],
        "test_accuracy": float(acc),
        "test_macro_f1": float(macro_f1),
        "test_errors": int(np.sum(y_test != predictions)),
    }

    (RESULTS_DIR / "holdout_test_results.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame(
        cm,
        index=CLASSES,
        columns=CLASSES,
    ).to_csv(RESULTS_DIR / "holdout_confusion_matrix.csv")

    print("\nHoldout results saved.")
    print(
        "Do not change hyperparameters or features "
        "based on this test score."
    )


def export_final_model():
    """Refit selected MLP on train + validation.

    Saves NumPy weights and preprocessing metadata.
    Does not require pred.py.
    Does not use test.csv.
    """
    config = get_best_config()

    train = load_data("train.csv")
    val = load_data("validation.csv")

    check_disjoint(train, val)

    full = pd.concat(
        [train, val],
        ignore_index=True,
    )

    processor = SurveyPreprocessor(project_dir=ROOT)
    X_full = processor.fit_transform(full)

    check_price_column(X_full)

    cols = feature_names_from_config(
        X_full.columns,
        config,
    )

    X = X_full[cols].to_numpy(dtype=np.float64)
    y = full["label"].to_numpy()

    model = make_model(config)
    fit_model(model, X, y)

    if model.out_activation_ != "softmax":
        raise ValueError("Expected multiclass softmax MLP")

    # Save weights and biases
    weights = {}

    for i, W in enumerate(model.coefs_):
        weights[f"W{i}"] = W

    for i, b in enumerate(model.intercepts_):
        weights[f"b{i}"] = b

    weights_path = ROOT / "model_params.npz"
    np.savez_compressed(weights_path, **weights)

    numeric_names = list(processor.medians_.index)

    metadata = {
        "model_type": "sklearn_mlp_relu_multiclass",
        "classes": [
            str(x) for x in model.classes_.tolist()
        ],
        "n_layers": len(model.coefs_),
        "all_feature_names": X_full.columns.tolist(),
        "selected_feature_names": cols,
        "feature_set": config["feature_set"],
        "removed_groups": config["removed_groups"],
        "text_names": {
            k: [str(x) for x in v]
            for k, v in processor.text_names_.items()
        },
        "numeric_feature_names": numeric_names,
        "numeric_medians": [
            float(processor.medians_[k])
            for k in numeric_names
        ],
        "numeric_means": [
            float(processor.means_[k])
            for k in numeric_names
        ],
        "numeric_stds": [
            float(processor.stds_[k])
            for k in numeric_names
        ],
        "hidden_layer_sizes": config["hidden_layer_sizes"],
        "alpha": config["alpha"],
        "learning_rate_init": config["learning_rate_init"],
        "max_iter": config["max_iter"],
        "random_state": config["random_state"],
        "cv_mean_accuracy": config["cv_mean_accuracy"],
        "cv_mean_macro_f1": config["cv_mean_macro_f1"],
        "train_plus_validation_rows": len(full),
        "fitted_iterations": int(model.n_iter_),
    }

    metadata_path = ROOT / "model_metadata.json"
    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Confirm saved weights reproduce sklearn probabilities
    # on training examples. This is NOT a test-set score.
    with np.load(weights_path) as saved:
        manual = X[:60].copy()
        for i in range(len(model.coefs_)):
            manual = manual @ saved[f"W{i}"] + saved[f"b{i}"]
            if i < len(model.coefs_) - 1:
                manual = np.maximum(manual, 0.0)

        manual = manual - manual.max(axis=1, keepdims=True)
        exp_values = np.exp(manual)
        manual_probs = (
            exp_values / exp_values.sum(axis=1, keepdims=True)
        )

    sklearn_probs = model.predict_proba(X[:60])

    np.testing.assert_allclose(
        manual_probs,
        sklearn_probs,
        rtol=1e-9,
        atol=1e-10,
    )

    print("\n" + "=" * 60)
    print("FINAL MLP EXPORTED")
    print("=" * 60)

    print("Feature set:", config["feature_set"])
    print("Number of features:", len(cols))
    print("Hidden layers:", config["hidden_layer_sizes"])
    print("Alpha:", config["alpha"])
    print("Learning rate:", config["learning_rate_init"])
    print("Training samples:", len(full))
    print("Training iterations:", model.n_iter_)

    print("\nNumPy prediction equivalence: PASS")
    print("Saved:", weights_path)
    print("Saved:", metadata_path)
    print("\nLocal test.csv was not used.")


def main():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--compare",
        action="store_true",
        help="Compare top 3 CV models on validation",
    )

    group.add_argument(
        "--test",
        action="store_true",
        help="Evaluate frozen model on local holdout test",
    )

    group.add_argument(
        "--export",
        action="store_true",
        help="Train final model on train + validation",
    )

    group.add_argument(
        "--search",
        action="store_true",
        help="Run full 96-configuration search",
    )

    group.add_argument(
        "--quick",
        action="store_true",
        help="Run 16-configuration quick search",
    )

    args = parser.parse_args()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.compare:
        compare_top_models()

    elif args.test:
        evaluate_holdout_test()

    elif args.export:
        export_final_model()

    elif args.search:
        run_search(quick=False)

    elif args.quick:
        run_search(quick=True)


if __name__ == "__main__":
    main()
