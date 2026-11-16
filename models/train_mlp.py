
"""CSC311 Sports Classification - Baseline MLP.

Run from repository root:
    python3 models/train_mlp.py

Uses train.csv for fitting and validation.csv for evaluation.
Does not use test.csv.
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.neural_network import MLPClassifier


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocessing import SurveyPreprocessor


DATA_DIR = ROOT / "data_splits"
RESULTS_DIR = ROOT / "results" / "mlp"

SEED = 42
CLASSES = ["Baseball", "Basketball", "Soccer"]


def load_data(filename):
    path = DATA_DIR / filename

    df = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
    )

    required = {"id", "label"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"{filename} must contain id and label columns"
        )

    if not df["label"].isin(CLASSES).all():
        raise ValueError(f"Unexpected labels in {filename}")

    return df


def evaluate_model(model, X, y, title):
    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)
    macro_f1 = f1_score(
        y,
        predictions,
        average="macro",
    )

    print(f"\n{title}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")

    return accuracy, macro_f1, predictions


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train = load_data("train.csv")
    validation = load_data("validation.csv")

    # The same respondent must not appear in both splits.
    train_ids = set(train["id"])
    val_ids = set(validation["id"])

    if not train_ids.isdisjoint(val_ids):
        raise ValueError(
            "Respondent leakage between training and validation"
        )

    # Fit preprocessing on TRAIN only.
    preprocessor = SurveyPreprocessor(project_dir=ROOT)

    X_train_df = preprocessor.fit_transform(train)
    X_val_df = preprocessor.transform(validation)

    if X_train_df.columns.tolist() != X_val_df.columns.tolist():
        raise ValueError("Feature columns do not match")

    feature_names = X_train_df.columns.tolist()

    X_train = X_train_df.to_numpy(dtype=np.float64)
    X_val = X_val_df.to_numpy(dtype=np.float64)

    if not np.isfinite(X_train).all():
        raise ValueError("Non-finite training features")

    if not np.isfinite(X_val).all():
        raise ValueError("Non-finite validation features")

    y_train = train["label"].to_numpy()
    y_val = validation["label"].to_numpy()

    print("=" * 60)
    print("BASELINE MLP")
    print("=" * 60)

    print("Training samples:", len(train))
    print("Validation samples:", len(validation))
    print("Training respondents:", train["id"].nunique())
    print("Validation respondents:", validation["id"].nunique())
    print("Number of features:", len(feature_names))

    # Original baseline configuration.
    model = MLPClassifier(
        hidden_layer_sizes=(32,),
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=500,
        random_state=SEED,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=ConvergenceWarning,
        )
        model.fit(X_train, y_train)

    train_acc, train_f1, _ = evaluate_model(
        model, X_train, y_train, "TRAINING"
    )

    val_acc, val_f1, val_predictions = evaluate_model(
        model, X_val, y_val, "VALIDATION"
    )

    cm = confusion_matrix(
        y_val,
        val_predictions,
        labels=CLASSES,
    )

    print("\nClass order:", CLASSES)
    print("Validation confusion matrix:")
    print(cm)

    print("\nClassification report:")
    print(
        classification_report(
            y_val,
            val_predictions,
            labels=CLASSES,
            zero_division=0,
        )
    )

    print("\nTraining iterations:", model.n_iter_)

    # Save baseline results for the final report.
    summary = pd.DataFrame([{
        "model": "baseline_mlp",
        "feature_set": "all_features",
        "n_features": len(feature_names),
        "hidden_layers": "(32,)",
        "alpha": 0.001,
        "learning_rate": 0.001,
        "max_iter": 500,
        "train_accuracy": train_acc,
        "train_macro_f1": train_f1,
        "validation_accuracy": val_acc,
        "validation_macro_f1": val_f1,
    }])

    summary.to_csv(
        RESULTS_DIR / "baseline_results.csv",
        index=False,
    )

    cm_df = pd.DataFrame(
        cm,
        index=CLASSES,
        columns=CLASSES,
    )

    cm_df.to_csv(
        RESULTS_DIR / "baseline_confusion_matrix.csv"
    )

    print("\nSaved baseline results to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
