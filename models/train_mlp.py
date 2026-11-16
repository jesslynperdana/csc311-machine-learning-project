"""CSC311 MLP baseline: train on train.csv, evaluate on validation.csv."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preprocessing import SurveyPreprocessor


def main():
    train = pd.read_csv(ROOT / "data_splits/train.csv", dtype=str, keep_default_na=False)
    val = pd.read_csv(ROOT / "data_splits/validation.csv", dtype=str, keep_default_na=False)

    pp = SurveyPreprocessor(project_dir=ROOT)
    X_train = pp.fit_transform(train).to_numpy(dtype=np.float64)
    X_val = pp.transform(val).to_numpy(dtype=np.float64)

    model = MLPClassifier(
        hidden_layer_sizes=(32,),
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42,
    )
    model.fit(X_train, train["label"])

    for name, X, y in [
        ("Train", X_train, train["label"]),
        ("Validation", X_val, val["label"]),
    ]:
        pred = model.predict(X)
        print(f"{name}: Accuracy={accuracy_score(y, pred):.4f}, "
              f"Macro-F1={f1_score(y, pred, average='macro'):.4f}")
        if name == "Validation":
            print("Class order:", list(model.classes_))
            print("Validation confusion matrix:\n",
                  confusion_matrix(y, pred, labels=model.classes_))


if __name__ == "__main__":
    main()
