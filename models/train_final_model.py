from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preprocessing import SurveyPreprocessor


def main():
    splits = ROOT / "data_splits"
    train = pd.read_csv(splits / "train.csv", dtype=str, keep_default_na=False)
    validation = pd.read_csv(splits / "validation.csv", dtype=str, keep_default_na=False)
    combined = pd.concat([train, validation], ignore_index=True)

    if "label" not in combined or "id" not in combined:
        raise ValueError("The train and validation files must contain id and label columns.")

    if set(train["id"]) & set(validation["id"]):
        raise ValueError("Train and validation contain overlapping student IDs.")

    preprocessor = SurveyPreprocessor(project_dir=ROOT)
    X = preprocessor.fit_transform(combined)
    y = combined["label"].to_numpy()
    model = Pipeline([
        ("var_thresh", VarianceThreshold(threshold=0.0)),
        ("clf", LogisticRegression(
            C=0.5, solver="lbfgs", max_iter=3000, random_state=42
        )),
    ])
    model.fit(X, y)

    output_path = ROOT / "results" / "final_model.joblib"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"preprocessor": preprocessor, "model": model}, output_path)
    print(f"Final fit: {len(train)} training + {len(validation)} validation samples = {len(combined)}")
    print("Selected model: Logistic Regression (C=0.5, solver=lbfgs, variance threshold=0.0)")
    print(f"Saved final model to {output_path}")
    print("Test split was not loaded or evaluated.")


if __name__ == "__main__":
    main()
