import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

from preprocessing import SurveyPreprocessor


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data_splits"
    train_path = data_dir / "train.csv"
    val_path = data_dir / "validation.csv"

    print(f"Loading data from {data_dir}...")
    train_df = pd.read_csv(train_path, dtype=str, keep_default_na=False)
    val_df = pd.read_csv(val_path, dtype=str, keep_default_na=False)

    target_col = "sports"
    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    X_train_raw = train_df.drop(columns=target_col)
    X_val_raw = val_df.drop(columns=target_col)

    print("Fitting preprocessor...")
    preprocessor = SurveyPreprocessor(project_dir=project_root)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.fit(X_val_raw)

    print(f"Processed training shape: {X_train.shape}")
    print(f"Processed validation shape: {X_val.shape}")

    print("Training Logistic Regression model...")
    clf = LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs", C=1.0, random_state=42)
    clf.fit(X_train, y_train)

    train_preds = clf.predict(X_train)
    val_preds = clf.predict(X_val)

    train_acc = accuracy_score(y_train, train_preds)
    val_acc = accuracy_score(y_val, val_preds)

    print("\n--- Results ---")
    print(f"Training Accuracy: {train_acc * 100:.2f}%")
    print(f"Validation Accuracy: {val_acc * 100:.2f}%")
    print("\nValidation Classification Report:")
    print(classification_report(y_val, val_preds))

    output_dir = project_root / "results" / "logistic_regression"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "logistic_regression_model.joblib"
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()