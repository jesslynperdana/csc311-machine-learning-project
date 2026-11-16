import numpy as np
import pandas as pd
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, classification_report
import joblib

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from preprocessing import SurveyPreprocessor


def main():
    data_dir = project_root / "data_splits"
    train_path = data_dir / "train.csv"
    val_path = data_dir / "validation.csv"

    print(f"Loading data from {data_dir}...")
    train_df = pd.read_csv(train_path, dtype=str, keep_default_na=False)
    val_df = pd.read_csv(val_path, dtype=str, keep_default_na=False)

    target_col = "label"
    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    groups = train_df["id"].values
    X_train_raw = train_df.drop(columns=target_col)
    X_val_raw = val_df.drop(columns=target_col)

    print("Fitting preprocessor...")
    preprocessor = SurveyPreprocessor(project_dir=project_root)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    
    print(f"Processed training shape: {X_train.shape}")
    print(f"Processed validation shape: {X_val.shape}")

    print("Training Random Forest model...")
    pipe = Pipeline([
        ('var_thresh', VarianceThreshold(threshold=0.015)),
        ('rf', RandomForestClassifier(random_state=42))
    ])
    param_grid = {
        'rf__n_estimators': [300],
        'rf__max_depth': [8, 10, 12],
        'rf__min_samples_leaf': [4, 6, 8],
        'rf__max_features': ['sqrt', 0.1],
        'rf__max_samples': [0.7, 0.85],
        'rf__ccp_alpha': [0.002, 0.005, 0.01]
    }
    cv = StratifiedGroupKFold(n_splits= 5)
    grid_search = GridSearchCV(pipe, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train, groups=groups)

    train_preds = grid_search.predict(X_train)
    val_preds = grid_search.predict(X_val)

    train_acc = accuracy_score(y_train, train_preds)
    val_acc = accuracy_score(y_val, val_preds)

    print("\n--- Results ---")
    print("Best Hyperparameters:", grid_search.best_params_)
    print("Best CV Accuracy:", grid_search.best_score_)
    print(f"Training Accuracy: {train_acc * 100:.2f}%")
    print(f"Validation Accuracy: {val_acc * 100:.2f}%")
    print("\nValidation Classification Report:")
    print(classification_report(y_val, val_preds))

    output_dir = project_root / "results" / "random_forest"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "random_forest_model.joblib"
    joblib.dump(grid_search.best_estimator_, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()