import numpy as np
import pandas as pd
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
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
    X_train_raw = train_df.drop(columns=target_col)
    X_val_raw = val_df.drop(columns=target_col)

    print("Fitting preprocessor...")
    preprocessor = SurveyPreprocessor(project_dir=project_root)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    
    print(f"Processed training shape: {X_train.shape}")
    print(f"Processed validation shape: {X_val.shape}")

    print("Training Random Forest model...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_leaf': [1, 5, 10],
        'max_features': ['sqrt', 'log2']
    }
    cv = StratifiedKFold(n_splits= 5, shuffle=True, random_state=42)
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(rf, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

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