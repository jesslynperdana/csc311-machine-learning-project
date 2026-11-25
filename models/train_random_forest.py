import numpy as np
import pandas as pd
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from preprocessing import SurveyPreprocessor
from models.save_model_results import save_model_results


def main():
    data_dir = project_root / "data_splits"
    train_path = data_dir / "train.csv"
    val_path = data_dir / "validation.csv"

    log_lines = []
    def log(message):
        print(message)
        log_lines.append(message)

    log(f"Loading data from {data_dir}...")
    train_df = pd.read_csv(train_path, dtype=str, keep_default_na=False)
    val_df = pd.read_csv(val_path, dtype=str, keep_default_na=False)

    y_train = train_df["label"].values
    y_val = val_df["label"].values
    groups = train_df["id"].values

    log("Fitting preprocessor...")
    preprocessor = SurveyPreprocessor(project_dir=project_root)
    X_train = preprocessor.fit_transform(train_df)
    X_val = preprocessor.transform(val_df)
    feature_names = list(X_train.columns)
    
    log(f"Processed training shape: {X_train.shape}")
    log(f"Processed validation shape: {X_val.shape}")

    log("Training Random Forest model...")
    pipe = Pipeline([
        ("var_thresh", VarianceThreshold(threshold=0.01)),
        ("rf", RandomForestClassifier(criterion="entropy", random_state=42)),
    ])

    param_grid = {
        "rf__n_estimators": [300, 500],
        "rf__max_depth": [12, 16, None],
        "rf__min_samples_leaf": [2, 3],
        "rf__max_features": ["sqrt", 0.2],
    }
    cv = StratifiedGroupKFold(n_splits= 5)
    grid_search = GridSearchCV(pipe, param_grid=param_grid, cv=cv, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_train, y_train, groups=groups)

    train_preds = grid_search.predict(X_train)
    val_preds = grid_search.predict(X_val)

    train_acc = accuracy_score(y_train, train_preds)
    val_acc = accuracy_score(y_val, val_preds)
    val_macro_f1 = f1_score(y_val, val_preds, average='macro')

    log("\n--- Random Forest Results ---")
    log(f"Best Hyperparameters: {grid_search.best_params_}")
    log(f"Best CV Macro-F1: {grid_search.best_score_:.4f}")
    log(f"Training Accuracy: {train_acc * 100:.2f}%")
    log(f"Validation Accuracy: {val_acc * 100:.2f}%")
    log(f"Validation Macro-F1: {val_macro_f1:.4f}")
    log("\nValidation Classification Report:")
    log(classification_report(y_val, val_preds))

    output_dir = project_root / "results" / "random_forest"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "random_forest_model.joblib"
    joblib.dump(grid_search.best_estimator_, model_path)
    log(f"Model saved to {model_path}")

    save_model_results(
        model_name="random_forest",
        grid_search=grid_search,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_names=feature_names,
        output_dir=output_dir,
        curve_param="rf__max_depth",
        curve_title="Random Forest",
        log_lines=log_lines
    )
    log(f"Model results saved to {output_dir}")

if __name__ == "__main__":
    main()