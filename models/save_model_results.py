import datetime as dt
import getpass
import json
import numpy as np
import pandas as pd
import sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

CLASSES = ["Baseball", "Basketball", "Soccer"]


def evaluate_model(model, X, y):
    preds = model.predict(X)
    p, r, f, _ = precision_recall_fscore_support(y, preds, labels=CLASSES, zero_division=0)
    out = {
        "accuracy": float(accuracy_score(y, preds)),
        "macro_f1": float(f1_score(y, preds, labels=CLASSES, average="macro", zero_division=0)),
    }
    for i, c in enumerate(CLASSES):
        out[f"precision_{c}"] = float(p[i])
        out[f"recall_{c}"] = float(r[i])
        out[f"f1_{c}"] = float(f[i])
    return out


def save_experiments_csv(model_name, grid_search, X_train, y_train, X_val, y_val, output_dir):
    rows = []
    cv_results = grid_search.cv_results_
    for i, params in enumerate(cv_results["params"]):
        m = clone(grid_search.estimator).set_params(**params)
        m.fit(X_train, y_train)
        tr = evaluate_model(m, X_train, y_train)
        va = evaluate_model(m, X_val, y_val)
        row = {
            "date": dt.date.today().isoformat(),
            "person": getpass.getuser(),
            "model": model_name,
            **params,
            "cv_macro_f1": float(cv_results["mean_test_score"][i]),
            "cv_std": float(cv_results["std_test_score"][i]),
            "train_accuracy": tr["accuracy"],
            "train_macro_f1": tr["macro_f1"],
            **{f"val_{k}": v for k, v in va.items()},
            "seed": 42,
            "sklearn_version": sklearn.__version__,
        }
        rows.append(row)

    exp_df = pd.DataFrame(rows)
    exp_df.round(4).to_csv(output_dir / "experiments.csv", index=False)
    return exp_df


def save_best_config_json(grid_search, X_train, y_train, X_val, y_val, output_dir):
    best_pipe = grid_search.best_estimator_
    best_params = grid_search.best_params_
    tr_best = evaluate_model(best_pipe, X_train, y_train)
    va_best = evaluate_model(best_pipe, X_val, y_val)
    summary = {
        "chosen": best_params,
        "cv_macro_f1": round(float(grid_search.best_score_), 4),
        "train_accuracy": round(tr_best["accuracy"], 4),
        "train_macro_f1": round(tr_best["macro_f1"], 4),
        "val_accuracy": round(va_best["accuracy"], 4),
        "val_macro_f1": round(va_best["macro_f1"], 4),
        "val_per_class": {
            c: {
                "precision": round(va_best[f"precision_{c}"], 4),
                "recall": round(va_best[f"recall_{c}"], 4),
                "f1": round(va_best[f"f1_{c}"], 4),
            }
            for c in CLASSES
        },
        "sklearn_version": sklearn.__version__,
    }
    (output_dir / "best_config.json").write_text(json.dumps(summary, indent=2))


def save_confusion_matrix(best_pipe, X_val, y_val, curve_title, output_dir):
    val_preds = best_pipe.predict(X_val)
    cm = confusion_matrix(y_val, val_preds, labels=CLASSES)
    pd.DataFrame(
        cm,
        index=[f"true_{c}" for c in CLASSES],
        columns=[f"pred_{c}" for c in CLASSES],
    ).to_csv(output_dir / "confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES)
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title(f"{curve_title}: Validation Confusion Matrix")
    for r_idx in range(len(CLASSES)):
        for c_idx in range(len(CLASSES)):
            color = "white" if cm[r_idx, c_idx] > cm.max() / 2 else "black"
            ax.text(c_idx, r_idx, str(cm[r_idx, c_idx]), ha="center", va="center", color=color, fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)


def save_tuning_curves(exp_df, best_params, curve_param, curve_title, output_dir):
    sub = exp_df.copy()
    for k, v in best_params.items():
        if k != curve_param:
            sub = sub[sub[k].isna()] if v is None else sub[sub[k] == v]
    sub = sub.copy()
    sub["_x_label"] = sub[curve_param].apply(lambda x: "None" if pd.isna(x) else str(x))

    fig, ax = plt.subplots(figsize=(6, 3.6))
    x_pos = np.arange(len(sub))
    ax.plot(x_pos, sub["train_macro_f1"], "-o", color="#2a78d6", lw=2, label="training")
    ax.plot(x_pos, sub["val_macro_f1"], "-o", color="#eb6834", lw=2, label="validation")
    ax.plot(x_pos, sub["cv_macro_f1"], "--s", color="#2ca02c", lw=1.5, label="5-fold CV")
    best_str = "None" if best_params[curve_param] is None else str(best_params[curve_param])
    if best_str in list(sub["_x_label"]):
        ax.axvline(list(sub["_x_label"]).index(best_str), color="0.6", ls="--", lw=1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sub["_x_label"])
    ax.set_xlabel(curve_param.split("__")[-1])
    ax.set_ylabel("Macro-F1")
    ax.set_title(f"{curve_title}: Macro-F1 across {curve_param.split('__')[-1]}")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.9")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "tuning_curves.png", dpi=200)
    plt.close(fig)


def save_feature_importance(best_pipe, feature_names, output_dir):
    mask = best_pipe.named_steps["var_thresh"].get_support()
    kept_features = np.array(feature_names)[mask]
    estimator = best_pipe.steps[-1][1]

    if hasattr(estimator, "feature_importances_"):
        imp = pd.Series(estimator.feature_importances_, index=kept_features, name="importance")
        imp[imp > 0].sort_values(ascending=False).round(4).to_csv(output_dir / "feature_importance.csv")
    elif hasattr(estimator, "coef_"):
        coef_df = pd.DataFrame(estimator.coef_.T, index=kept_features, columns=[f"coef_{c}" for c in estimator.classes_])
        coef_df["max_abs_coef"] = coef_df.abs().max(axis=1)
        coef_df.sort_values("max_abs_coef", ascending=False).round(4).to_csv(output_dir / "feature_importance.csv")


def save_run_output_txt(log_lines, output_dir):
    (output_dir / "run_output.txt").write_text("\n".join(log_lines) + "\n")


def save_model_results(
    model_name,
    grid_search,
    X_train,
    y_train,
    X_val,
    y_val,
    feature_names,
    output_dir,
    curve_param,
    curve_title,
    log_lines,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    best_pipe = grid_search.best_estimator_

    exp_df = save_experiments_csv(model_name, grid_search, X_train, y_train, X_val, y_val, output_dir)
    save_best_config_json(grid_search, X_train, y_train, X_val, y_val, output_dir)
    save_confusion_matrix(best_pipe, X_val, y_val, curve_title, output_dir)
    save_tuning_curves(exp_df, grid_search.best_params_, curve_param, curve_title, output_dir)
    save_feature_importance(best_pipe, feature_names, output_dir)
    save_run_output_txt(log_lines, output_dir)