"""
Skript pro výpočet a předpočtení výsledků pro Cvičení 1: Random Forest - Klasifikace onemocnění srdce (heart_data_normalized.csv)
Ukládá kompletní výsledky do 03_Advanced_ML_Neural_Networks/data/heart_rf_exercise_1_precomputed.json
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_score,
    recall_score, f1_score, accuracy_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.inspection import permutation_importance


def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "MAL_downloadable materials_session 2" / "Day 3" / "heart_data_normalized.csv"
    out_dir = base_dir / "03_Advanced_ML_Neural_Networks" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "heart_rf_exercise_1_precomputed.json"

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    X = df.drop("ahd_yes", axis=1)
    y = df["ahd_yes"]
    feature_names = X.columns.tolist()

    # Krok 4: Split 70:30, random_state=42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Baseline 1: Single Decision Tree
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    y_prob_dt = dt.predict_proba(X_test)[:, 1]

    # Baseline 2: Default Random Forest
    rf_def = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf_def.fit(X_train, y_train)
    y_pred_rf_def = rf_def.predict(X_test)
    y_prob_rf_def = rf_def.predict_proba(X_test)[:, 1]

    # Krok 5: rf_classifier instance
    rf_classifier = RandomForestClassifier(random_state=42, n_jobs=-1)

    # Krok 6: params dictionary
    params = {
        "max_depth": [3, 5, 7, None],
        "min_samples_leaf": [1, 2, 4],
        "n_estimators": [50, 100, 150]
    }

    # Krok 7: GridSearchCV with scoring='precision'
    grid_search = GridSearchCV(
        estimator=rf_classifier,
        param_grid=params,
        scoring="precision",
        cv=5,
        n_jobs=-1,
        return_train_score=True
    )
    print("Fitting GridSearchCV...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = float(grid_search.best_score_)
    print("Best params:", best_params, "Best CV precision:", best_cv_score)

    # Krok 9: Test predictions
    y_pred_test = best_model.predict(X_test)
    y_prob_test = best_model.predict_proba(X_test)[:, 1]

    # Krok 10: Classification report and metrics
    rep_dict = classification_report(y_test, y_pred_test, output_dict=True)
    rep_text = classification_report(y_test, y_pred_test)

    cm_test = confusion_matrix(y_test, y_pred_test).tolist()
    cm_dt = confusion_matrix(y_test, y_pred_dt).tolist()
    cm_def = confusion_matrix(y_test, y_pred_rf_def).tolist()

    # ROC and PR curves for best model
    fpr, tpr, roc_thresh = roc_curve(y_test, y_prob_test)
    roc_auc_val = float(auc(fpr, tpr))

    prec_curve, rec_curve, pr_thresh = precision_recall_curve(y_test, y_prob_test)
    pr_auc_val = float(average_precision_score(y_test, y_prob_test))

    # Feature importances: MDI vs Permutation
    mdi_importances = best_model.feature_importances_.tolist()
    perm_imp = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    perm_mean = perm_imp.importances_mean.tolist()
    perm_std = perm_imp.importances_std.tolist()

    # Grid search results table
    cv_res = grid_search.cv_results_
    grid_rows = []
    for i in range(len(cv_res["params"])):
        grid_rows.append({
            "param_max_depth": str(cv_res["param_max_depth"][i]),
            "param_min_samples_leaf": int(cv_res["param_min_samples_leaf"][i]),
            "param_n_estimators": int(cv_res["param_n_estimators"][i]),
            "mean_test_precision": float(cv_res["mean_test_score"][i]),
            "std_test_precision": float(cv_res["std_test_score"][i]),
            "rank_test_score": int(cv_res["rank_test_score"][i])
        })
    grid_rows.sort(key=lambda x: x["rank_test_score"])

    # Threshold sweep analysis (Clinical perspective)
    thresh_sweep = []
    for th in np.arange(0.1, 0.95, 0.05):
        y_th_pred = (y_prob_test >= th).astype(int)
        p = precision_score(y_test, y_th_pred, zero_division=0)
        r = recall_score(y_test, y_th_pred, zero_division=0)
        f1 = f1_score(y_test, y_th_pred, zero_division=0)
        cm_th = confusion_matrix(y_test, y_th_pred)
        fn_count = int(cm_th[1, 0])
        fp_count = int(cm_th[0, 1])
        thresh_sweep.append({
            "threshold": round(float(th), 2),
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1_score": round(float(f1), 4),
            "false_negatives": fn_count,
            "false_positives": fp_count
        })

    # Output structure
    payload = {
        "metadata": {
            "dataset_shape": [int(df.shape[0]), int(df.shape[1])],
            "train_shape": [int(X_train.shape[0]), int(X_train.shape[1])],
            "test_shape": [int(X_test.shape[0]), int(X_test.shape[1])],
            "feature_names": feature_names,
            "target_name": "ahd_yes",
            "class_distribution_all": {str(k): int(v) for k, v in y.value_counts().items()},
            "class_distribution_test": {str(k): int(v) for k, v in y_test.value_counts().items()},
        },
        "head_10": df.head(10).to_dict(orient="records"),
        "baseline_decision_tree": {
            "accuracy": float(accuracy_score(y_test, y_pred_dt)),
            "precision": float(precision_score(y_test, y_pred_dt)),
            "recall": float(recall_score(y_test, y_pred_dt)),
            "f1": float(f1_score(y_test, y_pred_dt)),
            "confusion_matrix": cm_dt
        },
        "baseline_default_rf": {
            "accuracy": float(accuracy_score(y_test, y_pred_rf_def)),
            "precision": float(precision_score(y_test, y_pred_rf_def)),
            "recall": float(recall_score(y_test, y_pred_rf_def)),
            "f1": float(f1_score(y_test, y_pred_rf_def)),
            "confusion_matrix": cm_def
        },
        "grid_search": {
            "params_grid": {
                "max_depth": ["3", "5", "7", "None"],
                "min_samples_leaf": [1, 2, 4],
                "n_estimators": [50, 100, 150]
            },
            "best_params": {
                "max_depth": best_params["max_depth"],
                "min_samples_leaf": best_params["min_samples_leaf"],
                "n_estimators": best_params["n_estimators"]
            },
            "best_cv_precision": best_cv_score,
            "top_results": grid_rows[:15]
        },
        "optimal_model_test": {
            "accuracy": float(accuracy_score(y_test, y_pred_test)),
            "precision": float(precision_score(y_test, y_pred_test)),
            "recall": float(recall_score(y_test, y_pred_test)),
            "f1": float(f1_score(y_test, y_pred_test)),
            "confusion_matrix": cm_test,
            "classification_report_dict": rep_dict,
            "classification_report_str": rep_text,
            "roc_auc": roc_auc_val,
            "pr_auc": pr_auc_val,
            "fpr": [float(x) for x in fpr],
            "tpr": [float(x) for x in tpr],
            "roc_thresholds": [float(x) for x in roc_thresh],
            "pr_precision": [float(x) for x in prec_curve],
            "pr_recall": [float(x) for x in rec_curve]
        },
        "feature_importances": {
            "features": feature_names,
            "mdi": mdi_importances,
            "permutation_mean": perm_mean,
            "permutation_std": perm_std
        },
        "threshold_sweep": thresh_sweep
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Successfully saved precomputed results to {out_file}")


if __name__ == "__main__":
    main()
