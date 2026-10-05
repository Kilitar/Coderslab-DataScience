import json
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_score,
    recall_score, f1_score, accuracy_score, roc_auc_score,
    roc_curve, precision_recall_curve
)
import xgboost as xgb

def main():
    print("=== XGBoost Classification - Diabetes Prediction ===")
    
    # 1. Paths
    base_dir = Path(".")
    csv_candidates = [
        base_dir / "data" / "diabetes.csv",
        base_dir / "data" / "MAL_downloadable materials_session 2" / "Homework" / "diabetes.csv"
    ]
    csv_path = next(p for p in csv_candidates if p.exists())
    out_dir = base_dir / "04_Homework" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Load dataset
    df = pd.read_csv(csv_path)
    print(f"Loaded diabetes dataset with shape: {df.shape}")
    
    # 3. Analyze biological zeros (missing values encoded as 0)
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    zero_summary = {}
    for col in zero_cols:
        count_zero = int((df[col] == 0).sum())
        pct_zero = round(float((df[col] == 0).mean() * 100), 2)
        zero_summary[col] = {"zero_count": count_zero, "zero_percentage": pct_zero}
        print(f"  {col}: {count_zero} zeros ({pct_zero}%)")
        
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    feature_names = X.columns.tolist()
    
    # 4. Train-test split (80:20 ratio, random_state=21 as requested in assignment)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.2, random_state=21
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # 5. Baseline comparisons (Logistic Regression & Random Forest)
    lr = LogisticRegression(max_iter=1000, random_state=21)
    lr.fit(X_train, y_train)
    p_lr = lr.predict(X_test)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=21)
    rf.fit(X_train, y_train)
    p_rf = rf.predict(X_test)
    
    # 6. XGBoost Classifier instance with objective
    xgb_classifier = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=21,
        eval_metric='logloss',
        n_jobs=-1
    )
    
    # 7. Hyperparameter grid params
    params = {
        'max_depth': [3, 4, 5],
        'n_estimators': [50, 100, 150],
        'gamma': [0.0, 0.1, 0.5],
        'learning_rate': [0.05, 0.1, 0.2],
        'objective': ['binary:logistic']
    }
    
    # 8. GridSearchCV with scoring='precision'
    print("Running GridSearchCV with scoring='precision'...")
    grid_search = GridSearchCV(
        estimator=xgb_classifier,
        param_grid=params,
        scoring='precision',
        cv=5,
        n_jobs=-1,
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    best_cv_precision = float(grid_search.best_score_)
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {best_params}")
    print(f"Best CV precision: {best_cv_precision:.4f}")
    
    # 9. Predictions on test set
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    # 10. Metrics calculation
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_test, y_prob))
    cm = confusion_matrix(y_test, y_pred).tolist()
    clf_rep_dict = classification_report(y_test, y_pred, output_dict=True)
    clf_rep_text = classification_report(y_test, y_pred)
    
    print("\n--- Test Set Classification Report ---")
    print(clf_rep_text)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC AUC  : {auc:.4f}")
    
    # 11. Feature importances (Weight, Gain, Cover)
    booster = best_model.get_booster()
    score_weight = booster.get_score(importance_type='weight')
    score_gain = booster.get_score(importance_type='gain')
    score_cover = booster.get_score(importance_type='cover')
    
    feat_imp = []
    for f in feature_names:
        feat_imp.append({
            "feature": f,
            "weight": int(score_weight.get(f, 0)),
            "gain": round(float(score_gain.get(f, 0.0)), 4),
            "cover": round(float(score_cover.get(f, 0.0)), 4)
        })
    feat_imp.sort(key=lambda x: x["gain"], reverse=True)
    
    # 12. Top 10 CV configurations
    cv_res = pd.DataFrame(grid_search.cv_results_)
    top_configs = []
    for _, row in cv_res.sort_values(by="rank_test_score").head(10).iterrows():
        top_configs.append({
            "params": row["params"],
            "mean_test_precision": round(float(row["mean_test_score"]), 4),
            "std_test_score": round(float(row["std_test_score"]), 4),
            "rank": int(row["rank_test_score"])
        })
        
    # 13. ROC & PR curves
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    p_curve, r_curve, _ = precision_recall_curve(y_test, y_prob)
    
    # 14. Threshold tuning simulation
    thresholds = np.linspace(0.1, 0.9, 17)
    thresh_sim = []
    for th in thresholds:
        th_pred = (y_prob >= th).astype(int)
        th_p = float(precision_score(y_test, th_pred, zero_division=0))
        th_r = float(recall_score(y_test, th_pred, zero_division=0))
        th_f1 = float(f1_score(y_test, th_pred, zero_division=0))
        th_cm = confusion_matrix(y_test, th_pred)
        thresh_sim.append({
            "threshold": round(float(th), 2),
            "precision": round(th_p, 4),
            "recall": round(th_r, 4),
            "f1": round(th_f1, 4),
            "false_negatives": int(th_cm[1, 0]),
            "false_positives": int(th_cm[0, 1])
        })
        
    # 15. Sample test predictions
    test_df = df.loc[idx_test].copy()
    test_df["actual"] = y_test.values
    test_df["predicted"] = y_pred
    test_df["prob_diabetes"] = np.round(y_prob, 4)
    test_df["is_correct"] = test_df["actual"] == test_df["predicted"]
    
    samples = []
    for _, r in test_df.head(25).iterrows():
        samples.append({
            "pregnancies": int(r["Pregnancies"]),
            "glucose": float(r["Glucose"]),
            "blood_pressure": float(r["BloodPressure"]),
            "skin_thickness": float(r["SkinThickness"]),
            "insulin": float(r["Insulin"]),
            "bmi": float(r["BMI"]),
            "dpf": float(r["DiabetesPedigreeFunction"]),
            "age": int(r["Age"]),
            "actual": "Diabetik" if r["actual"] == 1 else "Zdravý",
            "predicted": "Diabetik" if r["predicted"] == 1 else "Zdravý",
            "prob_diabetes": float(r["prob_diabetes"]),
            "is_correct": bool(r["is_correct"])
        })
        
    # 16. Save precomputed results and joblib model
    results = {
        "metadata": {
            "dataset": "diabetes.csv",
            "total_patients": len(df),
            "train_patients": len(X_train),
            "test_patients": len(X_test),
            "features": feature_names,
            "target_distribution": {
                "healthy": int((y == 0).sum()),
                "diabetic": int((y == 1).sum()),
                "diabetes_rate": round(float(y.mean()), 4)
            },
            "biological_zeros": zero_summary
        },
        "baselines": {
            "logistic_regression": {
                "precision": round(float(precision_score(y_test, p_lr, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, p_lr, zero_division=0)), 4),
                "accuracy": round(float(accuracy_score(y_test, p_lr)), 4),
                "f1": round(float(f1_score(y_test, p_lr, zero_division=0)), 4)
            },
            "random_forest": {
                "precision": round(float(precision_score(y_test, p_rf, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, p_rf, zero_division=0)), 4),
                "accuracy": round(float(accuracy_score(y_test, p_rf)), 4),
                "f1": round(float(f1_score(y_test, p_rf, zero_division=0)), 4)
            }
        },
        "grid_search": {
            "param_grid": params,
            "scoring_metric": "precision",
            "cv_folds": 5,
            "best_params": best_params,
            "best_cv_precision": round(best_cv_precision, 4),
            "top_10_configs": top_configs
        },
        "test_metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "confusion_matrix": cm,
            "classification_report": clf_rep_dict,
            "classification_report_text": clf_rep_text
        },
        "feature_importances": feat_imp,
        "roc_curve": {
            "fpr": [round(float(x), 4) for x in fpr[::max(1, len(fpr) // 50)]],
            "tpr": [round(float(x), 4) for x in tpr[::max(1, len(tpr) // 50)]]
        },
        "precision_recall_curve": {
            "precision": [round(float(x), 4) for x in p_curve[::max(1, len(p_curve) // 50)]],
            "recall": [round(float(x), 4) for x in r_curve[::max(1, len(r_curve) // 50)]]
        },
        "threshold_tuning": thresh_sim,
        "sample_predictions": samples
    }
    
    with open(out_dir / "diabetes_xgb_precomputed.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved precomputed results to {out_dir / 'diabetes_xgb_precomputed.json'}")
    
    joblib.dump(best_model, out_dir / "diabetes_xgb_model.joblib")
    print(f"Saved trained model to {out_dir / 'diabetes_xgb_model.joblib'}")

if __name__ == "__main__":
    main()
