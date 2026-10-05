import json
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

# 1. Paths
base_dir = Path(".")
csv_path = base_dir / "data" / "titanic_data.csv"
out_dir = base_dir / "04_Homework" / "data"
out_dir.mkdir(parents=True, exist_ok=True)

# 2. Load dataset
df = pd.read_csv(csv_path)
print(f"Loaded Titanic dataset with shape: {df.shape}")

# 3. Drop unnecessary columns
drop_cols = ["passengerid", "name", "ticket", "cabin"]
df_cleaned = df.drop(columns=drop_cols)
print(f"Features after dropping {drop_cols}: {df_cleaned.columns.tolist()}")

# 4. Transform categorical variables
cat_cols = ["sex", "embarked", "title"]
df_encoded = pd.get_dummies(df_cleaned, columns=cat_cols, drop_first=True)
feature_names = [c for c in df_encoded.columns if c != "survived"]
print(f"Encoded feature names ({len(feature_names)}): {feature_names}")

X = df_encoded[feature_names]
y = df_encoded["survived"]

# 5. Train-test split (70:30, random_state=42)
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index, test_size=0.3, random_state=42, stratify=y
)
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# 6. RandomForestClassifier instance
rf_classifier = RandomForestClassifier(random_state=42, n_jobs=-1)

# 7. Hyperparameter grid (2 to 4 values per key)
params = {
    "max_depth": [4, 6, 8, 12],
    "min_samples_leaf": [1, 2, 4],
    "n_estimators": [50, 100, 150]
}

# 8. GridSearchCV with scoring='precision'
print("Running GridSearchCV with scoring='precision'...")
grid_search = GridSearchCV(
    estimator=rf_classifier,
    param_grid=params,
    scoring="precision",
    cv=5,
    n_jobs=-1,
    return_train_score=True
)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_cv_score = float(grid_search.best_score_)
best_model = grid_search.best_estimator_
print(f"Best parameters: {best_params}")
print(f"Best CV precision: {best_cv_score:.4f}")

# 9. Predictions on test set
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# 10. Metrics
acc = float(accuracy_score(y_test, y_pred))
prec = float(precision_score(y_test, y_pred))
rec = float(recall_score(y_test, y_pred))
f1 = float(f1_score(y_test, y_pred))
auc = float(roc_auc_score(y_test, y_prob))
cm = confusion_matrix(y_test, y_pred).tolist()
clf_rep_dict = classification_report(y_test, y_pred, output_dict=True)
clf_rep_text = classification_report(y_test, y_pred)
print("\nClassification Report:\n", clf_rep_text)

# 11. Feature importances
importances = best_model.feature_importances_
feat_imp = [
    {"feature": f, "importance": round(float(imp), 4)}
    for f, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
]

# 12. Curves (ROC & Precision-Recall)
fpr, tpr, _ = roc_curve(y_test, y_prob)
p_curve, r_curve, _ = precision_recall_curve(y_test, y_prob)

# 13. Grid Search results table (top 10 configurations)
cv_res = pd.DataFrame(grid_search.cv_results_)
top_configs = []
for _, row in cv_res.sort_values(by="rank_test_score").head(10).iterrows():
    top_configs.append({
        "params": row["params"],
        "mean_test_precision": round(float(row["mean_test_score"]), 4),
        "std_test_score": round(float(row["std_test_score"]), 4),
        "rank": int(row["rank_test_score"])
    })

# 14. Sample test predictions
test_df = df.loc[idx_test].copy()
test_df["actual_survived"] = y_test.values
test_df["predicted_survived"] = y_pred
test_df["prob_survived"] = np.round(y_prob, 4)
test_df["is_correct"] = test_df["actual_survived"] == test_df["predicted_survived"]

samples = []
for _, r in test_df.head(20).iterrows():
    samples.append({
        "name": str(r["name"]),
        "pclass": int(r["pclass"]),
        "sex": str(r["sex"]),
        "age": float(r["age"]),
        "fare": float(r["fare"]),
        "title": str(r["title"]),
        "family_size": int(r["family_size"]),
        "actual": "Přežil" if r["actual_survived"] == 1 else "Nepřežil",
        "predicted": "Přežil" if r["predicted_survived"] == 1 else "Nepřežil",
        "prob_survived": float(r["prob_survived"]),
        "is_correct": bool(r["is_correct"])
    })

results = {
    "metadata": {
        "dataset": "titanic_data.csv",
        "total_rows": len(df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "dropped_columns": drop_cols,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "target_distribution": {
            "died": int((y == 0).sum()),
            "survived": int((y == 1).sum()),
            "survival_rate": round(float(y.mean()), 4)
        }
    },
    "grid_search": {
        "param_grid": params,
        "scoring_metric": "precision",
        "cv_folds": 5,
        "best_params": best_params,
        "best_cv_precision": round(best_cv_score, 4),
        "top_10_configs": top_configs
    },
    "test_metrics": {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": cm,
        "classification_report": clf_rep_dict
    },
    "feature_importances": feat_imp,
    "roc_curve": {
        "fpr": [round(float(x), 4) for x in fpr[::max(1, len(fpr)//50)]],
        "tpr": [round(float(x), 4) for x in tpr[::max(1, len(tpr)//50)]]
    },
    "precision_recall_curve": {
        "precision": [round(float(x), 4) for x in p_curve[::max(1, len(p_curve)//50)]],
        "recall": [round(float(x), 4) for x in r_curve[::max(1, len(r_curve)//50)]]
    },
    "sample_predictions": samples
}

with open(out_dir / "titanic_rf_precomputed.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

joblib.dump(best_model, out_dir / "titanic_rf_model.joblib")
print("Saved titanic_rf_precomputed.json and titanic_rf_model.joblib!")
