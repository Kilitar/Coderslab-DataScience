import json
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

def main():
    print("=== XGBoost Regression - Calories Burned Prediction ===")
    
    # 1. Paths
    base_dir = Path(".")
    csv_candidates = [
        base_dir / "data" / "calories_exercise_data.csv",
        base_dir / "data" / "MAL_downloadable materials_session 2" / "Homework" / "calories_exercise_data.csv"
    ]
    csv_path = next(p for p in csv_candidates if p.exists())
    out_dir = base_dir / "04_Homework" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Load dataset
    df = pd.read_csv(csv_path)
    print(f"Loaded calories dataset with shape: {df.shape}")
    
    # 3. Prepare data:
    # - Remove unnecessary columns (user_id)
    # - Encode categorical variables (gender -> 0/1)
    # - Normalize independent variables
    drop_cols = ['user_id', 'calories']
    X_raw = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['calories']
    
    # One-hot encode gender
    X_encoded = pd.get_dummies(X_raw, columns=['gender'], drop_first=True)
    feature_names = X_encoded.columns.tolist()
    print(f"Features after encoding ({len(feature_names)}): {feature_names}")
    
    # 4. Train-test split (70:30 ratio, random_state=42 as requested in assignment)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_encoded, y, df.index, test_size=0.3, random_state=42
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # 5. Normalization of independent variables (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names, index=X_test.index)
    
    # Baselines: Linear Regression and Random Forest
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    p_lr = lr.predict(X_test_scaled)
    mae_lr = float(mean_absolute_error(y_test, p_lr))
    r2_lr = float(r2_score(y_test, p_lr))
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    p_rf = rf.predict(X_test_scaled)
    mae_rf = float(mean_absolute_error(y_test, p_rf))
    r2_rf = float(r2_score(y_test, p_rf))
    
    # 6. Create XGBoost regressor instance
    xgb_regressor = xgb.XGBRegressor(
        random_state=42,
        n_jobs=-1
    )
    
    # 7. Define hyperparameter grid params (keys: max_depth, min_samples_leaf, n_estimators)
    # Note: min_samples_leaf is accepted via **kwargs and analyzed in expert critique
    params = {
        'max_depth': [3, 6, 9],
        'min_samples_leaf': [1, 2, 4],
        'n_estimators': [50, 100, 150]
    }
    
    # 8. RandomizedSearchCV with scoring='neg_mean_absolute_error'
    print("Running RandomizedSearchCV with scoring='neg_mean_absolute_error'...")
    rand_search = RandomizedSearchCV(
        estimator=xgb_regressor,
        param_distributions=params,
        n_iter=10,
        scoring='neg_mean_absolute_error',
        cv=3,
        random_state=42,
        n_jobs=-1,
        return_train_score=True
    )
    rand_search.fit(X_train_scaled, y_train)
    
    best_params = rand_search.best_params_
    best_cv_neg_mae = float(rand_search.best_score_)
    best_cv_mae = float(-best_cv_neg_mae)
    best_model = rand_search.best_estimator_
    print(f"Best parameters: {best_params}")
    print(f"Best CV MAE: {best_cv_mae:.2f} cal")
    
    # 9. Test set predictions & evaluation
    y_pred = best_model.predict(X_test_scaled)
    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))
    
    print("\n--- Test Set Evaluation ---")
    print(f"MAE  : {mae:.2f} calories")
    print(f"RMSE : {rmse:.2f} calories")
    print(f"R^2  : {r2:.4f}")
    
    # 10. Feature importances (Gain, Weight, Cover)
    booster = best_model.get_booster()
    score_weight = booster.get_score(importance_type='weight')
    score_gain = booster.get_score(importance_type='gain')
    score_cover = booster.get_score(importance_type='cover')
    
    feat_imp = []
    for f in feature_names:
        feat_imp.append({
            "feature": f,
            "weight": int(score_weight.get(f, 0)),
            "gain": round(float(score_gain.get(f, 0.0)), 2),
            "cover": round(float(score_cover.get(f, 0.0)), 2)
        })
    feat_imp.sort(key=lambda x: x["gain"], reverse=True)
    
    # 11. Top CV configurations
    cv_res = pd.DataFrame(rand_search.cv_results_)
    top_configs = []
    for _, row in cv_res.sort_values(by="rank_test_score").head(10).iterrows():
        top_configs.append({
            "params": row["params"],
            "mean_test_mae": round(float(-row["mean_test_score"]), 2),
            "std_test_score": round(float(row["std_test_score"]), 2),
            "rank": int(row["rank_test_score"])
        })
        
    # 12. Scatter sample for UI (500 representative points)
    step = max(1, len(y_test) // 500)
    scatter_sample = []
    for act, prd in zip(y_test.values[::step], y_pred[::step]):
        scatter_sample.append({
            "actual": round(float(act), 1),
            "predicted": round(float(prd), 1)
        })
        
    # 13. Residuals sample
    residuals = (y_test.values - y_pred).tolist()
    res_sample = residuals[::max(1, len(residuals) // 500)]
    
    # 14. Sample test predictions (25 sessions)
    test_df = df.loc[idx_test].copy()
    test_df["actual_calories"] = y_test.values
    test_df["predicted_calories"] = y_pred
    test_df["abs_error"] = np.abs(test_df["actual_calories"] - test_df["predicted_calories"])
    test_df["pct_error"] = np.where(
        test_df["actual_calories"] > 0,
        np.round((test_df["abs_error"] / test_df["actual_calories"]) * 100, 2),
        0.0
    )
    
    samples = []
    for _, r in test_df.head(25).iterrows():
        samples.append({
            "gender": str(r["gender"]),
            "age": int(r["age"]),
            "height": float(r["height"]),
            "weight": float(r["weight"]),
            "duration": float(r["duration"]),
            "heart_rate": float(r["heart_rate"]),
            "body_temp": float(r["body_temp"]),
            "actual_calories": float(r["actual_calories"]),
            "predicted_calories": round(float(r["predicted_calories"]), 1),
            "abs_error": round(float(r["abs_error"]), 2),
            "pct_error": float(r["pct_error"])
        })
        
    # 15. Scaler metadata for live interactive predictor
    scaler_means = {f: float(m) for f, m in zip(feature_names, scaler.mean_)}
    scaler_scales = {f: float(s) for f, s in zip(feature_names, scaler.scale_)}
    
    # 16. Save precomputed results and joblib model
    results = {
        "metadata": {
            "dataset": "calories_exercise_data.csv",
            "total_rows": len(df),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "dropped_columns": ["user_id"],
            "features": feature_names,
            "target_statistics": {
                "mean": round(float(y.mean()), 2),
                "median": round(float(y.median()), 2),
                "std": round(float(y.std()), 2),
                "min": round(float(y.min()), 2),
                "max": round(float(y.max()), 2)
            },
            "scaler": {
                "means": scaler_means,
                "scales": scaler_scales
            }
        },
        "baselines": {
            "linear_regression": {
                "mae": round(mae_lr, 2),
                "r2": round(r2_lr, 4)
            },
            "random_forest": {
                "mae": round(mae_rf, 2),
                "r2": round(r2_rf, 4)
            }
        },
        "random_search": {
            "param_distributions": params,
            "scoring_metric": "neg_mean_absolute_error",
            "n_iter": 10,
            "cv_folds": 3,
            "best_params": best_params,
            "best_cv_mae": round(best_cv_mae, 2),
            "top_10_configs": top_configs
        },
        "test_metrics": {
            "mae": round(mae, 2),
            "mse": round(mse, 2),
            "rmse": round(rmse, 2),
            "r2": round(r2, 4)
        },
        "feature_importances": feat_imp,
        "scatter_sample": scatter_sample,
        "residuals_sample": [round(float(r), 2) for r in res_sample],
        "sample_predictions": samples
    }
    
    with open(out_dir / "calories_xgb_precomputed.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved precomputed results to {out_dir / 'calories_xgb_precomputed.json'}")
    
    # Save model and scaler bundle
    bundle = {
        "model": best_model,
        "scaler": scaler,
        "feature_names": feature_names
    }
    joblib.dump(bundle, out_dir / "calories_xgb_model.joblib")
    print(f"Saved trained model and scaler to {out_dir / 'calories_xgb_model.joblib'}")

if __name__ == "__main__":
    main()
