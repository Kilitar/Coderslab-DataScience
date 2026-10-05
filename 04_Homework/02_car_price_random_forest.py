import json
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    print("=== Random Forest Regression - Car Price Prediction ===")
    
    # 1. Paths
    base_dir = Path(".")
    csv_candidates = [
        base_dir / "data" / "car_data.csv",
        base_dir / "data" / "MAL_downloadable materials_session 2" / "Homework" / "car_data.csv"
    ]
    csv_path = next(p for p in csv_candidates if p.exists())
    out_dir = base_dir / "04_Homework" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Load dataset
    df_raw = pd.read_csv(csv_path)
    print(f"Loaded car_data with shape: {df_raw.shape}")
    
    # 3. Clean corrupted rows & technical columns
    # Drop rows where price is NaN (10 unparsed quoted lines in raw file)
    df = df_raw.dropna(subset=['price']).copy()
    
    # Drop technical identifiers
    drop_cols = ['Unnamed: 0', 'id', 'model']
    df_cleaned = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
    print(f"Dropped columns: {[c for c in drop_cols if c in df.columns]}")
    
    # 4. Correct categorical issues
    # Doors: Excel auto-converted '4-5' to '04-May' and '2-3' to '02-Mar'
    door_mapping = {'04-May': '4-5', '02-Mar': '2-3', '>5': '>5'}
    df_cleaned['doors'] = df_cleaned['doors'].replace(door_mapping)
    
    # Outlier filter: Row 16983 is an Opel Combo listed for 26,307,500 (scraper/typo error)
    # Keeping it severely biases leaf node averages; standard cleaning caps extreme errors
    raw_max_price = float(df_cleaned['price'].max())
    outlier_mask = df_cleaned['price'] < 20000000
    df_cleaned = df_cleaned[outlier_mask].copy()
    print(f"Filtered extreme price outlier (max was {raw_max_price:,.0f}). Remaining rows: {len(df_cleaned)}")
    
    # 5. One-hot encoding of categorical features
    cat_cols = [
        'manufacturer', 'category', 'leather_interior', 'fuel_type',
        'gear_box_type', 'drive_wheels', 'doors', 'wheel', 'color'
    ]
    df_encoded = pd.get_dummies(df_cleaned, columns=cat_cols, drop_first=True)
    feature_names = [c for c in df_encoded.columns if c != 'price']
    print(f"One-hot encoded features count: {len(feature_names)}")
    
    X = df_encoded[feature_names]
    y = df_cleaned['price']
    
    # 6. Train-test split (70:30 ratio, random_state=42)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df_cleaned.index, test_size=0.3, random_state=42
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # 7. Create RandomForestRegressor instance
    rf_regressor = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    # 8. Define hyperparameter grid
    params = {
        'max_depth': [10, 15, 20],
        'min_samples_leaf': [1, 2, 4],
        'n_estimators': [50, 100, 150]
    }
    
    # 9. GridSearchCV with scoring='neg_mean_squared_error'
    print("Running GridSearchCV with scoring='neg_mean_squared_error'...")
    grid_search = GridSearchCV(
        estimator=rf_regressor,
        param_grid=params,
        scoring='neg_mean_squared_error',
        cv=3,
        n_jobs=-1,
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    best_cv_neg_mse = float(grid_search.best_score_)
    best_cv_rmse = float(np.sqrt(-best_cv_neg_mse))
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {best_params}")
    print(f"Best CV RMSE: {best_cv_rmse:.2f} USD")
    
    # 10. Predictions & Evaluation on test set
    y_pred = best_model.predict(X_test)
    
    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))
    
    print("\n--- Test Set Metrics ---")
    print(f"MAE  : {mae:.2f} USD")
    print(f"MSE  : {mse:.2f}")
    print(f"RMSE : {rmse:.2f} USD")
    print(f"R^2  : {r2:.4f}")
    
    # 11. Feature importances
    importances = best_model.feature_importances_
    feat_imp = [
        {"feature": f, "importance": round(float(imp), 4)}
        for f, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    ]
    
    # 12. Top 10 CV configurations
    cv_res = pd.DataFrame(grid_search.cv_results_)
    top_configs = []
    for _, row in cv_res.sort_values(by="rank_test_score").head(10).iterrows():
        top_configs.append({
            "params": row["params"],
            "mean_test_rmse": round(float(np.sqrt(-row["mean_test_score"])), 2),
            "mean_test_neg_mse": round(float(row["mean_test_score"]), 2),
            "std_test_score": round(float(row["std_test_score"]), 2),
            "rank": int(row["rank_test_score"])
        })
    
    # 13. Sample test predictions
    test_df = df_cleaned.loc[idx_test].copy()
    test_df["actual_price"] = y_test.values
    test_df["predicted_price"] = y_pred
    test_df["abs_error"] = np.abs(test_df["actual_price"] - test_df["predicted_price"])
    test_df["pct_error"] = np.where(
        test_df["actual_price"] > 0,
        np.round((test_df["abs_error"] / test_df["actual_price"]) * 100, 1),
        0.0
    )
    
    samples = []
    for _, r in test_df.head(25).iterrows():
        samples.append({
            "manufacturer": str(r["manufacturer"]),
            "category": str(r["category"]),
            "prod_year": int(r["prod_year"]),
            "fuel_type": str(r["fuel_type"]),
            "engine_volume": float(r["engine_volume"]),
            "mileage": float(r["mileage"]),
            "gear_box_type": str(r["gear_box_type"]),
            "drive_wheels": str(r["drive_wheels"]),
            "airbags": int(r["airbags"]),
            "actual_price": float(r["actual_price"]),
            "predicted_price": round(float(r["predicted_price"]), 2),
            "abs_error": round(float(r["abs_error"]), 2),
            "pct_error": float(r["pct_error"])
        })
    
    # Residual percentiles for distribution analysis
    residuals = (y_test.values - y_pred).tolist()
    res_sample = residuals[::max(1, len(residuals) // 500)]
    
    # Actual vs predicted for scatter plot (sample 500 points for sleek rendering)
    scatter_sample = []
    step = max(1, len(y_test) // 500)
    for act, prd in zip(y_test.values[::step], y_pred[::step]):
        scatter_sample.append({
            "actual": round(float(act), 2),
            "predicted": round(float(prd), 2)
        })
    
    # 14. Save precomputed metadata and model
    results = {
        "metadata": {
            "dataset": "car_data.csv",
            "total_rows": len(df_raw),
            "clean_rows": len(df_cleaned),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "dropped_columns": [c for c in drop_cols if c in df.columns],
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "categorical_columns": cat_cols,
            "target_statistics": {
                "mean": round(float(y.mean()), 2),
                "median": round(float(y.median()), 2),
                "std": round(float(y.std()), 2),
                "min": round(float(y.min()), 2),
                "max": round(float(y.max()), 2)
            }
        },
        "grid_search": {
            "param_grid": params,
            "scoring_metric": "neg_mean_squared_error",
            "cv_folds": 3,
            "best_params": best_params,
            "best_cv_rmse": round(best_cv_rmse, 2),
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
        "sample_predictions": samples,
        "category_unique_values": {
            col: sorted(df_cleaned[col].dropna().unique().tolist())
            for col in cat_cols
        }
    }
    
    with open(out_dir / "car_price_rf_precomputed.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved precomputed results to {out_dir / 'car_price_rf_precomputed.json'}")
    
    joblib.dump(best_model, out_dir / "car_price_rf_model.joblib")
    print(f"Saved trained model to {out_dir / 'car_price_rf_model.joblib'}")

if __name__ == "__main__":
    main()
