import os
os.environ['KERAS_BACKEND'] = 'torch'
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import keras
from keras import layers, models

def main():
    print("=== Neural Networks Regression - Auto MPG Fuel Consumption Prediction ===")
    
    # 1. File paths
    base_dir = Path(".")
    csv_candidates = [
        base_dir / "data" / "auto_mpg.csv",
        base_dir / "data" / "MAL_downloadable materials_session 2" / "Homework" / "auto_mpg.csv"
    ]
    csv_path = next(p for p in csv_candidates if p.exists())
    out_dir = base_dir / "04_Homework" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Load dataset with Pandas
    df_raw = pd.read_csv(csv_path)
    print(f"Loaded raw dataset shape: {df_raw.shape}")
    
    # Clean column names (strip whitespace and replace space with underscore)
    df = df_raw.copy()
    df.columns = df.columns.str.strip().str.replace(' ', '_')
    
    # 3. Profiling insights & Data Cleaning:
    # A. Remove unnecessary variables (car_name is a free text string with high cardinality and typo noise)
    df = df.drop(columns=['car_name'])
    print("Removed unnecessary column: 'car_name'")
    
    # B. Map origin categorical variable: 1 -> USA, 2 -> Europe, 3 -> Japan
    origin_map = {1: 'USA', 2: 'Europe', 3: 'Japan'}
    df['origin'] = df['origin'].map(origin_map)
    print(f"Mapped origin values: {origin_map}")
    
    # C. Handle invalid values in horsepower (stored as '?' strings)
    invalid_mask = df['horsepower'] == '?'
    num_invalid = int(invalid_mask.sum())
    print(f"Detected {num_invalid} rows with invalid horsepower ('?')")
    df['horsepower'] = pd.to_numeric(df['horsepower'].replace('?', np.nan))
    
    # Drop rows with NaN (per hint: replace invalid values with NaN and remove rows)
    rows_before = len(df)
    df = df.dropna().reset_index(drop=True)
    rows_after = len(df)
    print(f"Dropped {rows_before - rows_after} invalid rows. Cleaned dataset has {rows_after} records.")
    
    # D. Transform categorical variables using pd.get_dummies()
    # Using drop_first=True to avoid dummy variable trap (USA is implicit reference baseline)
    df_encoded = pd.get_dummies(df, columns=['origin'], drop_first=True, dtype=int)
    feature_names = [col for col in df_encoded.columns if col != 'mpg']
    print(f"Engineered features ({len(feature_names)}): {feature_names}")
    
    # 4. Divide data into train and test sets (80:20 ratio, random_state=42)
    X = df_encoded[feature_names].values
    y = df_encoded['mpg'].values
    
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df_encoded.index, test_size=0.20, random_state=42
    )
    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    
    # 5. Normalization using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Benchmark models for comparison
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    p_lr = lr.predict(X_test_scaled)
    mae_lr = float(mean_absolute_error(y_test, p_lr))
    rmse_lr = float(np.sqrt(mean_squared_error(y_test, p_lr)))
    r2_lr = float(r2_score(y_test, p_lr))
    
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    p_ridge = ridge.predict(X_test_scaled)
    mae_ridge = float(mean_absolute_error(y_test, p_ridge))
    rmse_ridge = float(np.sqrt(mean_squared_error(y_test, p_ridge)))
    r2_ridge = float(r2_score(y_test, p_ridge))
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)
    p_rf = rf.predict(X_test)
    mae_rf = float(mean_absolute_error(y_test, p_rf))
    rmse_rf = float(np.sqrt(mean_squared_error(y_test, p_rf)))
    r2_rf = float(r2_score(y_test, p_rf))
    
    print("\n--- Benchmark Baseline Models on Test Set ---")
    print(f"Linear Regression: MAE = {mae_lr:.3f} MPG, RMSE = {rmse_lr:.3f} MPG, R² = {r2_lr:.4f}")
    print(f"Ridge Regression : MAE = {mae_ridge:.3f} MPG, RMSE = {rmse_ridge:.3f} MPG, R² = {r2_ridge:.4f}")
    print(f"Random Forest    : MAE = {mae_rf:.3f} MPG, RMSE = {rmse_rf:.3f} MPG, R² = {r2_rf:.4f}")
    
    # 7. Build Keras Neural Network
    keras.utils.set_random_seed(42)
    model = models.Sequential([
        layers.Input(shape=(X_train_scaled.shape[1],)),
        layers.Dense(64, activation='relu', name="dense_1"),
        layers.Dense(32, activation='relu', name="dense_2"),
        layers.Dense(1, name="output_mpg")
    ], name="Auto_MPG_MLP_Regressor")
    
    # 8. Compile model: Adam optimizer + MSE loss + MAE metric
    learning_rate = 0.01
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    model.summary()
    
    # 9. Train model
    epochs = 100
    batch_size = 16
    val_split = 0.20
    print(f"\nTraining Keras MLP: epochs={epochs}, batch_size={batch_size}, validation_split={val_split}...")
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=val_split,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # 10. Prediction on X_test
    y_pred = model.predict(X_test_scaled, verbose=0).flatten()
    
    # 11. Metric calculation on test set: MAE, RMSE, MSE, R2
    mse_test = float(mean_squared_error(y_test, y_pred))
    rmse_test = float(np.sqrt(mse_test))
    mae_test = float(mean_absolute_error(y_test, y_pred))
    r2_test = float(r2_score(y_test, y_pred))
    mape_test = float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100)
    
    print("\n=== Final Test Set Evaluation (Neural Network) ===")
    print(f"MAE  (Mean Absolute Error)     : {mae_test:.3f} MPG")
    print(f"RMSE (Root Mean Squared Error) : {rmse_test:.3f} MPG")
    print(f"MSE  (Mean Squared Error)      : {mse_test:.3f} MPG²")
    print(f"R²   (Coefficient of Determ.)  : {r2_test:.4f}")
    print(f"MAPE (Mean Absolute % Error)   : {mape_test:.2f} %")
    
    # 12. Residuals & Training History
    residuals = (y_test - y_pred).tolist()
    hist_dict = {
        "epochs": list(range(1, epochs + 1)),
        "loss_mse": [round(float(v), 4) for v in history.history["loss"]],
        "val_loss_mse": [round(float(v), 4) for v in history.history["val_loss"]],
        "mae": [round(float(v), 4) for v in history.history["mae"]],
        "val_mae": [round(float(v), 4) for v in history.history["val_mae"]]
    }
    
    # 13. Test sample predictions table (with MPG and metric L/100km conversion)
    # Conversion formula: L/100km = 235.214583 / MPG
    test_df = df.loc[idx_test].copy()
    test_df['actual_mpg'] = np.round(y_test, 2)
    test_df['pred_mpg'] = np.round(y_pred, 2)
    test_df['abs_error'] = np.round(np.abs(y_test - y_pred), 2)
    test_df['actual_l_100km'] = np.round(235.214583 / y_test, 2)
    test_df['pred_l_100km'] = np.round(235.214583 / y_pred, 2)
    
    sample_records = []
    for _, r in test_df.head(30).iterrows():
        sample_records.append({
            "cylinders": int(r['cylinders']),
            "displacement": float(r['displacement']),
            "horsepower": float(r['horsepower']),
            "weight": float(r['weight']),
            "acceleration": float(r['acceleration']),
            "model_year": int(r['model_year']) + 1900 if r['model_year'] < 100 else int(r['model_year']),
            "origin": str(r['origin']),
            "actual_mpg": float(r['actual_mpg']),
            "pred_mpg": float(r['pred_mpg']),
            "abs_error": float(r['abs_error']),
            "actual_l_100km": float(r['actual_l_100km']),
            "pred_l_100km": float(r['pred_l_100km'])
        })
        
    # Correlation analysis for profiling report summary
    corr_series = df.drop(columns=['origin']).corr()['mpg'].round(4).to_dict()
    
    # 14. Save precomputed results bundle
    results = {
        "metadata": {
            "dataset": "auto_mpg.csv",
            "source": "UCI Machine Learning Repository / StatLib Carnegie Mellon",
            "total_raw_rows": len(df_raw),
            "invalid_rows_removed": num_invalid,
            "clean_rows": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features": feature_names,
            "target": "mpg (Miles Per Gallon)"
        },
        "profiling_insights": {
            "missing_values_before": {"horsepower": 6, "all_other_columns": 0},
            "missing_values_after": 0,
            "correlations_with_mpg": corr_series,
            "origin_counts": df['origin'].value_counts().to_dict(),
            "target_stats": {
                "mean": round(float(df['mpg'].mean()), 2),
                "std": round(float(df['mpg'].std()), 2),
                "min": round(float(df['mpg'].min()), 2),
                "median": round(float(df['mpg'].median()), 2),
                "max": round(float(df['mpg'].max()), 2)
            }
        },
        "network_architecture": {
            "input_dim": len(feature_names),
            "layers": [
                {"name": "Input", "type": "InputLayer", "shape": [len(feature_names)]},
                {"name": "Dense 1", "type": "Dense", "units": 64, "activation": "relu", "params": len(feature_names) * 64 + 64},
                {"name": "Dense 2", "type": "Dense", "units": 32, "activation": "relu", "params": 64 * 32 + 32},
                {"name": "Output (mpg)", "type": "Dense", "units": 1, "activation": "linear", "params": 32 * 1 + 1}
            ],
            "total_params": int(model.count_params()),
            "optimizer": f"Adam (lr={learning_rate})",
            "loss_function": "MSE (Mean Squared Error)",
            "metrics": ["mae"],
            "epochs": epochs,
            "batch_size": batch_size,
            "validation_split": val_split
        },
        "model_comparison": {
            "linear_regression": {"mae": round(mae_lr, 3), "rmse": round(rmse_lr, 3), "r2": round(r2_lr, 4)},
            "ridge_regression": {"mae": round(mae_ridge, 3), "rmse": round(rmse_ridge, 3), "r2": round(r2_ridge, 4)},
            "random_forest": {"mae": round(mae_rf, 3), "rmse": round(rmse_rf, 3), "r2": round(r2_rf, 4)},
            "keras_neural_network": {"mae": round(mae_test, 3), "rmse": round(rmse_test, 3), "r2": round(r2_test, 4)}
        },
        "test_metrics": {
            "mae": round(mae_test, 3),
            "rmse": round(rmse_test, 3),
            "mse": round(mse_test, 3),
            "r2": round(r2_test, 4),
            "mape": round(mape_test, 2)
        },
        "training_history": hist_dict,
        "test_scatter": {
            "actual": [round(float(v), 2) for v in y_test],
            "predicted": [round(float(v), 2) for v in y_pred],
            "residuals": [round(float(v), 2) for v in residuals]
        },
        "sample_predictions": sample_records
    }
    
    with open(out_dir / "auto_mpg_nn_precomputed.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved precomputed results to {out_dir / 'auto_mpg_nn_precomputed.json'}")
    
    # Save trained Keras model and scaler
    model.save(out_dir / "auto_mpg_nn_model.keras")
    print(f"Saved Keras model to {out_dir / 'auto_mpg_nn_model.keras'}")
    
    scaler_bundle = {
        "scaler": scaler,
        "feature_names": feature_names,
        "origin_map": origin_map
    }
    joblib.dump(scaler_bundle, out_dir / "auto_mpg_scaler.joblib")
    print(f"Saved scaler bundle to {out_dir / 'auto_mpg_scaler.joblib'}")

if __name__ == "__main__":
    main()
