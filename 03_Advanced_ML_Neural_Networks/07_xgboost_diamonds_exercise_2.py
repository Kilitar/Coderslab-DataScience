"""
Skript pro výpočet a předpočtení výsledků pro Cvičení 2: XGBoost - Regrese cen diamantů (diamonds_preprocessed.csv)
Ukládá kompletní výsledky do 03_Advanced_ML_Neural_Networks/data/diamonds_xgb_exercise_2_precomputed.json
"""

import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb


def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "MAL_downloadable materials_session 2" / "Day 3" / "diamonds_preprocessed.csv"
    out_dir = base_dir / "03_Advanced_ML_Neural_Networks" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "diamonds_xgb_exercise_2_precomputed.json"

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    X = df.drop("price", axis=1)
    y = df["price"]
    feature_names = X.columns.tolist()

    # Krok 4: Split 70:30, random_state=42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 1. Baseline: Single Decision Tree
    print("Fitting baseline DecisionTreeRegressor...")
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    dt_mae = float(mean_absolute_error(y_test, y_pred_dt))
    dt_rmse = float(mean_squared_error(y_test, y_pred_dt) ** 0.5)
    dt_r2 = float(r2_score(y_test, y_pred_dt))

    # 2. Baseline: Random Forest (from Ex 2)
    print("Fitting baseline RandomForestRegressor (100 trees)...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rf_mae = float(mean_absolute_error(y_test, y_pred_rf))
    rf_rmse = float(mean_squared_error(y_test, y_pred_rf) ** 0.5)
    rf_r2 = float(r2_score(y_test, y_pred_rf))

    # 3. Krok 5: xgb_regressor s random_state=42 a n_jobs=-1
    print("Instantiating xgb_regressor...")
    xgb_regressor = xgb.XGBRegressor(random_state=42, n_jobs=-1)

    # 4. Krok 6: params dictionary (dle zadání: max_depth, min_samples_leaf, n_estimators)
    # Poznámka: min_samples_leaf je v zadání doslovně, ačkoliv ho XGBoost interně nepoužívá
    params = {
        "max_depth": [3, 6, 9],
        "min_samples_leaf": [1, 2, 4],
        "n_estimators": [50, 100, 150]
    }

    # 5. Krok 7: RandomizedSearchCV s scoring='neg_mean_absolute_error'
    print("Fitting RandomizedSearchCV (školní zadání s min_samples_leaf)...")
    t0 = time.time()
    random_search = RandomizedSearchCV(
        estimator=xgb_regressor,
        param_distributions=params,
        scoring="neg_mean_absolute_error",
        n_iter=8,
        cv=3,
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    fit_duration_s = float(time.time() - t0)

    best_school_model = random_search.best_estimator_
    best_school_params = random_search.best_params_
    best_school_cv_mae = float(-random_search.best_score_)

    # 6. Predikce a MAE na testovacích datech
    y_pred_school = best_school_model.predict(X_test)
    school_mae = float(mean_absolute_error(y_test, y_pred_school))
    school_rmse = float(mean_squared_error(y_test, y_pred_school) ** 0.5)
    school_r2 = float(r2_score(y_test, y_pred_school))

    # 7. SOTA / Expert Model: Použití nativních parametrů XGBoost (min_child_weight, colsample_bytree, learning_rate)
    print("Fitting SOTA Tuned XGBoost (s min_child_weight a colsample_bytree)...")
    t_sota0 = time.time()
    sota_params = {
        "max_depth": [6, 8, 10],
        "n_estimators": [100, 150, 200],
        "learning_rate": [0.05, 0.1, 0.15],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }
    sota_rs = RandomizedSearchCV(
        estimator=xgb.XGBRegressor(random_state=42, n_jobs=-1),
        param_distributions=sota_params,
        scoring="neg_mean_absolute_error",
        n_iter=8,
        cv=3,
        random_state=42,
        n_jobs=-1
    )
    sota_rs.fit(X_train, y_train)
    sota_duration_s = float(time.time() - t_sota0)
    best_sota_model = sota_rs.best_estimator_
    best_sota_params = sota_rs.best_params_
    y_pred_sota = best_sota_model.predict(X_test)
    sota_mae = float(mean_absolute_error(y_test, y_pred_sota))
    sota_rmse = float(mean_squared_error(y_test, y_pred_sota) ** 0.5)
    sota_r2 = float(r2_score(y_test, y_pred_sota))

    # 8. Feature Importances
    school_importances = [float(x) for x in best_school_model.feature_importances_]
    sota_importances = [float(x) for x in best_sota_model.feature_importances_]

    # 9. Analýza chyb podle cenových pásem
    price_bins = [0, 1000, 3000, 6000, 10000, 25000]
    bin_labels = ["Do 1 000 USD", "1 000 – 3 000 USD", "3 000 – 6 000 USD", "6 000 – 10 000 USD", "Nad 10 000 USD"]
    y_test_np = y_test.values
    test_bins = pd.cut(y_test_np, bins=price_bins, labels=bin_labels)

    bracket_analysis = []
    for lbl in bin_labels:
        mask = (test_bins == lbl)
        cnt = int(mask.sum())
        if cnt > 0:
            b_mae_dt = float(mean_absolute_error(y_test_np[mask], y_pred_dt[mask]))
            b_mae_rf = float(mean_absolute_error(y_test_np[mask], y_pred_rf[mask]))
            b_mae_school = float(mean_absolute_error(y_test_np[mask], y_pred_school[mask]))
            b_mae_sota = float(mean_absolute_error(y_test_np[mask], y_pred_sota[mask]))
            bracket_analysis.append({
                "bracket": lbl,
                "count": cnt,
                "mae_dt": b_mae_dt,
                "mae_rf": b_mae_rf,
                "mae_school": b_mae_school,
                "mae_sota": b_mae_sota
            })

    # 10. Sample predikcí pro interaktivní tabulku a graf reziduí
    sample_indices = np.random.RandomState(42).choice(len(y_test), size=400, replace=False)
    residuals_sample = [
        {
            "actual": float(y_test_np[i]),
            "pred_school": float(y_pred_school[i]),
            "pred_sota": float(y_pred_sota[i]),
            "pred_rf": float(y_pred_rf[i]),
            "pred_dt": float(y_pred_dt[i]),
            "res_school": float(y_test_np[i] - y_pred_school[i]),
            "res_sota": float(y_test_np[i] - y_pred_sota[i]),
            "carat": float(X_test.iloc[i]["carat"]),
            "clarity": float(X_test.iloc[i]["clarity"]),
            "color": float(X_test.iloc[i]["color"])
        }
        for i in sample_indices
    ]

    # Výsledný payload
    results = {
        "dataset": {
            "n_total": len(df),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "features": feature_names
        },
        "school_model": {
            "best_params": best_school_params,
            "cv_mae": best_school_cv_mae,
            "test_mae": school_mae,
            "test_rmse": school_rmse,
            "test_r2": school_r2,
            "fit_duration_s": fit_duration_s,
            "feature_importances": dict(zip(feature_names, school_importances))
        },
        "sota_model": {
            "best_params": best_sota_params,
            "test_mae": sota_mae,
            "test_rmse": sota_rmse,
            "test_r2": sota_r2,
            "fit_duration_s": sota_duration_s,
            "feature_importances": dict(zip(feature_names, sota_importances))
        },
        "baselines": {
            "dt": {"mae": dt_mae, "rmse": dt_rmse, "r2": dt_r2},
            "rf": {"mae": rf_mae, "rmse": rf_rmse, "r2": rf_r2}
        },
        "bracket_analysis": bracket_analysis,
        "sample_residuals": residuals_sample
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Precomputed results saved successfully to {out_file}!")
    print(f"School Test MAE: {school_mae:.2f} USD (R2: {school_r2:.4f})")
    print(f"SOTA Test MAE:   {sota_mae:.2f} USD (R2: {sota_r2:.4f})")
    print(f"Random Forest MAE: {rf_mae:.2f} USD (R2: {rf_r2:.4f})")
    print(f"Decision Tree MAE: {dt_mae:.2f} USD (R2: {dt_r2:.4f})")


if __name__ == "__main__":
    main()
