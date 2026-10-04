"""
Skript pro výpočet a předpočtení výsledků pro Cvičení 2: Random Forest - Regrese cen diamantů (diamonds_preprocessed.csv)
Ukládá kompletní výsledky do 03_Advanced_ML_Neural_Networks/data/diamonds_rf_exercise_2_precomputed.json
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance


def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "MAL_downloadable materials_session 2" / "Day 3" / "diamonds_preprocessed.csv"
    out_dir = base_dir / "03_Advanced_ML_Neural_Networks" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "diamonds_rf_exercise_2_precomputed.json"

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    X = df.drop("price", axis=1)
    y = df["price"]
    feature_names = X.columns.tolist()

    # Krok 4: Split 70:30, random_state=42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Baseline 1: Single Decision Tree
    print("Fitting baseline DecisionTreeRegressor...")
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    dt_mae = float(mean_absolute_error(y_test, y_pred_dt))
    dt_mse = float(mean_squared_error(y_test, y_pred_dt))
    dt_rmse = float(dt_mse ** 0.5)
    dt_r2 = float(r2_score(y_test, y_pred_dt))

    # Krok 5: rf_regressor instance
    rf_regressor = RandomForestRegressor(random_state=42, n_jobs=-1)

    # Krok 6: params dictionary
    params = {
        "max_depth": [10, 15, 20, None],
        "min_samples_leaf": [1, 2, 4],
        "n_estimators": [50, 100, 150]
    }

    # Krok 7: RandomizedSearchCV with scoring='neg_mean_squared_error'
    print("Fitting RandomizedSearchCV on diamonds (37,729 train rows)...")
    random_search = RandomizedSearchCV(
        estimator=rf_regressor,
        param_distributions=params,
        scoring="neg_mean_squared_error",
        n_iter=8,
        cv=3,
        random_state=42,
        n_jobs=-1,
        return_train_score=True
    )
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_cv_neg_mse = float(random_search.best_score_)
    best_cv_rmse = float((-best_cv_neg_mse) ** 0.5)
    print("Best params:", best_params, "Best CV RMSE:", best_cv_rmse)

    # Krok 9: Test predictions
    print("Evaluating optimal model on test set...")
    y_pred_test = best_model.predict(X_test)
    test_mae = float(mean_absolute_error(y_test, y_pred_test))
    test_mse = float(mean_squared_error(y_test, y_pred_test))
    test_rmse = float(test_mse ** 0.5)
    test_r2 = float(r2_score(y_test, y_pred_test))

    print(f"Optimal RF -> Test MAE: {test_mae:.2f} USD, RMSE: {test_rmse:.2f} USD, R2: {test_r2:.4f}")

    # Feature importances: MDI vs Permutation (on subset of 2000 test samples for fast calculation)
    mdi_importances = best_model.feature_importances_.tolist()
    perm_sub_idx = np.random.RandomState(42).choice(len(X_test), size=min(2000, len(X_test)), replace=False)
    X_test_sub = X_test.iloc[perm_sub_idx]
    y_test_sub = y_test.iloc[perm_sub_idx]
    perm_imp = permutation_importance(best_model, X_test_sub, y_test_sub, n_repeats=5, random_state=42, n_jobs=-1, scoring="neg_mean_absolute_error")
    perm_mean = perm_imp.importances_mean.tolist()
    perm_std = perm_imp.importances_std.tolist()

    # Search results table
    cv_res = random_search.cv_results_
    search_rows = []
    for i in range(len(cv_res["params"])):
        neg_mse = float(cv_res["mean_test_score"][i])
        rmse_val = float((-neg_mse) ** 0.5)
        search_rows.append({
            "param_max_depth": str(cv_res["param_max_depth"][i]),
            "param_min_samples_leaf": int(cv_res["param_min_samples_leaf"][i]),
            "param_n_estimators": int(cv_res["param_n_estimators"][i]),
            "mean_cv_rmse": round(rmse_val, 2),
            "rank_test_score": int(cv_res["rank_test_score"][i])
        })
    search_rows.sort(key=lambda x: x["rank_test_score"])

    # Sample predictions comparison for UI (first 10 test rows)
    sample_df = pd.DataFrame({
        "carat": X_test.iloc[:10]["carat"].values,
        "cut": X_test.iloc[:10]["cut"].values,
        "color": X_test.iloc[:10]["color"].values,
        "clarity": X_test.iloc[:10]["clarity"].values,
        "Skutečná cena (USD)": y_test.iloc[:10].values,
        "Strom predikce (USD)": np.round(y_pred_dt[:10], 1),
        "Random Forest predikce (USD)": np.round(y_pred_test[:10], 1),
        "Chyba RF (USD)": np.round(np.abs(y_test.iloc[:10].values - y_pred_test[:10]), 1)
    }).to_dict(orient="records")

    # Residuals distribution and error by price bins
    test_analysis_df = pd.DataFrame({
        "actual": y_test.values,
        "predicted": y_pred_test,
        "abs_error": np.abs(y_test.values - y_pred_test)
    })
    test_analysis_df["price_bin"] = pd.qcut(test_analysis_df["actual"], q=5, labels=["Velmi levné (< 1k)", "Levné (1k–2k)", "Střední (2k–4.5k)", "Drahé (4.5k–9k)", "Luxusní (> 9k)"])
    bin_stats = test_analysis_df.groupby("price_bin", observed=True)["abs_error"].agg(["mean", "median", "count"]).reset_index()
    bin_stats.columns = ["Cenová kategorie", "Průměrná MAE (USD)", "Medián MAE (USD)", "Počet diamantů"]
    bin_stats["Průměrná MAE (USD)"] = bin_stats["Průměrná MAE (USD)"].round(2)
    bin_stats["Medián MAE (USD)"] = bin_stats["Medián MAE (USD)"].round(2)

    payload = {
        "metadata": {
            "dataset_shape": [int(df.shape[0]), int(df.shape[1])],
            "train_shape": [int(X_train.shape[0]), int(X_train.shape[1])],
            "test_shape": [int(X_test.shape[0]), int(X_test.shape[1])],
            "feature_names": feature_names,
            "target_name": "price",
            "price_min": float(y.min()),
            "price_max": float(y.max()),
            "price_mean": float(y.mean()),
            "price_median": float(y.median())
        },
        "head_10": df.head(10).to_dict(orient="records"),
        "baseline_decision_tree": {
            "mae": dt_mae,
            "rmse": dt_rmse,
            "r2": dt_r2
        },
        "random_search": {
            "params_grid": {
                "max_depth": ["10", "15", "20", "None"],
                "min_samples_leaf": [1, 2, 4],
                "n_estimators": [50, 100, 150]
            },
            "best_params": {
                "max_depth": best_params["max_depth"],
                "min_samples_leaf": best_params["min_samples_leaf"],
                "n_estimators": best_params["n_estimators"]
            },
            "best_cv_rmse": best_cv_rmse,
            "top_results": search_rows
        },
        "optimal_model_test": {
            "mae": test_mae,
            "rmse": test_rmse,
            "r2": test_r2,
            "improvement_mae_usd": dt_mae - test_mae,
            "improvement_mae_pct": ((dt_mae - test_mae) / dt_mae) * 100
        },
        "sample_predictions": sample_df,
        "feature_importances": {
            "features": feature_names,
            "mdi": mdi_importances,
            "permutation_mean": perm_mean,
            "permutation_std": perm_std
        },
        "price_bin_performance": bin_stats.to_dict(orient="records")
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Successfully saved precomputed results to {out_file}")


if __name__ == "__main__":
    main()
