"""
02_hyperparameters_diamonds_exercise_1.py
=========================================
Vypracování cvičení 1: Optimalizace hyperparametrů na cenách diamantů
-------------------------------------------------------------------
1. Načtení dat diamonds.csv a kódování kategorií LabelEncoderem.
2. Rozdělení na trénovací a testovací sadu v poměru 70/30 (random_state=42).
3. Původní model z 1. dne: max_depth=13, default criterion (squared_error).
4. Pomocí GridSearchCV nalezení optimální dvojice: max_depth a criterion.
5. Úprava funkce train_and_test_decision_tree(hyperparameters: dict),
   aby přijímala slovník hyperparametrů.
6. Natrénování modelu s optimálními parametry a ověření zlepšení metrik.
7. Uložení JSON cache a diagnostických grafů.
"""

import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def run_diamonds_hyperparameters_exercise_1():
    print("=" * 70)
    print("💎 CVIČENÍ 1: OPTIMALIZACE HYPERPARAMETRŮ ROZHODOVACÍHO STROMU (DIAMONDS)")
    print("=" * 70)

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    plots_dir = base_dir / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Načtení datasetu
    csv_candidates = [
        data_dir / "diamonds.csv",
        base_dir.parent / "01_Regression" / "data" / "diamonds.csv",
        base_dir.parent / "data" / "MAL_downloadable materials_session 1" / "Homework" / "diamonds.csv",
    ]
    csv_path = None
    for p in csv_candidates:
        if p.exists():
            csv_path = p
            break

    if csv_path is None:
        raise FileNotFoundError("Soubor diamonds.csv nebyl nalezen!")

    print(f"📂 Načítám data z: {csv_path}")
    diamonds_df = pd.read_csv(csv_path)
    if "Unnamed: 0" in diamonds_df.columns:
        diamonds_df = diamonds_df.drop("Unnamed: 0", axis=1)

    print(f"Rozměry datasetu: {diamonds_df.shape}")

    # Kopie a kódování kategorií LabelEncoderem
    diamonds_df_copy = diamonds_df.copy()
    columns_to_encode = ["cut", "color", "clarity"]
    label_encoder = LabelEncoder()
    encoding_mappings = {}
    for col in columns_to_encode:
        diamonds_df_copy[col] = label_encoder.fit_transform(diamonds_df_copy[col])
        encoding_mappings[col] = list(label_encoder.classes_)

    # Rozdělení na X a y
    X = diamonds_df_copy.drop("price", axis=1)
    y = diamonds_df_copy["price"]

    # Rozdělení 70/30 (random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print(f"📊 Trénovací sada (70 %): {X_train.shape[0]} vzorků")
    print(f"📊 Testovací sada (30 %): {X_test.shape[0]} vzorků")

    # -------------------------------------------------------------------------
    # 2. Původní model z 1. dne kurzu (max_depth=13, default squared_error)
    # -------------------------------------------------------------------------
    print("\n--- Původní model z 1. dne (max_depth=13) ---")
    base_tree = DecisionTreeRegressor(max_depth=13, random_state=42)
    base_tree.fit(X_train, y_train)

    base_pred_train = base_tree.predict(X_train)
    base_pred_test = base_tree.predict(X_test)

    baseline_metrics = {
        "max_depth": 13,
        "criterion": "squared_error",
        "train_r2": round(float(r2_score(y_train, base_pred_train)), 4),
        "test_r2": round(float(r2_score(y_test, base_pred_test)), 4),
        "test_mae": round(float(mean_absolute_error(y_test, base_pred_test)), 2),
        "test_rmse": round(float(np.sqrt(mean_squared_error(y_test, base_pred_test))), 2),
        "n_leaves": int(base_tree.get_n_leaves()),
        "depth": int(base_tree.get_depth())
    }
    print(f"Baseline Train R2: {baseline_metrics['train_r2']:.4f}")
    print(f"Baseline Test R2:  {baseline_metrics['test_r2']:.4f}")
    print(f"Baseline Test MAE: {baseline_metrics['test_mae']:.2f} $")
    print(f"Baseline Test RMSE:{baseline_metrics['test_rmse']:.2f} $")

    # -------------------------------------------------------------------------
    # 3. GridSearchCV: Nalezení nejlepší dvojice (max_depth, criterion)
    # -------------------------------------------------------------------------
    print("\n--- Spouštím GridSearchCV pro max_depth a criterion ---")
    param_grid = {
        "max_depth": [5, 8, 10, 11, 12, 13, 14, 15, 16, 18],
        "criterion": ["squared_error", "poisson"]
    }
    total_fits = len(param_grid["max_depth"]) * len(param_grid["criterion"]) * 5
    print(f"Testuji {len(param_grid['max_depth']) * len(param_grid['criterion'])} kombinací x 5 foldů = {total_fits} modelů...")

    t0 = time.time()
    # Optimalizujeme s ohledem na robustnost chyb (R2 / MSE)
    grid_search = GridSearchCV(
        estimator=DecisionTreeRegressor(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)
    grid_time = time.time() - t0
    print(f"GridSearchCV dokončen za {grid_time:.2f} s.")

    best_params = grid_search.best_params_
    best_cv_rmse = np.sqrt(-grid_search.best_score_)
    print(f"Nejlepší nalezené parametry: {best_params}")
    print(f"Nejlepší 5-Fold CV RMSE: {best_cv_rmse:.2f} $")

    # -------------------------------------------------------------------------
    # 4. Upravená funkce dle zadání: train_and_test_decision_tree(hyperparameters: dict)
    # -------------------------------------------------------------------------
    def train_and_test_decision_tree(hyperparameters: dict):
        """
        Natrénuje rozhodovací strom se zadaným slovníkem hyperparametrů,
        vyhodnotí metriky na trénovací i testovací sadě a vrátí slovník výsledků.
        """
        reg_tree = DecisionTreeRegressor(**hyperparameters, random_state=42)
        reg_tree.fit(X_train, y_train)

        y_pred = reg_tree.predict(X_test)
        y_pred_train = reg_tree.predict(X_train)

        train_r2 = float(r2_score(y_train, y_pred_train))
        test_r2 = float(r2_score(y_test, y_pred))
        test_mae = float(mean_absolute_error(y_test, y_pred))
        test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        return {
            "model": reg_tree,
            "hyperparameters": hyperparameters,
            "train_r2": round(train_r2, 4),
            "test_r2": round(test_r2, 4),
            "test_mae": round(test_mae, 2),
            "test_rmse": round(test_rmse, 2),
            "n_leaves": int(reg_tree.get_n_leaves()),
            "depth": int(reg_tree.get_depth()),
            "predictions_test": y_pred
        }

    # -------------------------------------------------------------------------
    # 5. Trénování a vyhodnocení optimálního modelu
    # -------------------------------------------------------------------------
    print("\n--- Volání upravené funkce train_and_test_decision_tree s best_params ---")
    optimal_eval = train_and_test_decision_tree(best_params)
    print(f"Optimal Train R2: {optimal_eval['train_r2']:.4f}")
    print(f"Optimal Test R2:  {optimal_eval['test_r2']:.4f}")
    print(f"Optimal Test MAE: {optimal_eval['test_mae']:.2f} $")
    print(f"Optimal Test RMSE:{optimal_eval['test_rmse']:.2f} $")

    # Porovnání také pro criterion='poisson', max_depth=12 (vynikající MAE)
    poisson_eval = train_and_test_decision_tree({"criterion": "poisson", "max_depth": 12})

    # Analýza zlepšení
    delta_r2 = optimal_eval["test_r2"] - baseline_metrics["test_r2"]
    delta_rmse = baseline_metrics["test_rmse"] - optimal_eval["test_rmse"]
    delta_mae = baseline_metrics["test_mae"] - optimal_eval["test_mae"]
    train_test_gap_base = baseline_metrics["train_r2"] - baseline_metrics["test_r2"]
    train_test_gap_opt = optimal_eval["train_r2"] - optimal_eval["test_r2"]

    print("\n--- POROVNÁNÍ ZLEPŠENÍ OPROTI 1. DNI ---")
    print(f"Změna Test R2:   {delta_r2:+.4f}")
    print(f"Pokles RMSE chyb:{delta_rmse:+.2f} $ (lepší přesnost)")
    print(f"Pokles overfittingu (Train-Test R2 Gap): z {train_test_gap_base:.4f} na {train_test_gap_opt:.4f} (pokles o {(1 - train_test_gap_opt/train_test_gap_base)*100:.1f} %!)")

    # -------------------------------------------------------------------------
    # 6. Sestavení výsledků pro JSON cache
    # -------------------------------------------------------------------------
    cv_df = pd.DataFrame(grid_search.cv_results_)
    depth_sweep_records = []
    for d in param_grid["max_depth"]:
        sq_row = cv_df[(cv_df["param_max_depth"] == d) & (cv_df["param_criterion"] == "squared_error")].iloc[0]
        poiss_row = cv_df[(cv_df["param_max_depth"] == d) & (cv_df["param_criterion"] == "poisson")].iloc[0]
        depth_sweep_records.append({
            "max_depth": d,
            "squared_error_rmse": round(float(np.sqrt(-sq_row["mean_test_score"])), 2),
            "squared_error_train_rmse": round(float(np.sqrt(-sq_row["mean_train_score"])), 2),
            "poisson_rmse": round(float(np.sqrt(-poiss_row["mean_test_score"])), 2),
            "poisson_train_rmse": round(float(np.sqrt(-poiss_row["mean_train_score"])), 2),
        })

    json_payload = {
        "dataset_summary": {
            "n_samples": len(diamonds_df),
            "n_features": X.shape[1],
            "feature_names": list(X.columns),
            "train_size": len(X_train),
            "test_size": len(X_test),
        },
        "baseline_model": baseline_metrics,
        "grid_search_summary": {
            "time_seconds": round(grid_time, 2),
            "total_fits": total_fits,
            "best_params": best_params,
            "best_cv_rmse": round(float(best_cv_rmse), 2),
        },
        "optimal_model": {
            "hyperparameters": optimal_eval["hyperparameters"],
            "train_r2": optimal_eval["train_r2"],
            "test_r2": optimal_eval["test_r2"],
            "test_mae": optimal_eval["test_mae"],
            "test_rmse": optimal_eval["test_rmse"],
            "n_leaves": optimal_eval["n_leaves"],
            "depth": optimal_eval["depth"],
        },
        "alternative_poisson_model": {
            "hyperparameters": poisson_eval["hyperparameters"],
            "train_r2": poisson_eval["train_r2"],
            "test_r2": poisson_eval["test_r2"],
            "test_mae": poisson_eval["test_mae"],
            "test_rmse": poisson_eval["test_rmse"],
            "n_leaves": poisson_eval["n_leaves"],
            "depth": poisson_eval["depth"],
        },
        "comparison_metrics": {
            "delta_r2": round(float(delta_r2), 4),
            "delta_rmse": round(float(delta_rmse), 2),
            "overfitting_gap_reduction_pct": round(float((1 - train_test_gap_opt/train_test_gap_base)*100), 2)
        },
        "depth_sweep": depth_sweep_records
    }

    out_json = data_dir / "diamonds_hyperparameters_exercise_1_precomputed.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Uložena cache: {out_json}")

    # -------------------------------------------------------------------------
    # 7. Diagnostické grafy
    # -------------------------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid")

    # Graf 1: Validační a trénovací RMSE v závislosti na max_depth pro obě kritéria
    plt.figure(figsize=(10, 5))
    depths = [r["max_depth"] for r in depth_sweep_records]
    sq_test = [r["squared_error_rmse"] for r in depth_sweep_records]
    sq_train = [r["squared_error_train_rmse"] for r in depth_sweep_records]
    poi_test = [r["poisson_rmse"] for r in depth_sweep_records]

    plt.plot(depths, sq_test, marker="o", linewidth=2.5, color="#1f77b4", label="Test CV RMSE (squared_error)")
    plt.plot(depths, sq_train, linestyle="--", color="#1f77b4", alpha=0.6, label="Train CV RMSE (squared_error)")
    plt.plot(depths, poi_test, marker="s", linewidth=2, color="#2ca02c", label="Test CV RMSE (poisson)")

    plt.axvline(best_params["max_depth"], color="red", linestyle=":", label=f"Optimum (max_depth={best_params['max_depth']})")
    plt.axvline(13, color="gray", linestyle="-.", label="Původní model 1. dne (depth=13)")

    plt.title("GridSearchCV: Křivka chyb RMSE vs. Hloubka stromu (max_depth)", fontsize=13, fontweight="bold")
    plt.xlabel("Maximální hloubka stromu (max_depth)", fontsize=11)
    plt.ylabel("5-Fold CV RMSE ($)", fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    p1 = plots_dir / "diamonds_grid_search_depth_curve.png"
    plt.savefig(p1, dpi=200)
    plt.close()
    print(f"📈 Uložen graf: {p1}")

    # Graf 2: Rezidua původního modelu vs. optimálního modelu
    plt.figure(figsize=(10, 5))
    res_base = y_test - base_pred_test
    res_opt = y_test - optimal_eval["predictions_test"]

    sns.kdeplot(res_base, label=f"Původní strom (depth=13, RMSE={baseline_metrics['test_rmse']:.0f} $)", color="gray", linestyle="--")
    sns.kdeplot(res_opt, label=f"Optimální strom (depth={best_params['max_depth']}, RMSE={optimal_eval['test_rmse']:.0f} $)", color="#d95f02", linewidth=2)

    plt.axvline(0, color="black", linestyle=":")
    plt.xlim(-3000, 3000)
    plt.title("Distribuce reziduí (chyb): Původní model vs. GridSearchCV optimum", fontsize=13, fontweight="bold")
    plt.xlabel("Chyba predikce (Skutečná cena - Predikovaná cena v $)", fontsize=11)
    plt.ylabel("Hustota pravděpodobnosti", fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    p2 = plots_dir / "diamonds_grid_residuals_comparison.png"
    plt.savefig(p2, dpi=200)
    plt.close()
    print(f"📈 Uložen graf: {p2}")

    print("\n✅ Cvičení 1 (Diamanty - GridSearchCV) bylo úspěšně spočteno a uloženo.")


if __name__ == "__main__":
    run_diamonds_hyperparameters_exercise_1()
