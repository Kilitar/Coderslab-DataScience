"""
Homework: Rozhodovací strom (RandomizedSearchCV) – Pevnost betonu (Concrete)
============================================================================
Tento skript řeší zadání 'Decision tree (regression) - exercise 2':
1. Import knihoven pro DecisionTreeRegressor, RandomizedSearchCV, plot_tree a metriky.
2. Načtení škálovaného datasetu concrete_data_preprocessed.csv.
3. Rozdělení na trénovací a testovací sadu v poměru 70/30 (random_state=42).
4. Vytvoření slovníku hyperparametrů: max_depth, criterion, min_samples_leaf, max_features.
5. Hledání optimálních parametrů pomocí RandomizedSearchCV s 5-násobnou křížovou validací
   a evaluační metrikou MAE (scoring='neg_mean_absolute_error').
6. Uložení nejlepší sady parametrů do proměnné best_hyperparams.
7. Natrénování finálního modelu DecisionTreeRegressor s těmito parametry.
8. Přidání vizualizace struktury natrénovaného stromu (plot_tree).
9. Otestování efektivity modelu pomocí koeficientu determinace (R2) na trénovací i testovací sadě.
10. Výpočet střední kvadratické chyby (MSE) a průměrné absolutní chyby (MAE) na testovací sadě.
11. Srovnání výsledků s GridSearchCV z Cvičení 1 a s lineární regresí.
12. Uložení výsledků do JSON cache a export diagnostických grafů.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)

# Cesty
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "concrete_data_preprocessed.csv"
PRECOMPUTED_JSON_PATH = DATA_DIR / "concrete_decision_tree_random_precomputed.json"

FEATURE_NAMES = [
    "cement", "slag", "flyash", "water",
    "superplasticizer", "coarseaggregate", "fineaggregate", "age"
]
TARGET_NAME = "csMPa"

FEATURE_DESCRIPTIONS = {
    "cement": "Obsah cementu (kg/m³)",
    "slag": "Obsah vysokopecní strusky (kg/m³)",
    "flyash": "Obsah popílku (kg/m³)",
    "water": "Obsah záměsové vody (kg/m³)",
    "superplasticizer": "Obsah superplastifikátoru (kg/m³)",
    "coarseaggregate": "Obsah hrubého kameniva / štěrku (kg/m³)",
    "fineaggregate": "Obsah jemného kameniva / písku (kg/m³)",
    "age": "Doba zrání betonu (dny)"
}


def run_decision_tree_random_pipeline():
    print("=" * 80)
    print("DÚ: ROZHODOVACÍ STROM PRO REGRESI S RANDOMIZEDSEARCHCV (BETON)")
    print("=" * 80)

    # 1. Načtení dat
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset nenalezen: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    print(f"Data úspěšně načtena: {df.shape[0]} řádků, {df.shape[1]} sloupců")

    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]

    # 2. Rozdělení na trénovací a testovací sadu (70/30)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print(f"Rozdělení: Trénovací {X_train.shape}, Testovací {X_test.shape}")

    # 3. Vytvoření slovníku hyperparametrů pro RandomizedSearch
    # max_depth, criterion, min_samples_leaf, max_features
    param_distributions = {
        "max_depth": [3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, None],
        "criterion": ["squared_error", "absolute_error", "poisson"],
        "min_samples_leaf": [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20],
        "max_features": [None, "sqrt", "log2", 0.5, 0.6, 0.7, 0.8, 1.0, 4, 5, 6, 7, 8]
    }

    # Celkový počet možných kombinací
    total_combinations = (
        len(param_distributions["max_depth"]) *
        len(param_distributions["criterion"]) *
        len(param_distributions["min_samples_leaf"]) *
        len(param_distributions["max_features"])
    )
    print(f"Celkový prostor možných konfigurací: {total_combinations} kombinací")

    # 4. RandomizedSearchCV s 5-násobnou CV a MAE metrikou
    n_iter_search = 60
    random_search = RandomizedSearchCV(
        estimator=DecisionTreeRegressor(random_state=42),
        param_distributions=param_distributions,
        n_iter=n_iter_search,
        scoring="neg_mean_absolute_error",
        cv=5,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    print(f"Spouštím RandomizedSearchCV ({n_iter_search} náhodných pokusů x 5-fold CV = {n_iter_search * 5} fitů)...")
    random_search.fit(X_train, y_train)

    # 5. Uložení nejlepší sady hyperparametrů do best_hyperparams
    best_hyperparams = random_search.best_params_
    best_cv_mae = -random_search.best_score_
    print("\n" + "-" * 50)
    print(f"Nejlepší nalezená sada (best_hyperparams): {best_hyperparams}")
    print(f"Nejlepší 5-Fold CV MAE: {best_cv_mae:.4f} MPa")
    print("-" * 50)

    # 6. Natrénování finálního modelu s nalezenými hyperparametry
    tree_reg = DecisionTreeRegressor(**best_hyperparams, random_state=42)
    tree_reg.fit(X_train, y_train)

    # 7. Predikce a metriky
    y_pred_train = tree_reg.predict(X_train)
    y_pred_test = tree_reg.predict(X_test)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)

    train_rmse = root_mean_squared_error(y_train, y_pred_train)
    test_rmse = root_mean_squared_error(y_test, y_pred_test)

    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    train_mape = mean_absolute_percentage_error(y_train, y_pred_train)
    test_mape = mean_absolute_percentage_error(y_test, y_pred_test)

    print("\nEVALUACE FINÁLNÍHO MODELU:")
    print(f"  Trénovací R²: {train_r2 * 100:.2f} %")
    print(f"  Testovací R²: {test_r2 * 100:.2f} %")
    print(f"  Testovací MSE: {test_mse:.4f}")
    print(f"  Testovací RMSE: {test_rmse:.4f} MPa")
    print(f"  Testovací MAE:  {test_mae:.4f} MPa")
    print(f"  Testovací MAPE: {test_mape * 100:.2f} %")

    # 8. Analýza důležitosti příznaků (Feature Importances)
    importances = tree_reg.feature_importances_
    fi_df = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "importance": importances,
        "description": [FEATURE_DESCRIPTIONS[f] for f in FEATURE_NAMES]
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    print("\nDŮLEŽITOST PŘÍZNAKŮ (FEATURE IMPORTANCES):")
    for _, row in fi_df.iterrows():
        print(f"  {row['feature']:18} {row['importance'] * 100:6.2f} %  ({row['description']})")

    # 9. Vizualizace struktury natrénovaného stromu (plot_tree)
    print("\nGeneruji vizualizaci struktury stromu (plot_tree)...")
    plt.figure(figsize=(24, 12), dpi=250)
    plot_tree(
        tree_reg,
        max_depth=3,
        feature_names=FEATURE_NAMES,
        filled=True,
        rounded=True,
        fontsize=9,
        precision=2
    )
    plt.title(
        f"Rozhodovací strom s RandomizedSearchCV – Pevnost betonu (vrchní 3 patra)\n"
        f"Parametry: max_depth={best_hyperparams['max_depth']}, criterion='{best_hyperparams['criterion']}', "
        f"min_samples_leaf={best_hyperparams['min_samples_leaf']}, max_features={best_hyperparams['max_features']}",
        fontsize=14,
        fontweight="bold",
        pad=15
    )
    plt.tight_layout()
    tree_plot_path = PLOTS_DIR / "concrete_tree_random_structure.png"
    plt.savefig(tree_plot_path, bbox_inches="tight")
    plt.close()
    print(f"Uloženo: {tree_plot_path}")

    # 10. Vizualizace důležitosti příznaků
    plt.figure(figsize=(10, 5), dpi=200)
    palette = sns.color_palette("viridis", len(fi_df))
    bars = plt.barh(fi_df["feature"][::-1], fi_df["importance"][::-1] * 100, color=palette)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.6, bar.get_y() + bar.get_height() / 2, f"{w:.1f} %", va="center", fontsize=9, fontweight="bold")
    plt.title("Relativní důležitost příznaků (Feature Importances) – RandomizedSearch Tree", fontsize=12, fontweight="bold")
    plt.xlabel("Důležitost (%)")
    plt.xlim(0, max(fi_df["importance"] * 100) + 8)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fi_plot_path = PLOTS_DIR / "concrete_tree_random_feature_importance.png"
    plt.savefig(fi_plot_path, bbox_inches="tight")
    plt.close()
    print(f"Uloženo: {fi_plot_path}")

    # 11. Vizualizace průběhu náhodného vyhledávání (RandomizedSearch CV Scores)
    cv_results_df = pd.DataFrame(random_search.cv_results_)
    cv_results_df["mean_mae"] = -cv_results_df["mean_test_score"]

    plt.figure(figsize=(11, 5), dpi=200)
    plt.scatter(
        range(1, len(cv_results_df) + 1),
        cv_results_df["mean_mae"],
        c=cv_results_df["rank_test_score"],
        cmap="plasma_r",
        s=60,
        alpha=0.85,
        edgecolors="k",
        linewidths=0.5
    )
    plt.axhline(best_cv_mae, color="red", linestyle="--", linewidth=1.5, label=f"Nejlepší CV MAE = {best_cv_mae:.2f} MPa")
    plt.title(f"Průběh {n_iter_search} náhodných pokusů v RandomizedSearchCV (K-Fold CV MAE)", fontsize=12, fontweight="bold")
    plt.xlabel("Index iterace vzorkování")
    plt.ylabel("Validační chyba MAE (MPa)")
    plt.colorbar(label="Pořadí (Rank)")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    search_plot_path = PLOTS_DIR / "concrete_tree_random_search_distribution.png"
    plt.savefig(search_plot_path, bbox_inches="tight")
    plt.close()
    print(f"Uloženo: {search_plot_path}")

    # 12. Načtení srovnávacích výsledků z předchozích úloh
    grid_tree_json_path = DATA_DIR / "concrete_decision_tree_precomputed.json"
    ols_json_path = DATA_DIR / "concrete_linear_regression_precomputed.json"

    comp_models = {}
    if grid_tree_json_path.exists():
        with open(grid_tree_json_path, "r", encoding="utf-8") as f:
            comp_models["grid_tree"] = json.load(f)["metrics"]["test"]
    if ols_json_path.exists():
        with open(ols_json_path, "r", encoding="utf-8") as f:
            comp_models["ols"] = json.load(f)["metrics"]["test"]

    # 13. Export kompletní JSON cache
    cache_data = {
        "dataset_info": {
            "n_samples": len(df),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "feature_names": FEATURE_NAMES,
            "target_name": TARGET_NAME
        },
        "search_space": {
            "max_depth_options": param_distributions["max_depth"],
            "criterion_options": param_distributions["criterion"],
            "min_samples_leaf_options": param_distributions["min_samples_leaf"],
            "max_features_options": [str(x) for x in param_distributions["max_features"]],
            "total_combinations": total_combinations,
            "n_iter": n_iter_search
        },
        "best_hyperparams": {
            k: (int(v) if isinstance(v, (np.integer, int)) else (float(v) if isinstance(v, (np.floating, float)) else (None if v is None else str(v))))
            for k, v in best_hyperparams.items()
        },
        "cv_score_mae": float(best_cv_mae),
        "metrics": {
            "train": {
                "r2": float(train_r2),
                "mse": float(train_mse),
                "rmse": float(train_rmse),
                "mae": float(train_mae),
                "mape": float(train_mape)
            },
            "test": {
                "r2": float(test_r2),
                "mse": float(test_mse),
                "rmse": float(test_rmse),
                "mae": float(test_mae),
                "mape": float(test_mape)
            }
        },
        "feature_importances": fi_df.to_dict(orient="records"),
        "top_trials": cv_results_df.sort_values(by="mean_mae").head(10)[[
            "rank_test_score", "mean_mae", "param_max_depth", "param_criterion", "param_min_samples_leaf", "param_max_features"
        ]].to_dict(orient="records"),
        "comparison_models": comp_models,
        "sample_predictions": [
            {
                "idx": int(idx),
                "actual_mpa": round(float(y_test.iloc[i]), 2),
                "pred_random_tree": round(float(y_pred_test[i]), 2),
                "abs_error": round(float(abs(y_test.iloc[i] - y_pred_test[i])), 2)
            }
            for i, idx in enumerate(y_test.index[:10])
        ]
    }

    with open(PRECOMPUTED_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)
    print(f"Předpočtené výsledky uloženy do: {PRECOMPUTED_JSON_PATH}")
    print("Všechny kroky Cvičení 2 úspěšně dokončeny!")


if __name__ == "__main__":
    run_decision_tree_random_pipeline()
