"""
Homework: Rozhodovací strom (Decision Tree Regression) – Pevnost betonu (Concrete)
==================================================================================
Tento skript řeší zadání 'Decision tree (regression) - exercise 1':
1. Načtení škálovaného datasetu concrete_data_preprocessed.csv.
2. Rozdělení na trénovací a testovací sadu v poměru 70/30 (random_state=42).
3. Definice mřížky hyperparametrů (max_depth, criterion, min_samples_leaf).
4. Hledání optimálních hyperparametrů pomocí GridSearchCV s 5-násobnou křížovou validací
   a evaluační metrikou MAE (scoring='neg_mean_absolute_error').
5. Uložení nalezených hyperparametrů do proměnné best_hyperparams.
6. Natrénování finálního modelu DecisionTreeRegressor(**best_hyperparams).
7. Vizualizace struktury natrénovaného rozhodovacího stromu (plot_tree).
8. Vyhodnocení efektivity modelu pomocí koeficientu determinace (R2) na Train i Test sadě,
   výpočet MSE, RMSE a MAE na testovací sadě.
9. Výpočet důležitosti příznaků (Feature Importances) a srovnání s lineární regresí.
10. Uložení výsledků do JSON cache pro Streamlit a export grafů.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
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
PRECOMPUTED_JSON_PATH = DATA_DIR / "concrete_decision_tree_precomputed.json"

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


def main():
    print("=" * 70)
    print("1. Načtení dat a rozdělení 70/30")
    print("=" * 70)
    
    df = pd.read_csv(CSV_PATH)
    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print(f"Trénovací sada (70 %): {X_train.shape[0]} vzorků")
    print(f"Testovací sada (30 %): {X_test.shape[0]} vzorků")
    
    print("\n" + "=" * 70)
    print("2. Mřížka hyperparametrů a GridSearchCV (MAE metric)")
    print("=" * 70)
    
    param_grid = {
        "max_depth": [3, 5, 7, 9, 11, 13, 15, None],
        "criterion": ["squared_error", "absolute_error"],
        "min_samples_leaf": [1, 2, 4, 6, 8, 12]
    }
    
    print("Parametry mřížky:", param_grid)
    
    grid_search = GridSearchCV(
        estimator=DecisionTreeRegressor(random_state=42),
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=5,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    # 5. Uložení do proměnné best_hyperparams dle zadání
    best_hyperparams = grid_search.best_params_
    best_cv_mae = float(round(-grid_search.best_score_, 4))
    
    print("\n" + "=" * 70)
    print("3. Nalezené optimální hyperparametry (best_hyperparams)")
    print("=" * 70)
    print("best_hyperparams =", best_hyperparams)
    print(f"Nejlepší 5-fold CV MAE skóre: {best_cv_mae:.4f} MPa")
    
    # 6. Natrénování modelu s parametry z grid search
    best_tree = DecisionTreeRegressor(**best_hyperparams, random_state=42)
    best_tree.fit(X_train, y_train)
    
    # 7. Predikce a testování efektivity
    y_pred_train = best_tree.predict(X_train)
    y_pred_test = best_tree.predict(X_test)
    
    train_r2 = float(round(r2_score(y_train, y_pred_train), 4))
    test_r2 = float(round(r2_score(y_test, y_pred_test), 4))
    test_mse = float(round(mean_squared_error(y_test, y_pred_test), 4))
    test_rmse = float(round(root_mean_squared_error(y_test, y_pred_test), 4))
    test_mae = float(round(mean_absolute_error(y_test, y_pred_test), 4))
    test_mape = float(round(mean_absolute_percentage_error(y_test, y_pred_test) * 100, 2))
    
    print("\n" + "=" * 70)
    print("4. Efektivita modelu (Koeficient determinace, MSE, RMSE, MAE)")
    print("=" * 70)
    print(f"Koeficient determinace na Train sadě (R²): {train_r2:.4f} ({train_r2*100:.2f} %)")
    print(f"Koeficient determinace na Test sadě (R²):  {test_r2:.4f} ({test_r2*100:.2f} %)")
    print(f"Mean Squared Error na Test sadě (MSE):     {test_mse:.4f}")
    print(f"Root Mean Squared Error na Test sadě (RMSE): {test_rmse:.4f} MPa")
    print(f"Mean Absolute Error na Test sadě (MAE):    {test_mae:.4f} MPa")
    print(f"Mean Absolute Percentage Error (MAPE):     {test_mape:.2f} %")
    
    # Důležitost příznaků
    feature_importances = {
        feat: float(round(imp, 4))
        for feat, imp in sorted(
            zip(FEATURE_NAMES, best_tree.feature_importances_),
            key=lambda x: x[1],
            reverse=True
        )
    }
    print("\nDůležitost příznaků (Feature Importances):")
    for feat, imp in feature_importances.items():
        print(f"  {feat:18s}: {imp*100:5.2f} % ({FEATURE_DESCRIPTIONS[feat]})")
        
    print("\n" + "=" * 70)
    print("5. Generování diagnostických PNG grafů a vizualizace stromu")
    print("=" * 70)
    
    # 1. Vizualizace struktury natrénovaného stromu (top 3 úrovně pro čitelnost)
    fig, ax = plt.subplots(figsize=(18, 9))
    plot_tree(
        best_tree,
        max_depth=3,
        feature_names=FEATURE_NAMES,
        filled=True,
        rounded=True,
        fontsize=9,
        precision=2,
        ax=ax
    )
    ax.set_title(
        f"Struktura rozhodovacího stromu (DecisionTreeRegressor) – Vrchní 3 patra\n"
        f"Kritérium: {best_hyperparams['criterion']}, max_depth: {best_hyperparams['max_depth']}, min_samples_leaf: {best_hyperparams['min_samples_leaf']}",
        fontsize=12,
        fontweight="bold"
    )
    plt.tight_layout()
    tree_plot_path = PLOTS_DIR / "concrete_tree_structure.png"
    fig.savefig(tree_plot_path, dpi=200)
    plt.close(fig)
    print(f"Uložen graf stromu: {tree_plot_path}")
    
    # 2. Sloupcový graf důležitosti příznaků
    fig, ax = plt.subplots(figsize=(9, 5))
    feats = list(reversed(list(feature_importances.keys())))
    imps = list(reversed(list(feature_importances.values())))
    bars = ax.barh(feats, [v * 100 for v in imps], color="#2ca02c", edgecolor="black", alpha=0.85)
    ax.set_title("Důležitost příznaků v rozhodovacím stromu (Gini / Variance Reduction)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Relativní důležitost (%)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.1f} %", xy=(w + 0.8, bar.get_y() + bar.get_height() / 2),
                    va="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    fi_plot_path = PLOTS_DIR / "concrete_tree_feature_importance.png"
    fig.savefig(fi_plot_path, dpi=200)
    plt.close(fig)
    print(f"Uložen graf důležitosti: {fi_plot_path}")
    
    # 3. Skutečnost vs. Predikce (Porovnání Decision Tree vs. Lineární regrese z cvičení 1)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(y_test, y_pred_test, color="#2ca02c", alpha=0.6, edgecolors="k", s=35, label=f"Decision Tree (R² = {test_r2*100:.1f} %)")
    
    # Přidáme pro kontext i přímku OLS
    from sklearn.linear_model import LinearRegression
    ols = LinearRegression()
    ols.fit(X_train, y_train)
    y_pred_ols = ols.predict(X_test)
    ols_r2 = r2_score(y_test, y_pred_ols)
    ax.scatter(y_test, y_pred_ols, color="#1f77b4", alpha=0.25, s=20, label=f"Lineární regrese OLS (R² = {ols_r2*100:.1f} %)")
    
    min_val = min(y_test.min(), y_pred_test.min()) - 2
    max_val = max(y_test.max(), y_pred_test.max()) + 2
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Ideální shoda (y = ŷ)")
    ax.set_title(f"Pevnost betonu: Skutečnost vs. Predikce\nDecision Tree R² = {test_r2:.4f}, RMSE = {test_rmse:.2f} MPa", fontsize=12, fontweight="bold")
    ax.set_xlabel("Skutečná pevnost csMPa (MPa)", fontsize=11)
    ax.set_ylabel("Predikovaná pevnost csMPa (MPa)", fontsize=11)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    pred_plot_path = PLOTS_DIR / "concrete_tree_actual_vs_predicted.png"
    fig.savefig(pred_plot_path, dpi=200)
    plt.close(fig)
    print(f"Uložen graf predikcí: {pred_plot_path}")
    
    # Srovnávací data vzorků pro tabulku
    sample_preds = pd.DataFrame({
        "cement": X_test["cement"].values[:10],
        "water": X_test["water"].values[:10],
        "age": X_test["age"].values[:10],
        "Skutečnost csMPa": np.round(y_test.values[:10], 2),
        "Predikce Strom csMPa": np.round(y_pred_test[:10], 2),
        "Predikce OLS csMPa": np.round(y_pred_ols[:10], 2),
        "Chyba Stromu (Abs)": np.round(np.abs(y_test.values[:10] - y_pred_test[:10]), 2),
        "Chyba OLS (Abs)": np.round(np.abs(y_test.values[:10] - y_pred_ols[:10]), 2)
    })
    
    precomputed = {
        "metadata": {
            "task": "Homework: Decision tree (regression) - exercise 1 (Concrete)",
            "model": "DecisionTreeRegressor",
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "features": FEATURE_NAMES,
            "feature_descriptions": FEATURE_DESCRIPTIONS,
            "target": TARGET_NAME
        },
        "best_hyperparams": best_hyperparams,
        "best_cv_mae": best_cv_mae,
        "metrics": {
            "train": {
                "r2": train_r2
            },
            "test": {
                "r2": test_r2,
                "mse": test_mse,
                "rmse": test_rmse,
                "mae": test_mae,
                "mape": test_mape
            },
            "ols_comparison": {
                "r2": float(round(ols_r2, 4)),
                "rmse": float(round(root_mean_squared_error(y_test, y_pred_ols), 4)),
                "mae": float(round(mean_absolute_error(y_test, y_pred_ols), 4)),
                "r2_improvement": float(round((test_r2 - ols_r2) * 100, 2)),
                "rmse_reduction": float(round(root_mean_squared_error(y_test, y_pred_ols) - test_rmse, 4)),
                "mae_reduction": float(round(mean_absolute_error(y_test, y_pred_ols) - test_mae, 4))
            }
        },
        "feature_importances": feature_importances,
        "sample_predictions": sample_preds.to_dict(orient="records"),
        "tree_stats": {
            "depth": int(best_tree.get_depth()),
            "n_leaves": int(best_tree.get_n_leaves())
        }
    }
    
    with open(PRECOMPUTED_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(precomputed, f, indent=2, ensure_ascii=False)
    print(f"\nUložena JSON cache: {PRECOMPUTED_JSON_PATH}")
    print("HOTOVO!")


if __name__ == "__main__":
    main()
