"""
Decision Tree in Regression - Exercise 1
Dataset: King County Real Estate (kc_house_data.csv)
Module: Decision Tree Regressor, Hyperparameter Tuning & Overfitting Control

ZADÁNÍ / ASSIGNMENT:
------------------------------------------------
1. Import all necessary methods from libraries (or modules from these libraries).
2. Load the dataset attached to the exercise (the dataset before data processing, kc_house_data.csv),
   using the appropriate method from the Pandas library.
3. Divide the dataset into training and test datasets.
4. Initialize the decision tree model. Select a starting set of hyperparameters.
5. Train the decision tree model.
6. Generate a graph that visualizes how the trained model calculates predictions.
7. Calculate the metrics R2, MAE, MSE and RMSE.
8. Repeat steps 2, 3, 4 and 5 until you have an optimal model.
------------------------------------------------
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def calculate_metrics(y_true, y_pred, dataset_name: str = "Test") -> dict:
    """Vypočítá 4 základní regresní metriky požadované zadáním."""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {
        "Dataset": dataset_name,
        "R2": r2,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
    }


def print_metrics_table(metrics_list: list, title: str) -> None:
    """Vytiskne přehlednou tabulku metrik do konzole."""
    print(f"\n{title}")
    print("-" * 75)
    print(f"{'Model / Dataset':<32} | {'R²':>8} | {'MAE (USD)':>12} | {'RMSE (USD)':>12}")
    print("-" * 75)
    for m in metrics_list:
        print(f"{m['Name']:<32} | {m['R2']:>8.4f} | {m['MAE']:>12,.1f} | {m['RMSE']:>12,.1f}")
    print("-" * 75)


def main() -> None:
    print("=" * 80)
    print(" CVIČENÍ: ROZHODOVACÍ STROM V REGRESI (KING COUNTY REALITY)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Nastavení cest
    # -------------------------------------------------------------------------
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    raw_data_path = data_dir / "kc_house_data.csv"
    if not raw_data_path.exists():
        raise FileNotFoundError(f"Vstupní soubor nebyl nalezen: {raw_data_path}")

    # =========================================================================
    # KROK 1 & 2: Načtení dat před zpracováním (Raw Dataset)
    # =========================================================================
    print("\n[Krok 1 & 2] Načítání datasetu před zpracováním (kc_house_data.csv)...")
    housing_df = pd.read_csv(raw_data_path)
    print(f"Původní rozměry dat: {housing_df.shape[0]:,} řádků, {housing_df.shape[1]} sloupců")

    # Kontrola chybějících hodnot
    missing_count = housing_df.isnull().sum().sum()
    print(f"Počet chybějících hodnot (NaN): {missing_count}")

    # Odstranění čistě technických / identifikátorových sloupců
    cols_to_drop = [col for col in ["id"] if col in housing_df.columns]
    df_clean = housing_df.drop(columns=cols_to_drop)

    # Základní extrakce roku a měsíce z textového data prodeje
    if "date" in df_clean.columns:
        df_clean["yr_sold"] = df_clean["date"].str[:4].astype(int)
        df_clean["month_sold"] = df_clean["date"].str[4:6].astype(int)
        df_clean = df_clean.drop(columns=["date"])

    print(f"Rozměry dat po odstranění id a úpravě data: {df_clean.shape}")

    # =========================================================================
    # KROK 3: Rozdělení dat na trénovací a testovací sadu (80/20)
    # =========================================================================
    print("\n[Krok 3] Rozdělení dat na trénovací (80 %) a testovací (20 %) sadu...")
    X = df_clean.drop(columns=["price"])
    y = df_clean["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Trénovací sada: {X_train.shape[0]:,} řádků")
    print(f"Testovací sada: {X_test.shape[0]:,} řádků")

    all_models_summary = []

    # =========================================================================
    # KROK 4, 5, 6, 7: MODEL 1 – Výchozí neomezený strom (Baseline)
    # =========================================================================
    print("\n" + "=" * 80)
    print("[Model 1] Výchozí rozhodovací strom (Bez omezení hloubky - Baseline)")
    print("=" * 80)
    # Startovací sada parametrů: výchozí parametry Scikit-learnu
    tree_m1 = DecisionTreeRegressor(random_state=42)
    tree_m1.fit(X_train, y_train)

    train_preds_m1 = tree_m1.predict(X_train)
    test_preds_m1 = tree_m1.predict(X_test)

    m1_train_metrics = calculate_metrics(y_train, train_preds_m1, "Train")
    m1_test_metrics = calculate_metrics(y_test, test_preds_m1, "Test")

    print(f"Trénovací R²: {m1_train_metrics['R2']:.4f} | RMSE: {m1_train_metrics['RMSE']:,.1f} USD")
    print(f"Testovací  R²: {m1_test_metrics['R2']:.4f} | RMSE: {m1_test_metrics['RMSE']:,.1f} USD")
    print(f"Hloubka vyrostlého stromu: {tree_m1.get_depth()} | Počet listů: {tree_m1.get_n_leaves():,}")
    print(f"⚠️ Overfitting Gap (Δ R²): {m1_train_metrics['R2'] - m1_test_metrics['R2']:.4f} (Extrémní přeučení!)")

    all_models_summary.append({
        "Name": "1. Neomezený strom (Baseline)",
        "Train_R2": m1_train_metrics["R2"],
        "R2": m1_test_metrics["R2"],
        "MAE": m1_test_metrics["MAE"],
        "MSE": m1_test_metrics["MSE"],
        "RMSE": m1_test_metrics["RMSE"],
        "Depth": tree_m1.get_depth(),
        "Leaves": tree_m1.get_n_leaves(),
    })

    # Graf predikcí pro Model 1: Skutečné vs. Predikované hodnoty
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, test_preds_m1, alpha=0.3, color="#0284c7", edgecolors="none", s=25)
    max_val = max(y_test.max(), test_preds_m1.max())
    plt.plot([0, max_val], [0, max_val], color="#ef4444", linestyle="--", linewidth=2, label="Ideální predikce (y = ŷ)")
    plt.title("Model 1 (Neomezený strom): Skutečné vs. Predikované ceny nemovitostí", fontsize=13, fontweight="bold")
    plt.xlabel("Skutečná cena nemovitosti (USD)", fontsize=11)
    plt.ylabel("Predikovaná cena stromem (USD)", fontsize=11)
    plt.xlim(0, 4_000_000)
    plt.ylim(0, 4_000_000)
    plt.legend(frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plot_m1_path = plots_dir / "08_decision_tree_m1_actual_vs_predicted.png"
    plt.savefig(plot_m1_path, dpi=150)
    plt.close()
    print(f"Uložen graf predikcí: {plot_m1_path.name}")

    # =========================================================================
    # KROK 8: ITERACE 2 – Model 2 s prořezáním (Omezená hloubka & min_samples_leaf)
    # =========================================================================
    print("\n" + "=" * 80)
    print("[Model 2] Iterace 2: Strom s manuálním omezením hloubky a listů")
    print("=" * 80)
    # Zastavíme růst stromu dříve, abychom zamezili přeučení
    tree_m2 = DecisionTreeRegressor(
        max_depth=8,
        min_samples_leaf=20,
        min_samples_split=40,
        random_state=42
    )
    tree_m2.fit(X_train, y_train)

    train_preds_m2 = tree_m2.predict(X_train)
    test_preds_m2 = tree_m2.predict(X_test)

    m2_train_metrics = calculate_metrics(y_train, train_preds_m2, "Train")
    m2_test_metrics = calculate_metrics(y_test, test_preds_m2, "Test")

    print(f"Trénovací R²: {m2_train_metrics['R2']:.4f} | RMSE: {m2_train_metrics['RMSE']:,.1f} USD")
    print(f"Testovací  R²: {m2_test_metrics['R2']:.4f} | RMSE: {m2_test_metrics['RMSE']:,.1f} USD")
    print(f"Hloubka: {tree_m2.get_depth()} | Počet listů: {tree_m2.get_n_leaves():,}")
    print(f"✅ Overfitting Gap (Δ R²): {m2_train_metrics['R2'] - m2_test_metrics['R2']:.4f} (Sníženo na zlomek!)")

    all_models_summary.append({
        "Name": "2. Prořezaný strom (depth=8, leaf=20)",
        "Train_R2": m2_train_metrics["R2"],
        "R2": m2_test_metrics["R2"],
        "MAE": m2_test_metrics["MAE"],
        "MSE": m2_test_metrics["MSE"],
        "RMSE": m2_test_metrics["RMSE"],
        "Depth": tree_m2.get_depth(),
        "Leaves": tree_m2.get_n_leaves(),
    })

    # =========================================================================
    # KROK 8: ITERACE 3 – Systematická optimalizace přes GridSearchCV
    # =========================================================================
    print("\n" + "=" * 80)
    print("[Model 3] Iterace 3: Systematické ladění hyperparametrů (GridSearchCV)")
    print("=" * 80)
    param_grid = {
        "max_depth": [8, 10, 12, 14],
        "min_samples_leaf": [5, 10, 20],
        "min_samples_split": [10, 20, 40],
        "max_features": [None, 0.8]
    }

    print("Spouštím 5-násobnou křížovou validaci pro 72 kombinací hyperparametrů...")
    grid_search = GridSearchCV(
        estimator=DecisionTreeRegressor(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)

    best_tree = grid_search.best_estimator_
    print(f"Optimum nalezeno: {grid_search.best_params_}")
    print(f"Průměrné validační CV R²: {grid_search.best_score_:.4f}")

    train_preds_m3 = best_tree.predict(X_train)
    test_preds_m3 = best_tree.predict(X_test)

    m3_train_metrics = calculate_metrics(y_train, train_preds_m3, "Train")
    m3_test_metrics = calculate_metrics(y_test, test_preds_m3, "Test")

    print(f"Trénovací R²: {m3_train_metrics['R2']:.4f} | RMSE: {m3_train_metrics['RMSE']:,.1f} USD")
    print(f"Testovací  R²: {m3_test_metrics['R2']:.4f} | RMSE: {m3_test_metrics['RMSE']:,.1f} USD")

    all_models_summary.append({
        "Name": "3. Optimální strom (GridSearchCV)",
        "Train_R2": m3_train_metrics["R2"],
        "R2": m3_test_metrics["R2"],
        "MAE": m3_test_metrics["MAE"],
        "MSE": m3_test_metrics["MSE"],
        "RMSE": m3_test_metrics["RMSE"],
        "Depth": best_tree.get_depth(),
        "Leaves": best_tree.get_n_leaves(),
    })

    # =========================================================================
    # KROK 8: ITERACE 4 – Feature Engineering + Optimální strom
    # =========================================================================
    print("\n" + "=" * 80)
    print("[Model 4] Iterace 4: Přidání doménových příznaků nemovitostí")
    print("=" * 80)
    X_fe = X.copy()
    X_fe["house_age"] = X_fe["yr_sold"] - X_fe["yr_built"]
    X_fe["is_renovated"] = (X_fe["yr_renovated"] > 0).astype(int)
    X_fe["sqft_living_ratio"] = X_fe["sqft_living"] / (X_fe["sqft_lot"] + 1)
    X_fe["total_baths_beds"] = X_fe["bathrooms"] + X_fe["bedrooms"]

    X_train_fe, X_test_fe, y_train_fe, y_test_fe = train_test_split(
        X_fe, y, test_size=0.20, random_state=42
    )

    tree_m4 = DecisionTreeRegressor(
        max_depth=grid_search.best_params_["max_depth"],
        min_samples_leaf=grid_search.best_params_["min_samples_leaf"],
        min_samples_split=grid_search.best_params_["min_samples_split"],
        max_features=grid_search.best_params_["max_features"],
        random_state=42
    )
    tree_m4.fit(X_train_fe, y_train_fe)

    train_preds_m4 = tree_m4.predict(X_train_fe)
    test_preds_m4 = tree_m4.predict(X_test_fe)

    m4_train_metrics = calculate_metrics(y_train_fe, train_preds_m4, "Train")
    m4_test_metrics = calculate_metrics(y_test_fe, test_preds_m4, "Test")

    print(f"Trénovací R²: {m4_train_metrics['R2']:.4f} | RMSE: {m4_train_metrics['RMSE']:,.1f} USD")
    print(f"Testovací  R²: {m4_test_metrics['R2']:.4f} | RMSE: {m4_test_metrics['RMSE']:,.1f} USD")

    all_models_summary.append({
        "Name": "4. Optimální strom + Feature Eng.",
        "Train_R2": m4_train_metrics["R2"],
        "R2": m4_test_metrics["R2"],
        "MAE": m4_test_metrics["MAE"],
        "MSE": m4_test_metrics["MSE"],
        "RMSE": m4_test_metrics["RMSE"],
        "Depth": tree_m4.get_depth(),
        "Leaves": tree_m4.get_n_leaves(),
    })

    # =========================================================================
    # SOUHRNNÉ VYHODNOCENÍ VŠECH ITERACÍ
    # =========================================================================
    print_metrics_table(all_models_summary, "SOUHRNNÉ SROVNÁNÍ VŠECH ITERACÍ ROZHODOVACÍHO STROMU:")

    # -------------------------------------------------------------------------
    # GRAFICKÝ VÝSTUP 1: Srovnání predikcí Model 1 vs. Model 3 (Scatter + 45° přímka)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Model 1
    axes[0].scatter(y_test, test_preds_m1, alpha=0.35, color="#ef4444", s=25)
    axes[0].plot([0, 3_500_000], [0, 3_500_000], color="black", linestyle="--", linewidth=1.5)
    axes[0].set_title(f"Model 1: Neomezený strom\nTest R² = {m1_test_metrics['R2']:.4f} | RMSE = {m1_test_metrics['RMSE']:,.0f} USD", fontsize=12)
    axes[0].set_xlabel("Skutečná cena (USD)")
    axes[0].set_ylabel("Predikovaná cena (USD)")
    axes[0].set_xlim(0, 3_500_000)
    axes[0].set_ylim(0, 3_500_000)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Model 3
    axes[1].scatter(y_test, test_preds_m3, alpha=0.35, color="#10b981", s=25)
    axes[1].plot([0, 3_500_000], [0, 3_500_000], color="black", linestyle="--", linewidth=1.5)
    axes[1].set_title(f"Model 3: Optimální strom (GridSearchCV)\nTest R² = {m3_test_metrics['R2']:.4f} | RMSE = {m3_test_metrics['RMSE']:,.0f} USD", fontsize=12)
    axes[1].set_xlabel("Skutečná cena (USD)")
    axes[1].set_ylabel("Predikovaná cena (USD)")
    axes[1].set_xlim(0, 3_500_000)
    axes[1].set_ylim(0, 3_500_000)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    comp_plot_path = plots_dir / "08_decision_tree_comparison_scatter.png"
    plt.savefig(comp_plot_path, dpi=150)
    plt.close()
    print(f"Uložen srovnávací graf predikcí: {comp_plot_path.name}")

    # -------------------------------------------------------------------------
    # GRAFICKÝ VÝSTUP 2: Feature Importance optimálního modelu
    # -------------------------------------------------------------------------
    feat_imp = pd.Series(best_tree.feature_importances_, index=X.columns).sort_values(ascending=True)
    plt.figure(figsize=(10, 7))
    feat_imp.plot(kind="barh", color="#0284c7")
    plt.title("Důležitost příznaků (Feature Importance) v optimálním rozhodovacím stromu", fontsize=12, fontweight="bold")
    plt.xlabel("Relativní důležitost (Mean Decrease in Impurity)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fi_plot_path = plots_dir / "08_decision_tree_feature_importance.png"
    plt.savefig(fi_plot_path, dpi=150)
    plt.close()
    print(f"Uložen graf důležitosti příznaků: {fi_plot_path.name}")

    # -------------------------------------------------------------------------
    # GRAFICKÝ VÝSTUP 3: Diagram prvních 2 pater optimálního stromu
    # -------------------------------------------------------------------------
    fig_tree, ax_tr = plt.subplots(figsize=(16, 6))
    plot_tree(
        best_tree,
        max_depth=2,
        feature_names=X.columns.tolist(),
        filled=True,
        rounded=True,
        precision=1,
        fontsize=9,
        ax=ax_tr
    )
    plt.title("Vrchní 2 patra optimálního rozhodovacího stromu (King County Reality)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    tree_arch_path = plots_dir / "08_decision_tree_architecture.png"
    plt.savefig(tree_arch_path, dpi=150)
    plt.close()
    print(f"Uložen diagram stromu: {tree_arch_path.name}")

    print("\n" + "=" * 80)
    print(" CVIČENÍ 1 PRO ROZHODOVACÍ STROM DOKONČENO ÚSPĚŠNĚ.")
    print("=" * 80)


if __name__ == "__main__":
    main()
