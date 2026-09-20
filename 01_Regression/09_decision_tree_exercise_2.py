"""
Decision Tree in Regression - Exercise 2
Dataset: Diamond prices (diamonds.csv)
Module: Reusable Evaluation Wrapper Function, Hyperparameter Tuning & Precision Benchmarking

ZADÁNÍ / ASSIGNMENT:
------------------------------------------------
1. Import all necessary methods from libraries (or modules from these libraries).
2. Load a dataset from the diamonds.csv file, using the appropriate method from the Pandas library.
3. Divide the dataset into training and test datasets.
4. Initialize the decision tree model. Select a starting set of hyperparameters.
5. Train the decision tree model.
6. Generate a graph that visualizes how the trained model calculates predictions.
7. Calculate the metrics R2, MAE, MSE and RMSE.
8. Repeat steps 2, 3, 4 and 5 until you have an optimal model.
To speed up the model building process:
- call the method that fits the data to the model,
- generate a graph that shows how the trained tree works,
- calculate the metrics of the model,
Wrap it in a function that takes as arguments the values of the hyperparameters you want to set.
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


# Globální reference pro vizualizace
PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# WRAPPER FUNCTION REQUIRED BY ASSIGNMENT
# =============================================================================
def train_and_evaluate_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: int | float | str | None = None,
    model_name: str = "Decision Tree",
    save_plot_filename: str | None = None,
    show_plot: bool = False
) -> dict:
    """
    Společná funkce požadovaná zadáním:
    1. Natrénuje model DecisionTreeRegressor s předanými hyperparametry.
    2. Vygeneruje a uloží graf predikcí (Skutečné vs. Predikované hodnoty + přímka y=ŷ).
    3. Spočítá metriky R2, MAE, MSE a RMSE pro trénovací i testovací sadu.
    
    Vrací slovník s natrénovaným modelem a přehledem metrik.
    """
    # 1. Inicializace a trénování modelu
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 2. Predikce
    preds_train = model.predict(X_train)
    preds_test = model.predict(X_test)

    # 3. Výpočet metrik
    r2_tr = r2_score(y_train, preds_train)
    r2_te = r2_score(y_test, preds_test)
    mae_te = mean_absolute_error(y_test, preds_test)
    mse_te = mean_squared_error(y_test, preds_test)
    rmse_te = np.sqrt(mse_te)
    gap = r2_tr - r2_te

    # 4. Generování a uložení grafu predikcí
    if save_plot_filename:
        plot_path = PLOTS_DIR / save_plot_filename
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(y_test, preds_test, alpha=0.30, color="#0284c7", s=18, edgecolors="none")
        max_limit = max(y_test.max(), preds_test.max())
        ax.plot([0, max_limit], [0, max_limit], color="#ef4444", linestyle="--", linewidth=2, label="Ideální predikce (y = ŷ)")
        ax.set_title(f"{model_name}\nTest R² = {r2_te:.4f} | RMSE = {rmse_te:,.1f} USD | MAE = {mae_te:,.1f} USD", fontsize=12, fontweight="bold")
        ax.set_xlabel("Skutečná cena diamantu (USD)", fontsize=11)
        ax.set_ylabel("Predikovaná cena stromem (USD)", fontsize=11)
        ax.set_xlim(0, 20_000)
        ax.set_ylim(0, 20_000)
        ax.legend(frameon=True)
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        if show_plot:
            plt.show()
        plt.close()
        print(f"  -> Uložen graf predikcí: {plot_path.name}")

    # Výstupní slovník
    return {
        "model": model,
        "metrics": {
            "Model": model_name,
            "max_depth": str(max_depth),
            "min_leaf": min_samples_leaf,
            "min_split": min_samples_split,
            "Train_R2": r2_tr,
            "Test_R2": r2_te,
            "Gap_R2": gap,
            "Test_MAE": mae_te,
            "Test_MSE": mse_te,
            "Test_RMSE": rmse_te,
            "Depth": model.get_depth(),
            "Leaves": model.get_n_leaves()
        }
    }


def main() -> None:
    print("=" * 80)
    print(" CVIČENÍ 2: ROZHODOVACÍ STROM V REGRESI (CENY DIAMANTŮ - DIAMONDS.CSV)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # KROK 1 & 2: Načtení a předzpracování dat
    # -------------------------------------------------------------------------
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "data" / "diamonds.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Soubor s diamanty nebyl nalezen: {data_path}")

    print("\n[Krok 1 & 2] Načítání datasetu diamonds.csv...")
    df_raw = pd.read_csv(data_path)
    if "Unnamed: 0" in df_raw.columns:
        df_raw = df_raw.drop(columns=["Unnamed: 0"])

    print(f"Původní rozměry databáze diamantů: {df_raw.shape[0]:,} řádků, {df_raw.shape[1]} sloupců")

    # Čištění fyzikálně nemožných nulových rozměrů (x=0, y=0, z=0)
    clean_df = df_raw[(df_raw["x"] > 0) & (df_raw["y"] > 0) & (df_raw["z"] > 0)].copy()
    print(f"Odstraněno {len(df_raw) - len(clean_df)} nulových rozměrů. Čistých záznamů: {len(clean_df):,}")

    # Ordinální kódování 4C (přirozené pro větvení stromu v uzlech)
    cut_order = {"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5}
    color_order = {"J": 1, "I": 2, "H": 3, "G": 4, "F": 5, "E": 6, "D": 7}
    clarity_order = {"I1": 1, "SI2": 2, "SI1": 3, "VS2": 4, "VS1": 5, "VVS2": 6, "VVS1": 7, "IF": 8}

    clean_df["cut"] = clean_df["cut"].map(cut_order)
    clean_df["color"] = clean_df["color"].map(color_order)
    clean_df["clarity"] = clean_df["clarity"].map(clarity_order)

    # -------------------------------------------------------------------------
    # KROK 3: Rozdělení dat na trénovací a testovací sadu (80/20)
    # -------------------------------------------------------------------------
    print("\n[Krok 3] Rozdělení dat na trénovací (80 %) a testovací (20 %) sadu...")
    X = clean_df.drop(columns=["price"])
    y = clean_df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Trénovací data: {X_train.shape[0]:,} diamantů")
    print(f"Testovací data:  {X_test.shape[0]:,} diamantů")

    all_iterations = []

    # =========================================================================
    # KROK 4–7: ITERACE 1 – Výchozí model (Neomezený strom - Baseline)
    # =========================================================================
    print("\n" + "=" * 80)
    print("[Iterace 1] Výchozí neomezený strom (Baseline: max_depth=None)")
    print("=" * 80)
    res_m1 = train_and_evaluate_tree(
        X_train, y_train, X_test, y_test,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        model_name="1. Neomezený strom (Baseline)",
        save_plot_filename="09_diamonds_tree_m1_baseline.png"
    )
    m1 = res_m1["metrics"]
    print(f"Trénovací R²: {m1['Train_R2']:.4f} | Testovací R²: {m1['Test_R2']:.4f}")
    print(f"Testovací MAE: {m1['Test_MAE']:,.1f} USD | Testovací RMSE: {m1['Test_RMSE']:,.1f} USD")
    print(f"Hloubka: {m1['Depth']} | Listů: {m1['Leaves']:,} | Overfitting Gap: {m1['Gap_R2']:.4f}")
    all_iterations.append(m1)

    # =========================================================================
    # KROK 8: ITERACE 2 – Manuální prořezání (Mělký strom pro rychlý baseline)
    # =========================================================================
    print("\n" + "=" * 80)
    print("[Iterace 2] Mělký prořezaný strom (max_depth=6, min_leaf=20)")
    print("=" * 80)
    res_m2 = train_and_evaluate_tree(
        X_train, y_train, X_test, y_test,
        max_depth=6,
        min_samples_split=40,
        min_samples_leaf=20,
        model_name="2. Mělký strom (depth=6)",
        save_plot_filename="09_diamonds_tree_m2_pruned.png"
    )
    m2 = res_m2["metrics"]
    print(f"Trénovací R²: {m2['Train_R2']:.4f} | Testovací R²: {m2['Test_R2']:.4f}")
    print(f"Testovací MAE: {m2['Test_MAE']:,.1f} USD | Testovací RMSE: {m2['Test_RMSE']:,.1f} USD")
    print(f"Hloubka: {m2['Depth']} | Listů: {m2['Leaves']:,} | Overfitting Gap: {m2['Gap_R2']:.4f}")
    all_iterations.append(m2)

    # =========================================================================
    # KROK 8: ITERACE 3 – Vyvážený střední strom (max_depth=10)
    # =========================================================================
    print("\n" + "=" * 80)
    print("[Iterace 3] Vyvážený strom (max_depth=10, min_leaf=10)")
    print("=" * 80)
    res_m3 = train_and_evaluate_tree(
        X_train, y_train, X_test, y_test,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        model_name="3. Vyvážený strom (depth=10)",
        save_plot_filename="09_diamonds_tree_m3_balanced.png"
    )
    m3 = res_m3["metrics"]
    print(f"Trénovací R²: {m3['Train_R2']:.4f} | Testovací R²: {m3['Test_R2']:.4f}")
    print(f"Testovací MAE: {m3['Test_MAE']:,.1f} USD | Testovací RMSE: {m3['Test_RMSE']:,.1f} USD")
    print(f"Hloubka: {m3['Depth']} | Listů: {m3['Leaves']:,} | Overfitting Gap: {m3['Gap_R2']:.4f}")
    all_iterations.append(m3)

    # =========================================================================
    # KROK 8: ITERACE 4 – Systematická optimalizace přes GridSearchCV
    # =========================================================================
    print("\n" + "=" * 80)
    print("[Iterace 4] Systematické hledání globálního optima (GridSearchCV)")
    print("=" * 80)
    param_grid = {
        "max_depth": [11, 12, 13, 14],
        "min_samples_leaf": [4, 6, 8, 12],
        "min_samples_split": [15, 25, 35]
    }
    print("Prohledávám 48 kombinací pomocí 5-násobné křížové validace...")
    grid_search = GridSearchCV(
        estimator=DecisionTreeRegressor(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    best_p = grid_search.best_params_
    print(f"Nalezeno optimální nastavení: {best_p}")
    print(f"Průměrné validační CV R²: {grid_search.best_score_:.4f}")

    # Vyhodnocení finálního optima přes naši wrapper funkci
    res_m4 = train_and_evaluate_tree(
        X_train, y_train, X_test, y_test,
        max_depth=best_p["max_depth"],
        min_samples_split=best_p["min_samples_split"],
        min_samples_leaf=best_p["min_samples_leaf"],
        model_name="4. Optimální strom (GridSearchCV)",
        save_plot_filename="09_diamonds_tree_m4_optimal.png"
    )
    m4 = res_m4["metrics"]
    print(f"Trénovací R²: {m4['Train_R2']:.4f} | Testovací R²: {m4['Test_R2']:.4f}")
    print(f"Testovací MAE: {m4['Test_MAE']:,.1f} USD | Testovací RMSE: {m4['Test_RMSE']:,.1f} USD")
    print(f"Hloubka: {m4['Depth']} | Listů: {m4['Leaves']:,} | Overfitting Gap: {m4['Gap_R2']:.4f}")
    all_iterations.append(m4)

    # =========================================================================
    # SOUHRNNÁ TABULKA VŠECH ITERACÍ
    # =========================================================================
    df_results = pd.DataFrame(all_iterations)
    print("\n" + "=" * 90)
    print("SOUHRNNÁ SROVNÁVACÍ TABULKA VŠECH ITERACÍ ROZHODOVACÍHO STROMU (DIAMONDS):")
    print("=" * 90)
    print(f"{'Model':<34} | {'Train R²':>8} | {'Test R²':>8} | {'Δ R²':>7} | {'Test MAE':>10} | {'Test RMSE':>10}")
    print("-" * 90)
    for _, row in df_results.iterrows():
        print(f"{row['Model']:<34} | {row['Train_R2']:>8.4f} | {row['Test_R2']:>8.4f} | {row['Gap_R2']:>7.4f} | {row['Test_MAE']:>9,.1f}$ | {row['Test_RMSE']:>9,.1f}$")
    print("-" * 90)

    # -------------------------------------------------------------------------
    # GRAFICKÝ VÝSTUP: Feature Importance optimálního stromu
    # -------------------------------------------------------------------------
    opt_tree = res_m4["model"]
    feat_imp = pd.Series(opt_tree.feature_importances_, index=X.columns).sort_values(ascending=True)

    plt.figure(figsize=(10, 6))
    feat_imp.plot(kind="barh", color="#0284c7")
    plt.title("Důležitost příznaků (Feature Importance) v optimálním stromu pro diamanty", fontsize=12, fontweight="bold")
    plt.xlabel("Relativní významnost (MDI)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fi_plot_path = PLOTS_DIR / "09_diamonds_feature_importance.png"
    plt.savefig(fi_plot_path, dpi=150)
    plt.close()
    print(f"Uložen graf Feature Importance: {fi_plot_path.name}")

    print("\n" + "=" * 80)
    print(" CVIČENÍ 2 DOKONČENO ÚSPĚŠNĚ.")
    print("=" * 80)


if __name__ == "__main__":
    main()
