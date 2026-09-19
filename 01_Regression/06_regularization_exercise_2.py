"""
Linear Regression - Regularization in Linear Regression (Exercise 2)
Dataset: Diamonds Preprocessed (diamonds_preprocessed.csv)
Module: Regularization (Lasso L1, Ridge L2, Elastic Net & Gemology Feature Selection)

ZADÁNÍ / ASSIGNMENT:
------------------------------------------------
1. Load the diamond dataset (diamonds_preprocessed.csv) created in the "Linear Regression - exercise 2".
2. Import all necessary methods from libraries (or modules from these libraries).
3. Divide the dataset into training and test datasets.
4. Using the appropriate class, train a linear regression model in Lasso regularization.
5. Verify the values of the coefficients - check which ones have been zeroed out by the model.
6. Calculate the metrics R2, MAE, MSE and RMSE. Compare these values with the metrics the linear
   regression model legitimized.
7. Using the appropriate class, train a linear regression model with Ridge regularization.
   Use different values of regularization strength.
8. Calculate the R2, MAE, MSE and RMSE metrics. Compare these values with the metrics legitimized
   by the linear regression model without regularization and the model with Lasso regularization.
------------------------------------------------
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LinearRegression, Lasso, Ridge, lasso_path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def adjusted_r2_score(r2: float, n: int, k: int) -> float:
    """Vypočítá korigovaný koeficient determinace."""
    if n - k - 1 <= 0:
        return np.nan
    return 1.0 - ((1.0 - r2) * (n - 1) / (n - k - 1))


def calculate_metrics_dict(y_true, y_pred, n: int, k: int, model_name: str = "Model", alpha: float = 0.0) -> dict:
    """Vypočítá standardní sadu regresních metrik."""
    r2 = r2_score(y_true, y_pred)
    adj_r2 = adjusted_r2_score(r2, n, k)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    return {
        "Model": model_name,
        "Alpha": alpha,
        "R2": r2,
        "Adjusted R2": adj_r2,
        "MAE ($)": mae,
        "MSE ($^2)": mse,
        "RMSE ($)": rmse,
    }


def main() -> None:
    # -------------------------------------------------------------------------
    # 1. Nastavení cest a adresářů
    # -------------------------------------------------------------------------
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    plots_dir = base_dir / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "diamonds_preprocessed.csv"
    if not csv_path.exists():
        fallback_path = base_dir.parent / "data" / "MAL_downloadable materials_session 1" / "Day 1" / "diamonds.csv"
        if fallback_path.exists():
            print(f"Generuji diamonds_preprocessed.csv z {fallback_path}...")
            # Fallback vytvoření pokud by chyběl
            raw = pd.read_csv(fallback_path)
            clean = raw[(raw["x"] > 0) & (raw["y"] > 0) & (raw["z"] > 0) & (raw["y"] < 20) & (raw["z"] < 20) & (raw["carat"] <= 3.5)].copy()
            cut_m = {"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4}
            col_m = {"J": 0, "I": 1, "H": 2, "G": 3, "F": 4, "E": 5, "D": 6}
            cla_m = {"I1": 0, "SI2": 1, "SI1": 2, "VS2": 3, "VS1": 4, "VVS2": 5, "VVS1": 6, "IF": 7}
            clean["cut"] = clean["cut"].map(cut_m)
            clean["color"] = clean["color"].map(col_m)
            clean["clarity"] = clean["clarity"].map(cla_m)
            clean.to_csv(csv_path, index=False)

    # =========================================================================
    # KROK 1: Načtení datasetu diamantů (diamonds_preprocessed.csv)
    # =========================================================================
    print("=" * 80)
    print("KROK 1: Načtení předzpracovaného datasetu diamantů (diamonds_preprocessed.csv)")
    print("=" * 80)
    df = pd.read_csv(csv_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    print(f"Rozměry datasetu: {df.shape[0]} řádků, {df.shape[1]} sloupců.")
    print("Seznam sloupců:", list(df.columns))

    # =========================================================================
    # KROK 2 & 3: Rozdělení dat na Train a Test sadu (80 % / 20 %)
    # =========================================================================
    print("\n" + "=" * 80)
    print("KROK 2 & 3: Rozdělení na matici příznaků X a cílovou proměnnou y (price)")
    print("=" * 80)
    X = df.drop(columns=["price"])
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    n_train, k_features = X_train.shape
    n_test = X_test.shape[0]
    feature_names = list(X.columns)

    print(f"Trénovací sada (Train): n = {n_train} řádků, k = {k_features} příznaků.")
    print(f"Testovací sada  (Test):  n = {n_test} řádků, k = {k_features} příznaků.")

    # =========================================================================
    # KROK 4: Standardizace příznaků (StandardScaler)
    # =========================================================================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # =========================================================================
    # KROK 5: Referenční OLS model (LinearRegression bez regularizace)
    # =========================================================================
    print("\n" + "=" * 80)
    print("KROK 5: Trénování referenčního modelu OLS (LinearRegression bez regularizace)")
    print("=" * 80)
    ols = LinearRegression()
    ols.fit(X_train_scaled, y_train)
    y_pred_ols_test = ols.predict(X_test_scaled)
    y_pred_ols_train = ols.predict(X_train_scaled)

    ols_metrics = calculate_metrics_dict(y_test, y_pred_ols_test, n_test, k_features, "OLS (Bez regularizace)", 0.0)
    print(f"OLS Test R2:      {ols_metrics['R2']:.5f}")
    print(f"OLS Test Adj R2:  {ols_metrics['Adjusted R2']:.5f}")
    print(f"OLS Test MAE:     ${ols_metrics['MAE ($)']:,.2f}")
    print(f"OLS Test RMSE:    ${ols_metrics['RMSE ($)']:,.2f}")

    # =========================================================================
    # KROK 6: Lasso (L1) regularizace – Grid search a detekce nulových vah
    # =========================================================================
    print("\n" + "=" * 80)
    print("KROK 6: Trénování Lasso (L1) modelů pro různé hodnoty alpha")
    print("=" * 80)

    lasso_alphas = [0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    lasso_results = []
    lasso_zeroed_dict = {}

    for a in lasso_alphas:
        lasso = Lasso(alpha=a, random_state=42, max_iter=3000, tol=0.01)
        lasso.fit(X_train_scaled, y_train)
        pred_test = lasso.predict(X_test_scaled)

        zeroed_features = [f for f, w in zip(feature_names, lasso.coef_) if abs(w) < 1e-4]
        active_features = [f for f, w in zip(feature_names, lasso.coef_) if abs(w) >= 1e-4]
        lasso_zeroed_dict[a] = zeroed_features

        m = calculate_metrics_dict(y_test, pred_test, n_test, k_features, f"Lasso (alpha={a})", a)
        m["Zeroed Features Count"] = len(zeroed_features)
        m["Zeroed Features"] = ", ".join(zeroed_features) if zeroed_features else "Žádné"
        lasso_results.append(m)

        print(f"Alpha {a:6.1f} | Test R2: {m['R2']:.5f} | MAE: ${m['MAE ($)']:8,.2f} | RMSE: ${m['RMSE ($)']:8,.2f} | Vynulováno {len(zeroed_features)}/{k_features}: {zeroed_features}")

    lasso_df = pd.DataFrame(lasso_results)
    best_lasso_r2_idx = lasso_df["R2"].idxmax()
    best_lasso_row = lasso_df.loc[best_lasso_r2_idx]
    print(f"\n>> Nejlepší Lasso model podle R2: Alpha = {best_lasso_row['Alpha']} s R2 = {best_lasso_row['R2']:.5f}, MAE = ${best_lasso_row['MAE ($)']:,.2f}")

    # =========================================================================
    # KROK 7: Ridge (L2) regularizace – Grid search a shrinkage vah
    # =========================================================================
    print("\n" + "=" * 80)
    print("KROK 7: Trénování Ridge (L2) modelů pro různé hodnoty alpha")
    print("=" * 80)

    ridge_alphas = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0]
    ridge_results = []

    for a in ridge_alphas:
        ridge = Ridge(alpha=a, random_state=42)
        ridge.fit(X_train_scaled, y_train)
        pred_test = ridge.predict(X_test_scaled)

        m = calculate_metrics_dict(y_test, pred_test, n_test, k_features, f"Ridge (alpha={a})", a)
        ridge_results.append(m)

        print(f"Alpha {a:8.2f} | Test R2: {m['R2']:.5f} | MAE: ${m['MAE ($)']:8,.2f} | RMSE: ${m['RMSE ($)']:8,.2f}")

    ridge_df = pd.DataFrame(ridge_results)
    best_ridge_r2_idx = ridge_df["R2"].idxmax()
    best_ridge_row = ridge_df.loc[best_ridge_r2_idx]
    print(f"\n>> Nejlepší Ridge model podle R2: Alpha = {best_ridge_row['Alpha']} s R2 = {best_ridge_row['R2']:.5f}, MAE = ${best_ridge_row['MAE ($)']:,.2f}")

    # =========================================================================
    # KROK 8: Porovnávací tabulka OLS vs. Nejlepší Lasso vs. Nejlepší Ridge
    # =========================================================================
    print("\n" + "=" * 80)
    print("KROK 8: Souhrnné srovnání – OLS vs. Nejlepší Lasso vs. Nejlepší Ridge")
    print("=" * 80)

    summary_rows = [
        ols_metrics,
        {
            "Model": f"Lasso (alpha={best_lasso_row['Alpha']})",
            "Alpha": best_lasso_row["Alpha"],
            "R2": best_lasso_row["R2"],
            "Adjusted R2": best_lasso_row["Adjusted R2"],
            "MAE ($)": best_lasso_row["MAE ($)"],
            "MSE ($^2)": best_lasso_row["MSE ($^2)"],
            "RMSE ($)": best_lasso_row["RMSE ($)"],
        },
        {
            "Model": f"Ridge (alpha={best_ridge_row['Alpha']})",
            "Alpha": best_ridge_row["Alpha"],
            "R2": best_ridge_row["R2"],
            "Adjusted R2": best_ridge_row["Adjusted R2"],
            "MAE ($)": best_ridge_row["MAE ($)"],
            "MSE ($^2)": best_ridge_row["MSE ($^2)"],
            "RMSE ($)": best_ridge_row["RMSE ($)"],
        },
    ]
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    # =========================================================================
    # KROK 9: Generování diagnostických grafů
    # =========================================================================
    print("\n" + "=" * 80)
    print("KROK 9: Generování a ukládání grafů")
    print("=" * 80)

    # Graf 1: R2 a MAE vs. Alpha (Lasso vs Ridge)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].plot(lasso_df["Alpha"], lasso_df["R2"], marker="o", linewidth=2, color="#E63946", label="Lasso (L1)")
    axes[0].plot(ridge_df["Alpha"], ridge_df["R2"], marker="s", linewidth=2, color="#1D3557", label="Ridge (L2)")
    axes[0].axhline(ols_metrics["R2"], color="black", linestyle="--", alpha=0.7, label=f"OLS Baseline ({ols_metrics['R2']:.4f})")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Sila regularizace alpha (log scale)", fontsize=11)
    axes[0].set_ylabel("R2 skore", fontsize=11)
    axes[0].set_title("Vyvoj R2 skore u diamantu podle sily regularizace", fontsize=13, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(lasso_df["Alpha"], lasso_df["MAE ($)"], marker="o", linewidth=2, color="#E63946", label="Lasso (L1)")
    axes[1].plot(ridge_df["Alpha"], ridge_df["MAE ($)"], marker="s", linewidth=2, color="#1D3557", label="Ridge (L2)")
    axes[1].axhline(ols_metrics["MAE ($)"], color="black", linestyle="--", alpha=0.7, label=f"OLS Baseline (${ols_metrics['MAE ($)']:,.0f})")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Sila regularizace alpha (log scale)", fontsize=11)
    axes[1].set_ylabel("MAE v USD", fontsize=11)
    axes[1].set_title("Vyvoj MAE u diamantu podle sily regularizace", fontsize=13, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path_metrics = plots_dir / "14_diamonds_regularization_metrics_vs_alpha.png"
    plt.savefig(plot_path_metrics, dpi=300)
    plt.close()
    print(f"Graf metrik ulozen: {plot_path_metrics}")

    # Graf 2: Lasso Regularization Path u diamantu
    alphas_path, coefs_path, _ = lasso_path(X_train_scaled, y_train, eps=1e-3)

    plt.figure(figsize=(14, 8))
    cmap = plt.get_cmap("tab10")
    for i, feature in enumerate(feature_names):
        plt.plot(alphas_path, coefs_path[i, :], label=feature, linewidth=2, color=cmap(i % 10))

    plt.xscale("log")
    plt.axhline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.8)
    plt.xlabel("Sila regularizace alpha (logaritmicka skala)", fontsize=12)
    plt.ylabel("Hodnota standardizovaneho regresniho koeficientu", fontsize=12)
    plt.title("Lasso Regularization Path pro diamanty: Eliminace prostorove multikolinearity", fontsize=14, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path_lasso = plots_dir / "15_diamonds_lasso_coefficient_paths.png"
    plt.savefig(plot_path_lasso, dpi=300)
    plt.close()
    print(f"Graf Lasso Path ulozen: {plot_path_lasso}")

    print("\n" + "=" * 80)
    print("DOKONCENO: Vypocty a vizualizace pro Cviceni 2 (Regularizace diamantu) probehly uspesne.")
    print("=" * 80)


if __name__ == "__main__":
    main()
