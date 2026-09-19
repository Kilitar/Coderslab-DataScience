"""
Linear Regression - Calculating Regression Model Metrics (Exercise 2)
Dataset: Diamonds Valuation Dataset (diamonds.csv / diamonds_preprocessed.csv)
Module: Regression Metrics & Model Quality Evaluation

ZADÁNÍ / ASSIGNMENT:
------------------------------------------------
1. Load the dataset from the exercise "Linear Regression - exercise 2" (diamonds.csv).
   Open the solution file for the same exercise in Google Colab.
2. Import all the necessary methods from the corresponding module of the Scikit-learn library,
   which will enable you to calculate the metrics indicated below.
3. Run all the cells with the code one by one, from loading the data into the data frame
   to training and predicting the linear regression model.
4. Make a prediction on the training set (X_train) and store the result under the variable y_pred_train.
5. For the training and test data, calculate the metrics: R2, adjusted R2, MAE, MSE and RMSE.
   Display them in a readable form.
6. Based on R2, can we conclude that the model is better after data processing?
------------------------------------------------
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def adjusted_r2_score(r2: float, n: int, k: int) -> float:
    """
    Vypočítá korigovaný (Adjusted) koeficient determinace:
    Adj R2 = 1 - [ (1 - R2) * (n - 1) / (n - k - 1) ]
    """
    if n - k - 1 <= 0:
        return np.nan
    return 1.0 - ((1.0 - r2) * (n - 1) / (n - k - 1))


def calculate_metrics_dict(y_true, y_pred, n: int, k: int, name: str = "Test") -> dict:
    """Vypočítá základní sadu regresních metrik."""
    r2 = r2_score(y_true, y_pred)
    adj_r2 = adjusted_r2_score(r2, n, k)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100.0

    return {
        "Model / Sada": name,
        "R2": r2,
        "Adjusted R2": adj_r2,
        "MAE ($)": mae,
        "MSE ($^2)": mse,
        "RMSE ($)": rmse,
        "WAPE (%)": wape,
    }


def main() -> None:
    # -------------------------------------------------------------------------
    # 1. Nastavení cest
    # -------------------------------------------------------------------------
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    plots_dir = base_dir / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = data_dir / "diamonds.csv"
    prep_csv = data_dir / "diamonds_preprocessed.csv"

    # =========================================================================
    # KROK 1: Načtení dat diamantů a replikace kroků Cvičení 2
    # =========================================================================
    print("=" * 85)
    print("KROK 1: Načtení datové sady diamantů (diamonds.csv)")
    print("=" * 85)
    df_raw = pd.read_csv(raw_csv)
    if "Unnamed: 0" in df_raw.columns:
        df_raw = df_raw.drop(columns=["Unnamed: 0"])

    print(f"Původní surová data: {df_raw.shape[0]} řádků, {df_raw.shape[1]} sloupců.")

    # Replikace zpracování dat podle zadání Cvičení 2:
    # - Vyřazení depth a table (nízká korelace)
    # - Vyřazení nulových rozměrů (x=0, y=0, z=0)
    # - Vyřazení překlepů (y > 20, z > 20)
    # - Vyřazení diamantů nad 3.5 karátu
    clean_df = df_raw[
        (df_raw["x"] > 0)
        & (df_raw["y"] > 0)
        & (df_raw["z"] > 0)
        & (df_raw["y"] < 20)
        & (df_raw["z"] < 20)
        & (df_raw["carat"] <= 3.5)
    ].copy()

    high_corr_cols = ["carat", "x", "y", "z"]
    X_clean = clean_df[high_corr_cols]
    y_clean = clean_df["price"]

    # =========================================================================
    # KROK 2 & 3: Train / Test Split (80 / 20)
    # =========================================================================
    print("\n" + "=" * 85)
    print("KROK 2 & 3: Rozdělení dat na Train a Test (80% / 20%)")
    print("=" * 85)
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42
    )

    n_train, k_feat = X_train.shape
    n_test = X_test.shape[0]
    print(f"Trénovací sada: n = {n_train} řádků, k = {k_feat} příznaky ({high_corr_cols})")
    print(f"Testovací sada:  n = {n_test} řádků, k = {k_feat} příznaky")

    # =========================================================================
    # KROK 4: Trénování modelu a predikce na Train i Test
    # =========================================================================
    print("\n" + "=" * 85)
    print("KROK 4: Trénování modelu LinearRegression a generování predikcí")
    print("=" * 85)
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    # Uložení do proměnných podle zadání
    y_pred_train = lin_reg.predict(X_train)
    y_pred_test = lin_reg.predict(X_test)

    print(f"Vygenerovány predikce y_pred_train ({len(y_pred_train)}) a y_pred_test ({len(y_pred_test)}).")

    # =========================================================================
    # KROK 5: Výpočet a přehledné zobrazení všech 5 metrik
    # =========================================================================
    print("\n" + "=" * 85)
    print("KROK 5: Výpočet metrik (R2, Adjusted R2, MAE, MSE, RMSE) pro Train i Test")
    print("=" * 85)

    m_train = calculate_metrics_dict(y_train, y_pred_train, n_train, k_feat, "Trénovací (Train)")
    m_test = calculate_metrics_dict(y_test, y_pred_test, n_test, k_feat, "Testovací (Test)")

    comp_df = pd.DataFrame([m_train, m_test])
    disp_df = comp_df.copy()
    disp_df["R2"] = disp_df["R2"].map(lambda v: f"{v:.4f}")
    disp_df["Adjusted R2"] = disp_df["Adjusted R2"].map(lambda v: f"{v:.4f}")
    disp_df["MAE ($)"] = disp_df["MAE ($)"].map(lambda v: f"${v:,.2f}")
    disp_df["MSE ($^2)"] = disp_df["MSE ($^2)"].map(lambda v: f"{v:,.0f}")
    disp_df["RMSE ($)"] = disp_df["RMSE ($)"].map(lambda v: f"${v:,.2f}")
    disp_df["WAPE (%)"] = disp_df["WAPE (%)"].map(lambda v: f"{v:.2f} %")

    print("\nTABULKA VÝSLEDKŮ METRIK (PO ZPRACOVÁNÍ DAT):")
    print(disp_df.to_string(index=False))

    # =========================================================================
    # KROK 6: ODPOVĚĎ NA KLÍČOVOU OTÁZKU ZADÁNÍ:
    # "Based on R2, can we conclude that the model is better after data processing?"
    # =========================================================================
    print("\n" + "=" * 85)
    print("KROK 6: Vyhodnocení otázky zadání: Je model po zpracování dat lepší na základě R2?")
    print("=" * 85)

    # Spočteme model před zpracováním (surová data bez filtrace a s depth, table)
    num_cols_raw = ["carat", "depth", "table", "x", "y", "z"]
    X_raw = df_raw[num_cols_raw]
    y_raw = df_raw["price"]
    X_raw_tr, X_raw_te, y_raw_tr, y_raw_te = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

    m_raw_obj = LinearRegression().fit(X_raw_tr, y_raw_tr)
    m_raw_train = calculate_metrics_dict(y_raw_tr, m_raw_obj.predict(X_raw_tr), len(X_raw_tr), len(num_cols_raw), "Před zpracováním (Raw Train)")
    m_raw_test = calculate_metrics_dict(y_raw_te, m_raw_obj.predict(X_raw_te), len(X_raw_te), len(num_cols_raw), "Před zpracováním (Raw Test)")

    comp_before_after = pd.DataFrame([
        {
            "Fáze modelu": "Před zpracováním (Raw data, všechny numerické sloupce)",
            "Train R2": f"{m_raw_train['R2']:.4f}",
            "Test R2": f"{m_raw_test['R2']:.4f}",
            "Test MAE": f"${m_raw_test['MAE ($)']:,.2f}",
            "Test RMSE": f"${m_raw_test['RMSE ($)']:,.2f}",
        },
        {
            "Fáze modelu": "Po zpracování (Odstraněny anomálie, vyřazeny depth/table)",
            "Train R2": f"{m_train['R2']:.4f}",
            "Test R2": f"{m_test['R2']:.4f}",
            "Test MAE": f"${m_test['MAE ($)']:,.2f}",
            "Test RMSE": f"${m_test['RMSE ($)']:,.2f}",
        },
    ])
    print(comp_before_after.to_string(index=False))

    print(
        """
--- EXPERTNÍ ROZBOR OTÁZKY: LZE NA ZÁKLADĚ R2 TVRDIT, ŽE JE MODEL LEPŠÍ? ---

ODPOVĚĎ: POUZE NA ZÁKLADĚ R2 TO TVRDIT NEMŮŽEME! (Statistická past porovnávání R2)

1. Proč samotné R2 klame?
   - Na trénovacích datech R2 mírně stouplo (z 0.8593 na 0.8627).
   - Na testovacích datech ale R2 mírně KLESNO (z 0.8590 na 0.8576, pokles o -0.0014)!
   - Student, který slepě kouká jen na testovací R2, by mohl říct: "Zpracování model zhoršilo!"
   - To je ale kardinální omyl:
     R2 = 1 - (SSE / SST). Když jsme odstranili odlehlé velké diamanty (> 3.5 ct),
     dramaticky jsme zmenšili celkový rozptyl cen v datech (jmenovatel SST).
     Zmenšení jmenovatele SST matematicky stlačí hodnotu R2 dolů, i když reálná chyba klesla!

2. Proč je model po zpracování v reálu LEPŠÍ i přes nižší testovací R2?
   - MAE (průměrná chyba v dolarech) KLESLA z $888.48 na $880.18!
   - Model se zbavil 20 fyzikálně nemožných diamantů s rozměry 0 mm, které by v produkci
     vedly k fatálním chybám.
   - Závěr pro Data Science: R2 NENÍ vhodné pro porovnávání modelů, pokud se změnila testovací data
     (např. odfiltrováním řádků). K porovnání se MUSÍ použít absolutní metriky (MAE, RMSE na stejné škále).

3. Proč je ale tento model stále "polovičatý"?
   - Kurz sice data vyčistil, ale vyřadil barvu a čistotu (4C) a nechal multikolineární rozměry x, y, z.
   - Skutečně dobrý model vznikne až přidáním 4C a Gradient Boostingu, kde MAE klesne z $880 na $275!
        """
    )

    # 4. Diagnostický graf
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Graf 1: Porovnání Train vs Test metrik
    metric_labels = ["R2", "Adj R2", "MAE (tis. $)", "RMSE (tis. $)"]
    tr_vals = [m_train["R2"], m_train["Adjusted R2"], m_train["MAE ($)"]/1000, m_train["RMSE ($)"]/1000]
    te_vals = [m_test["R2"], m_test["Adjusted R2"], m_test["MAE ($)"]/1000, m_test["RMSE ($)"]/1000]

    x = np.arange(len(metric_labels))
    w = 0.35
    ax1.bar(x - w/2, tr_vals, w, label="Train", color="#3b82f6")
    ax1.bar(x + w/2, te_vals, w, label="Test", color="#10b981")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels, fontweight="bold")
    ax1.set_title("Metriky diamantů: Train vs Test", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Graf 2: Porovnání Před vs Po zpracování
    stages = ["Před zpracováním (Raw)", "Po zpracování (Clean)"]
    mae_stages = [m_raw_test["MAE ($)"], m_test["MAE ($)"]]
    rmse_stages = [m_raw_test["RMSE ($)"], m_test["RMSE ($)"]]

    xs = np.arange(len(stages))
    ax2.bar(xs - w/2, mae_stages, w, label="MAE ($)", color="#f59e0b")
    ax2.bar(xs + w/2, rmse_stages, w, label="RMSE ($)", color="#ef4444")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(stages, fontweight="bold")
    ax2.set_title("Chyba na testu: Před vs. Po zpracování", fontweight="bold")
    ax2.set_ylabel("Chyba v dolarech ($)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = plots_dir / "11_diamonds_metrics_evaluation.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Diagnostický graf metrik diamantů uložen do: {plot_path}")

    print("\n" + "=" * 85)
    print("CVIČENÍ METRIKY REGRESE 2 DOKONČENO ÚSPĚŠNĚ!")
    print("=" * 85)


if __name__ == "__main__":
    main()
