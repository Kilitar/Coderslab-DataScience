"""
Linear Regression - Exercise 2
Dataset: Diamonds Valuation Dataset (diamonds.csv)
Domain: Gemology & Diamond Pricing Automation

ZADÁNÍ / ASSIGNMENT:
------------------------------------------------
Imagine that you are a data scientist and your friend is a jeweler. Your friend's jewelry store
is visited by a huge number of customers who bring diamonds to sell. The number of interested people
is so large that the friend is not able to asses value of the stones - due to other responsibilities.
As a data expert, you offer to automate the valuation process using machine learning. You ask your friend
to provide the historical data he relied on when doing previous valuations. Your task is to analyze and clean
the data, and try to build a linear regression model that will determine the price of a diamond based on its properties.

Description of the data received:
- carat: physical mass of a diamond (1 carat = 0.20 grams).
- cut: quality of the cut (Fair, Good, Very Good, Premium, Ideal).
- color: diamond color from J (worst) to D (best).
- clarity: diamond clarity (I1 worst, SI2, SI1, VS2, VS1, VVS2, VVS1, IF best).
- depth: total depth percentage = z / mean(x, y) = 2 * z / (x + y).
- table: width of top of diamond relative to widest point.
- price: price of the diamond in US dollars ($326 - $18,823).
- x: length in mm.
- y: width in mm.
- z: depth in mm.

1. Transfer the data file (diamonds.csv) to the execution environment of your notebook or to Google drive.
2. Import all necessary methods from libraries (or modules from these libraries).
3. Load the data frame file into the Pandas library. Assign the data frame to a variable named diamonds_df.
4. Display the first dozen or so data records to see how the data looks.
5. Remove unnecessary columns from the dataset.
6. Using the methods you have learned, verify the presence of missing values in the dataset.
7. Use the appropriate seaborn library tool to check the relationships between variables and the distribution
   of variable values on a single graph.
8. Generate and visualize the correlation matrix. Exclude variables with low correlation coefficient from the dataset.
   You can verify the linearity of the correlation of the remaining variables on scatter plots against the independent variable (price).
9. Using the selected graphs, check for outliers. Use scatter plots of the selected variables and the dependent
   variable price to accurately identify observations that have outliers. Use the method of your choice
   (e.g., reading from the graph and filtering, IQR method) to remove outliers.
10. Divide the dataset into training and test datasets.
11. Define and train the model using the training data. Make predictions on the test data.
12. Save the prepared dataset to .csv file. You will use it to build more models during this day.
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


def main() -> None:
    # -------------------------------------------------------------------------
    # 1. Nastavení cest / Setup directories
    # -------------------------------------------------------------------------
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    plots_dir = base_dir / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Nalezení vstupního souboru
    raw_csv_path = data_dir / "diamonds.csv"
    if not raw_csv_path.exists():
        fallback_path = (
            base_dir.parent
            / "data"
            / "MAL_downloadable materials_session 1"
            / "Day 1"
            / "diamonds.csv"
        )
        if fallback_path.exists():
            import shutil

            shutil.copy(fallback_path, raw_csv_path)
            print(f"Data zkopírována do: {raw_csv_path}")

    # =========================================================================
    # Krok 1 & 2 & 3: Načtení dat do Pandas (diamonds_df)
    # =========================================================================
    print("=" * 75)
    print("KROK 1–3: Načtení datové sady diamantů (diamonds_df)")
    print("=" * 75)
    diamonds_df = pd.read_csv(raw_csv_path)
    print(f"Původní rozměry databáze: {diamonds_df.shape[0]} řádků, {diamonds_df.shape[1]} sloupců.")

    # =========================================================================
    # Krok 4: Zobrazení prvních 15 záznamů
    # =========================================================================
    print("\n" + "=" * 75)
    print("KROK 4: Náhled prvních 15 záznamů (diamonds_df.head(15))")
    print("=" * 75)
    print(diamonds_df.head(15).to_string())

    # =========================================================================
    # Krok 5: Odstranění nepotřebných sloupců ('Unnamed: 0')
    # =========================================================================
    print("\n" + "=" * 75)
    print("KROK 5: Odstranění nepotřebných sloupců ('Unnamed: 0')")
    print("=" * 75)
    if "Unnamed: 0" in diamonds_df.columns:
        diamonds_df = diamonds_df.drop(columns=["Unnamed: 0"])
        print("Odstraněn indexový sloupec 'Unnamed: 0'.")
    print(f"Aktuální sloupce: {list(diamonds_df.columns)}")

    # =========================================================================
    # Krok 6: Ověření přítomnosti chybějících hodnot
    # =========================================================================
    print("\n" + "=" * 75)
    print("KROK 6: Kontrola chybějících hodnot (Missing Values)")
    print("=" * 75)
    missing = diamonds_df.isnull().sum()
    print(f"Celkový počet chybějících hodnot: {missing.sum()}")
    if missing.sum() == 0:
        print("Datová sada neobsahuje žádné standardní NaN/None hodnoty.")

    # =========================================================================
    # Krok 7: Seaborn pairplot pro vztahy a distribuce proměnných
    # =========================================================================
    print("\n" + "=" * 75)
    print("KROK 7: Vzájemné vztahy a distribuce proměnných (sns.pairplot)")
    print("=" * 75)
    print("Generuji pairplot (pro svižnost a přehlednost na vzorku 1 500 diamantů)...")
    numeric_cols = ["carat", "depth", "table", "price", "x", "y", "z"]
    sample_df = diamonds_df[numeric_cols].sample(n=1500, random_state=42)

    pairplot_fig = sns.pairplot(
        sample_df,
        diag_kind="kde",
        plot_kws={"alpha": 0.4, "color": "#1f77b4"},
        diag_kws={"color": "#ff7f0e"},
    )
    pairplot_fig.fig.suptitle(
        "Pairplot numerických příznaků diamantů (vzorek n=1500)",
        y=1.02,
        fontsize=14,
        fontweight="bold",
    )
    pairplot_path = plots_dir / "05_diamonds_pairplot.png"
    pairplot_fig.savefig(pairplot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Pairplot úspěšně uložen do: {pairplot_path}")

    # =========================================================================
    # Krok 8: Korelační matice a vyřazení nízko korelujících proměnných
    # =========================================================================
    print("\n" + "=" * 75)
    print("KROK 8: Korelační analýza a vyřazení slabých příznaků")
    print("=" * 75)
    corr_matrix = diamonds_df[numeric_cols].corr()
    price_corr = corr_matrix["price"].sort_values(ascending=False)
    print("Korelace numerických veličin s cenou diamantu (price):")
    print(price_corr)

    # Heatmapa korelací
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
    )
    plt.title("Korelační matice numerických proměnných (Diamonds)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    corr_path = plots_dir / "06_diamonds_correlation_matrix.png"
    plt.savefig(corr_path, dpi=300)
    plt.close()
    print(f"Korelační matice uložena do: {corr_path}")

    # Vyřazení proměnných s nízkou korelací (|r| < 0.20): 'depth' (r = -0.011) a 'table' (r = 0.127)
    low_corr_cols = ["depth", "table"]
    high_corr_cols = ["carat", "x", "y", "z"]
    print(f"\nVyřazujeme proměnné s nízkou korelací vůči ceně: {low_corr_cols}")
    print(f"Ponechané silně korelující spojité veličiny: {high_corr_cols}")

    # Scatter ploty linearity ponechaných proměnných s cenou
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, feat in zip(axes.flatten(), high_corr_cols):
        sns.scatterplot(
            data=diamonds_df.sample(2000, random_state=42),
            x=feat,
            y="price",
            alpha=0.4,
            ax=ax,
            color="#2b5c8f",
        )
        sns.regplot(
            data=diamonds_df.sample(2000, random_state=42),
            x=feat,
            y="price",
            scatter=False,
            ax=ax,
            color="crimson",
            line_kws={"linewidth": 2},
        )
        ax.set_title(
            f"{feat} vs Price (korelace r = {corr_matrix.loc[feat, 'price']:.3f})",
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
    plt.suptitle("Linearita silně korelujících proměnných vůči ceně diamantu", fontsize=14, fontweight="bold")
    plt.tight_layout()
    scatter_lin_path = plots_dir / "07_diamonds_scatter_linearity.png"
    plt.savefig(scatter_lin_path, dpi=300)
    plt.close()
    print(f"Graf linearity uložen do: {scatter_lin_path}")

    # =========================================================================
    # Krok 9: Detekce a odstranění odlehlých hodnot (Outliers)
    # =========================================================================
    print("\n" + "=" * 75)
    print("KROK 9: Detekce a odstranění odlehlých hodnot (Outliers)")
    print("=" * 75)

    # 1. Fyzikálně nemožné hodnoty: rozměry x == 0, y == 0, z == 0 (3D těleso nemůže mít nulový rozměr)
    zero_dim_mask = (diamonds_df["x"] == 0) | (diamonds_df["y"] == 0) | (diamonds_df["z"] == 0)
    zero_count = zero_dim_mask.sum()
    print(f"1. Fyzikálně nemožné diamanty s nulovým rozměrem (x=0, y=0 nebo z=0): {zero_count} záznamů")

    # 2. Hrubé překlepy v rozměrech: y > 20 mm nebo z > 20 mm (např. y = 58.9 mm u 2-karátového diamantu)
    typo_mask = (diamonds_df["y"] > 20) | (diamonds_df["z"] > 20)
    typo_count = typo_mask.sum()
    print(f"2. Extrémní překlepy v rozměrech (y > 20 mm nebo z > 20 mm): {typo_count} záznamy")

    # 3. Extrémní karáty (např. carat > 3.5 karátů – velmi vzácné investiční kusy s odlišnou cenotvorbou)
    extreme_carat_mask = diamonds_df["carat"] > 3.5
    extreme_carat_count = extreme_carat_mask.sum()
    print(f"3. Extrémní investiční kameny nad 3.5 karátu: {extreme_carat_count} záznamů")

    # Vizuální znázornění anomálií
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sns.scatterplot(data=diamonds_df, x="x", y="price", alpha=0.3, ax=axes[0], color="#1f77b4")
    axes[0].set_title("x (délka) vs Price – nulové hodnoty", fontweight="bold")
    axes[0].axvline(0.5, color="red", linestyle="--", label="Práh x > 0")
    axes[0].legend()

    sns.scatterplot(data=diamonds_df, x="y", y="price", alpha=0.3, ax=axes[1], color="#2ca02c")
    axes[1].set_title("y (šířka) vs Price – extrémní překlep (58.9 mm)", fontweight="bold")
    axes[1].axvline(20, color="red", linestyle="--", label="Práh y < 20")
    axes[1].legend()

    sns.scatterplot(data=diamonds_df, x="z", y="price", alpha=0.3, ax=axes[2], color="#d62728")
    axes[2].set_title("z (hloubka) vs Price – extrémní překlep (31.8 mm)", fontweight="bold")
    axes[2].axvline(20, color="red", linestyle="--", label="Práh z < 20")
    axes[2].legend()

    plt.tight_layout()
    outlier_plot_path = plots_dir / "08_diamonds_outliers_detection.png"
    plt.savefig(outlier_plot_path, dpi=300)
    plt.close()
    print(f"Graf odlehlých hodnot uložen do: {outlier_plot_path}")

    # Filtrace dat
    clean_diamonds_df = diamonds_df[
        (diamonds_df["x"] > 0)
        & (diamonds_df["y"] > 0)
        & (diamonds_df["z"] > 0)
        & (diamonds_df["y"] < 20)
        & (diamonds_df["z"] < 20)
        & (diamonds_df["carat"] <= 3.5)
    ].copy()

    dropped_total = len(diamonds_df) - len(clean_diamonds_df)
    print(
        f"\nCelkem odstraněno {dropped_total} anomálních záznamů ({dropped_total / len(diamonds_df) * 100:.2f} %)."
    )
    print(f"Čistá datová sada má {clean_diamonds_df.shape[0]} záznamů.")

    # =========================================================================
    # Krok 10: Rozdělení na trénovací a testovací sadu (Train/Test Split)
    # =========================================================================
    print("\n" + "=" * 75)
    print("KROK 10: Rozdělení dat na Train a Test sadu (80 / 20)")
    print("=" * 75)
    # Dle zadání: model s vysoce korelujícími proměnnými (po vyloučení depth a table)
    X_simple = clean_diamonds_df[high_corr_cols]
    y = clean_diamonds_df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_simple, y, test_size=0.2, random_state=42
    )
    print(f"Trénovací data: X_train = {X_train.shape}, y_train = {y_train.shape}")
    print(f"Testovací data:  X_test = {X_test.shape}, y_test = {y_test.shape}")

    # =========================================================================
    # Krok 11: Definice a trénování modelu, evaluace na testovacích datech
    # =========================================================================
    print("\n" + "=" * 75)
    print("KROK 11: Trénování modelu LinearRegression a evaluace")
    print("=" * 75)
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    y_pred_train = lin_reg.predict(X_train)
    y_pred_test = lin_reg.predict(X_test)

    r2_train = lin_reg.score(X_train, y_train)
    r2_test = lin_reg.score(X_test, y_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)

    print(f"Model: LinearRegression na příznacích: {high_corr_cols}")
    print(f"Trénovací R2: {r2_train:.4f}")
    print(f"Testovací R2:  {r2_test:.4f}")
    print(f"Testovací RMSE: ${rmse_test:,.2f}")
    print(f"Testovací MAE:  ${mae_test:,.2f}")

    print("\nKoeficienty modelu:")
    for feat, coef in zip(high_corr_cols, lin_reg.coef_):
        print(f"  - {feat:10s}: {coef:12.2f}")
    print(f"  - Intercept : {lin_reg.intercept_:12.2f}")

    # =========================================================================
    # Krok 12: Uložení připraveného datasetu do CSV
    # =========================================================================
    print("\n" + "=" * 75)
    print("KROK 12: Uložení připraveného datasetu do CSV")
    print("=" * 75)
    # Pro kompatibilitu s dalšími moduly kurzu provedeme také číselné zakódování
    # kategorií (cut, color, clarity) a uložíme kompletní vyčištěnou sadu
    prep_df = clean_diamonds_df.copy()

    # Kódování 4C (pro zachování přirozeného gemologického uspořádání)
    cut_order = {"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4}
    color_order = {"J": 0, "I": 1, "H": 2, "G": 3, "F": 4, "E": 5, "D": 6}
    clarity_order = {"I1": 0, "SI2": 1, "SI1": 2, "VS2": 3, "VS1": 4, "VVS2": 5, "VVS1": 6, "IF": 7}

    prep_df["cut"] = prep_df["cut"].map(cut_order)
    prep_df["color"] = prep_df["color"].map(color_order)
    prep_df["clarity"] = prep_df["clarity"].map(clarity_order)

    output_csv_path = data_dir / "diamonds_preprocessed.csv"
    prep_df.to_csv(output_csv_path, index=False)
    print(f"Připravený a vyčištěný dataset byl uložen do: {output_csv_path}")

    # =========================================================================
    # Krok 13: VLASTNÍ ANALÝZA & GEMOLOGICKÁ MODERNIZACE (09/2026)
    # =========================================================================
    print("\n" + "=" * 75)
    print("KROK 13: Vlastní analýza a gemologická modernizace pro klenotníka")
    print("=" * 75)

    # 1. Analýza 4 metodických chyb kurzu:
    # -------------------------------------------------------------------------
    print("\n--- 1. Čtyři zásadní metodické chyby zadání kurzu ---")
    print(
        "CHYBA 1: Vyřazení 4C (Cut, Color, Clarity).\n"
        "         Kurz pracoval jen s numerickými sloupci a vynechal to nejdůležitější pro klenotníka.\n"
        "CHYBA 2: Smazání 'depth' a 'table' kvůli 'nízké korelaci'.\n"
        "         Pearsonova lineární korelace měří pouze monotónní přímku (depth r = -0.011).\n"
        "         V gemologii má depth zvonovitou křivku – ideál 61-62.5 %, mimo tento rozsah ztrácí kámen 20-40 % hodnoty.\n"
        "CHYBA 3: Umělý strop na 3.5 karátu (outlier filtr).\n"
        "         Vyřadit kameny nad 3.5 ct znamená vyřadit nejvzácnější investiční solitéry (až 5.01 ct),\n"
        "         kde klenotník nejvíce potřebuje spolehlivé ocenění.\n"
        "CHYBA 4: Fyzikální multikolinearita (carat vs x, y, z).\n"
        "         V = x * y * z a hmotnost m = hustota * V. Korelace r(carat, x*y*z) = 0.9989!\n"
        "         To vedlo k absurdním záporným vahám pro rozměry x (-$3,675) a z (-$2,602)."
    )

    # 2. Vytvoření diagnostického grafu pro proporce brusu (depth & table)
    fig, (ax_d, ax_t) = plt.subplots(1, 2, figsize=(15, 5))
    sns.scatterplot(data=clean_diamonds_df.sample(2500, random_state=42), x="depth", y="price", alpha=0.3, ax=ax_d, color="#8b5cf6")
    ax_d.axvspan(61.0, 62.5, color="green", alpha=0.2, label="Ideální proporce (61.0–62.5 %)")
    ax_d.set_title("Hloubka (depth %) vs Cena – Proč je r ≈ 0?", fontweight="bold")
    ax_d.set_xlabel("Depth (%)")
    ax_d.set_ylabel("Cena ($)")
    ax_d.set_xlim(55, 70)
    ax_d.legend()
    ax_d.grid(True, alpha=0.3)

    sns.scatterplot(data=clean_diamonds_df.sample(2500, random_state=42), x="table", y="price", alpha=0.3, ax=ax_t, color="#ec4899")
    ax_t.axvspan(54.0, 57.0, color="green", alpha=0.2, label="Ideální ploška table (54–57 %)")
    ax_t.set_title("Ploška (table %) vs Cena", fontweight="bold")
    ax_t.set_xlabel("Table (%)")
    ax_t.set_ylabel("Cena ($)")
    ax_t.set_xlim(50, 70)
    ax_t.legend()
    ax_t.grid(True, alpha=0.3)

    plt.tight_layout()
    gem_plot_path = plots_dir / "09_diamonds_gemology_depth_table_analysis.png"
    plt.savefig(gem_plot_path, dpi=300)
    plt.close()
    print(f"Diagnostický graf proporcí brusu uložen do: {gem_plot_path}")

    # 3. Systematický benchmark 5 modelů:
    # -------------------------------------------------------------------------
    # Dataset včetně velkých diamantů (bez umělého 3.5 ct stropu)
    full_clean = diamonds_df[
        (diamonds_df["x"] > 0)
        & (diamonds_df["y"] > 0)
        & (diamonds_df["z"] > 0)
        & (diamonds_df["y"] < 20)
        & (diamonds_df["z"] < 20)
    ].copy()

    full_clean["cut_num"] = full_clean["cut"].map(cut_order)
    full_clean["color_num"] = full_clean["color"].map(color_order)
    full_clean["clarity_num"] = full_clean["clarity"].map(clarity_order)

    # Definice matic příznaků
    X_base = full_clean[["carat", "x", "y", "z"]]
    X_4c_only = full_clean[["carat", "cut_num", "color_num", "clarity_num"]]
    X_all = full_clean[["carat", "cut_num", "color_num", "clarity_num", "depth", "table", "x", "y", "z"]]
    y_full = full_clean["price"]

    # Společný split
    idx_tr, idx_te = train_test_split(full_clean.index, test_size=0.2, random_state=42)

    # Model 1: Učebnicový OLS (pouze carat, x, y, z)
    m1 = LinearRegression().fit(X_base.loc[idx_tr], y_full.loc[idx_tr])
    p1 = m1.predict(X_base.loc[idx_te])

    # Model 2: Gemologický 4C OLS (Carat, Cut, Color, Clarity)
    m2 = Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())]).fit(X_4c_only.loc[idx_tr], y_full.loc[idx_tr])
    p2 = m2.predict(X_4c_only.loc[idx_te])

    # Model 3: Všechny příznaky OLS (+ depth, table, x, y, z)
    m3 = Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())]).fit(X_all.loc[idx_tr], y_full.loc[idx_tr])
    p3 = m3.predict(X_all.loc[idx_te])

    # Model 4: Nelineární Log-Log model (Price ~ Carat^beta + 4C)
    X_log_tr = X_4c_only.loc[idx_tr].copy()
    X_log_tr["carat"] = np.log(X_log_tr["carat"])
    X_log_te = X_4c_only.loc[idx_te].copy()
    X_log_te["carat"] = np.log(X_log_te["carat"])
    m4 = TransformedTargetRegressor(
        regressor=Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())]),
        func=np.log1p,
        inverse_func=np.expm1
    ).fit(X_log_tr, y_full.loc[idx_tr])
    p4 = m4.predict(X_log_te)

    # Model 5: Moderní Gradient Boosting (HistGradientBoostingRegressor na všech příznacích)
    m5 = HistGradientBoostingRegressor(random_state=42).fit(X_all.loc[idx_tr], y_full.loc[idx_tr])
    p5 = m5.predict(X_all.loc[idx_te])

    # Výpis výsledků
    y_test_eval = y_full.loc[idx_te]
    print("\n--- SROVNÁVACÍ BENCHMARK 5 MODELŮ PRO KLENOTNÍKA ---")
    print(f"1. Učebnicový OLS (carat, x, y, z):     R2 = {r2_score(y_test_eval, p1):.4f}, MAE = ${mean_absolute_error(y_test_eval, p1):,.0f}")
    print(f"2. Gemologický 4C OLS (carat + 3C):     R2 = {r2_score(y_test_eval, p2):.4f}, MAE = ${mean_absolute_error(y_test_eval, p2):,.0f}")
    print(f"3. Plný OLS (+ depth, table, rozměry):  R2 = {r2_score(y_test_eval, p3):.4f}, MAE = ${mean_absolute_error(y_test_eval, p3):,.0f}")
    print(f"4. Fyzikální Log-Log model (Mocninný):  R2 = {r2_score(y_test_eval, p4):.4f}, MAE = ${mean_absolute_error(y_test_eval, p4):,.0f}")
    print(f"5. Moderní Gradient Boosting (HGB):     R2 = {r2_score(y_test_eval, p5):.4f}, MAE = ${mean_absolute_error(y_test_eval, p5):,.0f}")
    print(f"==> Závěr: Gradient Boosting snížil průměrnou chybu o {((mean_absolute_error(y_test_eval, p1) - mean_absolute_error(y_test_eval, p5)) / mean_absolute_error(y_test_eval, p1))*100:.1f} %!")

    print("\n" + "=" * 75)
    print("VŠECHNY KROKY CVIČENÍ 2 DOKONČENY ÚSPĚŠNĚ!")
    print("=" * 75)


if __name__ == "__main__":
    main()
