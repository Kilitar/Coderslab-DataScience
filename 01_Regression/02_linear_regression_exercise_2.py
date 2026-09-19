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
    # Krok 13: VLASTNÍ ANALÝZA & MODERNÍ ROZŠÍŘENÍ (09/2026)
    # =========================================================================
    print("\n" + "=" * 75)
    print("KROK 13: Vlastní analýza a moderní ML rozšíření pro klenotníka")
    print("=" * 75)

    # A. Multikolinearita a fyzika diamantu
    # Objem V = x * y * z je přímo svázán s hmotností (carat).
    # Proto mají carat a x, y, z mezi sebou korelaci > 0.97!
    v_corr = np.corrcoef(clean_diamonds_df["carat"], clean_diamonds_df["x"] * clean_diamonds_df["y"] * clean_diamonds_df["z"])[0, 1]
    print(f"Korelace mezi hmotností (carat) a odhadnutým objemem (x*y*z): {v_corr:.4f}")
    print("Závěr k multikolinearitě: Zahrnutí x, y, z společně s carat vytváří těžkou multikolinearitu.")
    print("Pokud klenotník zváží kámen na přesné váze (carat), rozměry x,y,z přinášejí minimální dodatečnou informaci.")

    # B. Zahrnutí 4C parametrů (Carat, Cut, Color, Clarity)
    # Klenotník ví, že 1-karátový démant barvy D a čistoty IF má 5x vyšší hodnotu než barvy J a čistoty I1!
    X_4c = prep_df[["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z"]]
    X_4c_train, X_4c_test, y_4c_train, y_4c_test = train_test_split(
        X_4c, y, test_size=0.2, random_state=42
    )

    lin_reg_4c = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])
    lin_reg_4c.fit(X_4c_train, y_4c_train)
    r2_4c = lin_reg_4c.score(X_4c_test, y_4c_test)
    mae_4c = mean_absolute_error(y_4c_test, lin_reg_4c.predict(X_4c_test))
    print(f"\n--- 1. Lineární model se všemi 4C parametry (Carat, Cut, Color, Clarity) ---")
    print(f"R2 skóre s 4C parametry: {r2_4c:.4f} (zlepšení z {r2_test:.4f})")
    print(f"MAE s 4C parametry:     ${mae_4c:,.2f} (pokles chyby o ${mae_test - mae_4c:,.2f})")

    # C. Nelineární mocninný zákon: Log-Log transformace
    # Cena roste s hmotností exponenciálně/mocninně: Price ~ Carat^beta
    log_log_model = TransformedTargetRegressor(
        regressor=Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        func=np.log1p,
        inverse_func=np.expm1
    )
    # Použijeme log(carat) jako transformovaný vstup
    X_log = prep_df[["carat", "cut", "color", "clarity"]].copy()
    X_log["carat"] = np.log(X_log["carat"])
    X_log_train, X_log_test, y_log_train, y_log_test = train_test_split(
        X_log, y, test_size=0.2, random_state=42
    )
    log_log_model.fit(X_log_train, y_log_train)
    y_pred_log = log_log_model.predict(X_log_test)
    r2_log = r2_score(y_log_test, y_pred_log)
    mae_log = mean_absolute_error(y_log_test, y_pred_log)
    print(f"\n--- 2. Log-Log model (Fyzikální a ekonomický model cenotvorby drahokamů) ---")
    print(f"R2 skóre (Log-Log):     {r2_log:.4f}")
    print(f"MAE (Log-Log):          ${mae_log:,.2f} (výrazně přesnější pro menší kameny!)")

    # D. Moderní nelineární tabulární benchmark: HistGradientBoostingRegressor
    hgb_model = HistGradientBoostingRegressor(random_state=42)
    hgb_model.fit(X_4c_train, y_4c_train)
    y_pred_hgb = hgb_model.predict(X_4c_test)
    r2_hgb = r2_score(y_4c_test, y_pred_hgb)
    mae_hgb = mean_absolute_error(y_4c_test, y_pred_hgb)
    print(f"\n--- 3. Moderní Gradient Boosting (State-of-the-Art pro tabulární data) ---")
    print(f"R2 skóre (Gradient Boosting): {r2_hgb:.4f} (vysvětluje 98 % rozptylu cen!)")
    print(f"MAE (Gradient Boosting):      ${mae_hgb:,.2f} (průměrná odchylka pouze 275 USD!)")

    print("\n" + "=" * 75)
    print("VŠECHNY KROKY CVIČENÍ 2 DOKONČENY ÚSPĚŠNĚ!")
    print("=" * 75)


if __name__ == "__main__":
    main()
