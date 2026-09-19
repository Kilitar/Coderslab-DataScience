"""
Linear Regression - Exercise 1
Dataset: Real estate prices in King County, USA (kc_house_data.csv)

ZADÁNÍ / ASSIGNMENT:
------------------------------------------------
1. Into the execution environment in the notebook or on Google drive, transfer the data file
   (kc_house_data.csv) about real estate in King County, USA (the same one used in the presentation
   on the implementation of the linear regression model in Python).
2. Import all necessary methods from libraries (or modules from these libraries).
3. Load the data frame file into the Pandas library. Assign the data frame to a variable named housing_df.
4. Display the first dozen or so data records to see how the data looks.
5. Remove unnecessary columns from the dataset.
6. Using the methods you have learned, verify the presence of missing values in the dataset.
7. Generate and visualize the correlation matrix. You can verify the linearity of the correlation
   of the selected variables with the independent variable (price) graphically.
8. Using the selected graphs, check the presence of outliers. Use scatter plots of the selected variables
   and the dependent variable price to accurately identify observations that have outliers. Use the method
   of your choice (e.g., reading from the graph and filtering, IQR method) to remove outliers.
9. Divide the dataset into training and test datasets.
10. Define and train the model using the training data. Make predictions on the test data.
11. Using the .score method, check the score of the metric.
12. Use standardization/normalization of independent variables and check whether it positively affected
    the quality of the model. Perform the standardization on the variable X and then train the model,
    make a prediction on the test set and calculate the R2 metric and compare with the previously calculated value.
13. Save the prepared dataset to .csv file. You will use it to build more models during this day.
------------------------------------------------
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    # -------------------------------------------------------------------------
    # Nastavení cest / Paths setup
    # -------------------------------------------------------------------------
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    plots_dir = base_dir / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Nalezení vstupního souboru
    raw_data_path = data_dir / "kc_house_data.csv"
    if not raw_data_path.exists():
        fallback_path = (
            base_dir.parent
            / "data"
            / "MAL_downloadable materials_session 1"
            / "Day 1"
            / "kc_house_data.csv"
        )
        if fallback_path.exists():
            import shutil

            shutil.copy(fallback_path, raw_data_path)
            print(f"Data zkopírována z fallback cesty do: {raw_data_path}")

    # =========================================================================
    # Krok 1 & 2: Načtení dat do Pandas (housing_df)
    # =========================================================================
    print("=" * 70)
    print("KROK 1 & 2: Načtení datové sady King County")
    print("=" * 70)
    housing_df = pd.read_csv(raw_data_path)
    print(f"Původní rozměry dat: {housing_df.shape} (řádky, sloupce)")

    # =========================================================================
    # Krok 3: Zobrazení prvních cca 15 záznamů
    # =========================================================================
    print("\n" + "=" * 70)
    print("KROK 3: Náhled prvních 15 záznamů (housing_df.head(15))")
    print("=" * 70)
    print(housing_df.head(15).to_string())

    # =========================================================================
    # Krok 4: Odstranění nepotřebných sloupců
    # =========================================================================
    print("\n" + "=" * 70)
    print("KROK 4: Odstranění neprediktivních sloupců ('id', 'date')")
    print("=" * 70)
    cols_to_drop = [col for col in ["id", "date"] if col in housing_df.columns]
    housing_df = housing_df.drop(columns=cols_to_drop)
    print(f"Odstraněny sloupce: {cols_to_drop}")
    print(f"Nové rozměry dat: {housing_df.shape}")

    # =========================================================================
    # Krok 5: Ověření chybějících hodnot
    # =========================================================================
    print("\n" + "=" * 70)
    print("KROK 5: Kontrola chybějících hodnot (Missing Values)")
    print("=" * 70)
    null_counts = housing_df.isnull().sum()
    total_missing = null_counts.sum()
    print(f"Celkový počet chybějících hodnot: {total_missing}")
    if total_missing > 0:
        print(null_counts[null_counts > 0])
    else:
        print(
            "Datová sada neobsahuje žádné chybějící (null) hodnoty – je kompletní."
        )

    # =========================================================================
    # Krok 6: Korelační matice a vizualizace linearity s 'price'
    # =========================================================================
    print("\n" + "=" * 70)
    print("KROK 6: Analýza korelací s cílovou proměnnou 'price'")
    print("=" * 70)
    corr_matrix = housing_df.corr()
    price_corr = corr_matrix["price"].sort_values(ascending=False)
    print("Korelace příznaků s cenou (seřazeno sestupně):")
    print(price_corr)

    # Heatmapa korelací
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        vmin=-1,
        vmax=1,
    )
    plt.title(
        "Korelační matice příznaků – King County Housing",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    heatmap_path = plots_dir / "01_correlation_matrix.png"
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"Graf korelační matice uložen do: {heatmap_path}")

    # Scatter ploty top 4 lineárně korelujících proměnných s cenou
    top_features = ["sqft_living", "grade", "sqft_above", "sqft_living15"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, feat in zip(axes.flatten(), top_features):
        sns.scatterplot(
            data=housing_df,
            x=feat,
            y="price",
            alpha=0.3,
            ax=ax,
            color="#1f77b4",
        )
        # Proložení trendové přímky
        sns.regplot(
            data=housing_df,
            x=feat,
            y="price",
            scatter=False,
            ax=ax,
            color="crimson",
            line_kws={"linewidth": 2},
        )
        ax.set_title(
            f"{feat} vs Price (korelace = {corr_matrix.loc[feat, 'price']:.2f})",
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
    plt.suptitle(
        "Scatter Plots: Linearita vztahů s cenou nemovitosti",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout()
    scatter_path = plots_dir / "02_scatter_features_vs_price.png"
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    print(f"Graf linearity příznaků uložen do: {scatter_path}")

    # =========================================================================
    # Krok 7: Detekce a odstranění odlehlých hodnot (Outliers)
    # =========================================================================
    print("\n" + "=" * 70)
    print("KROK 7: Detekce odlehlých hodnot (Outliers)")
    print("=" * 70)

    # Inspekce extrémních anomálií:
    # 1. Dům s 33 ložnicemi (známý překlep v datasetu – plocha jen 1620 sqft!)
    outlier_bedrooms = housing_df[housing_df["bedrooms"] >= 15]
    print(
        "Extrémní anomálie ložnic (např. 33 ložnic při ploše 1620 sqft):",
        len(outlier_bedrooms),
    )

    # 2. Obrovské domy nad 10 000 sqft
    outlier_sqft = housing_df[housing_df["sqft_living"] > 10000]
    print(
        "Extrémní domy nad 10 000 sqft (mega-villy):",
        len(outlier_sqft),
    )

    # Vizuální znázornění odlehlých hodnot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.scatterplot(
        data=housing_df,
        x="bedrooms",
        y="price",
        alpha=0.4,
        ax=ax1,
        color="#2ca02c",
    )
    ax1.set_title("Před filtrováním: bedrooms vs price", fontweight="bold")
    ax1.axvline(10.5, color="red", linestyle="--", label="Práh filtru (>10)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    sns.scatterplot(
        data=housing_df,
        x="sqft_living",
        y="price",
        alpha=0.4,
        ax=ax2,
        color="#d62728",
    )
    ax2.set_title("Před filtrováním: sqft_living vs price", fontweight="bold")
    ax2.axvline(
        10000, color="darkred", linestyle="--", label="Práh filtru (>10 000)"
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    outlier_plot_path = plots_dir / "03_outliers_detection_bedrooms_sqft.png"
    plt.savefig(outlier_plot_path, dpi=300)
    plt.close()
    print(f"Graf odlehlých hodnot uložen do: {outlier_plot_path}")

    # Odfiltrování zjevných datových anomálií:
    # - Ložnice <= 10 (odstranění překlepu 33 ložnic a nereálných extrémů)
    # - sqft_living <= 10 000 (extrémní luxusní residence s jinou cenovou dynamikou)
    # - cena do horního 99.8% percentilu (extrémní multimilionové úlety, které deformují OLS SSE)
    price_cap = housing_df["price"].quantile(0.998)
    initial_len = len(housing_df)
    clean_df = housing_df[
        (housing_df["bedrooms"] <= 10)
        & (housing_df["bedrooms"] > 0)
        & (housing_df["sqft_living"] <= 10000)
        & (housing_df["price"] <= price_cap)
    ].copy()
    dropped_count = initial_len - len(clean_df)
    print(
        f"Odstraněno {dropped_count} odlehlých záznamů ({dropped_count/initial_len*100:.2f} % dat)."
    )
    print(f"Rozměry vyčištěného datasetu: {clean_df.shape}")

    # =========================================================================
    # Krok 8: Rozdělení dat na trénovací a testovací sadu (Train/Test Split)
    # =========================================================================
    print("\n" + "=" * 70)
    print("KROK 8: Rozdělení dat na Train a Test (80 / 20)")
    print("=" * 70)
    X = clean_df.drop(columns=["price"])
    y = clean_df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Trénovací sada X: {X_train.shape}, y: {y_train.shape}")
    print(f"Testovací sada X: {X_test.shape}, y: {y_test.shape}")

    # =========================================================================
    # Krok 9 & 10: Definice a trénování modelu, evaluace na neskálovaných datech
    # =========================================================================
    print("\n" + "=" * 70)
    print("KROK 9 & 10: Trénování základního modelu LinearRegression")
    print("=" * 70)
    lin_reg_raw = LinearRegression()
    lin_reg_raw.fit(X_train, y_train)

    y_pred_train_raw = lin_reg_raw.predict(X_train)
    y_pred_test_raw = lin_reg_raw.predict(X_test)

    r2_train_raw = lin_reg_raw.score(X_train, y_train)
    r2_test_raw = lin_reg_raw.score(X_test, y_test)
    rmse_test_raw = np.sqrt(mean_squared_error(y_test, y_pred_test_raw))
    mae_test_raw = mean_absolute_error(y_test, y_pred_test_raw)

    print(f"Trénovací R2 skóre (Raw): {r2_train_raw:.4f}")
    print(f"Testovací R2 skóre (Raw):  {r2_test_raw:.4f}")
    print(f"Testovací RMSE (Raw):     ${rmse_test_raw:,.2f}")
    print(f"Testovací MAE (Raw):      ${mae_test_raw:,.2f}")

    # =========================================================================
    # Krok 11: Standardizace proměnných X a porovnání metrik
    # =========================================================================
    print("\n" + "=" * 70)
    print("KROK 11: Standardizace příznaků X (StandardScaler)")
    print("=" * 70)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lin_reg_scaled = LinearRegression()
    lin_reg_scaled.fit(X_train_scaled, y_train)

    y_pred_test_scaled = lin_reg_scaled.predict(X_test_scaled)
    r2_test_scaled = lin_reg_scaled.score(X_test_scaled, y_test)
    rmse_test_scaled = np.sqrt(mean_squared_error(y_test, y_pred_test_scaled))
    mae_test_scaled = mean_absolute_error(y_test, y_pred_test_scaled)

    print(f"Testovací R2 skóre po standardizaci: {r2_test_scaled:.4f}")
    print(f"Rozdíl R2 (Scaled vs Raw):          {r2_test_scaled - r2_test_raw:.2e}")
    print(f"Testovací RMSE (Scaled):            ${rmse_test_scaled:,.2f}")
    print(f"Testovací MAE (Scaled):             ${mae_test_scaled:,.2f}")

    print("\n--- TEORETICKÝ DŮVOD, PROČ SE R2 U OLS NEZMĚNILO ---")
    print(
        "Standardizace X je lineární (afinní) transformace: Z = (X - mu) / sigma.\n"
        "V analytickém OLS řešení se váhy modelu pouze algebraicky přeškálují (w_scaled = w_raw * sigma),\n"
        "ale výsledný predikovaný vektor y_pred i celková plocha chyby zůstávají exaktně identické!\n"
        "Standardizace je však klíčová pro:\n"
        "1. Porovnatelnost důležitosti koeficientů (Feature Importance).\n"
        "2. Regularizované modely (Ridge, Lasso), které penalizují velikost vah.\n"
        "3. Gradientní optimalizátory (rychlost konvergence)."
    )

    # Porovnání interpretace vah před a po standardizaci
    coef_comparison = pd.DataFrame(
        {
            "Feature": X.columns,
            "Raw_Coef": lin_reg_raw.coef_,
            "Standardized_Coef": lin_reg_scaled.coef_,
            "Abs_Std_Impact": np.abs(lin_reg_scaled.coef_),
        }
    ).sort_values(by="Abs_Std_Impact", ascending=False)

    print("\nSrovnání koeficientů (Důležitost příznaků dle škálovaných vah):")
    print(coef_comparison.to_string(index=False))

    # =========================================================================
    # Krok 12: Uložení připraveného datasetu do CSV
    # =========================================================================
    print("\n" + "=" * 70)
    print("KROK 12: Uložení připraveného datasetu do CSV")
    print("=" * 70)
    output_csv_path = data_dir / "kc_house_data_preprocessed.csv"
    clean_df.to_csv(output_csv_path, index=False)
    print(f"Vyčištěný a připravený dataset byl uložen do: {output_csv_path}")

    # =========================================================================
    # Krok 13: VLASTNÍ ANALÝZA & MODERNÍ ROZŠÍŘENÍ (09/2026)
    # =========================================================================
    print("\n" + "=" * 70)
    print("KROK 13: Vlastní analýza a moderní ML rozšíření")
    print("=" * 70)

    # A. Diagnostika reziduí (Visual check normality & homoscedasticity)
    residuals = y_test - y_pred_test_scaled

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.scatterplot(
        x=y_pred_test_scaled,
        y=residuals,
        alpha=0.3,
        color="#9467bd",
        ax=ax1,
    )
    ax1.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax1.set_title("Rezidua vs Predikované hodnoty (Homoskedasticita)", fontweight="bold")
    ax1.set_xlabel("Predikovaná cena ($)")
    ax1.set_ylabel("Rezidua (y_true - y_pred)")
    ax1.grid(True, alpha=0.3)

    sns.histplot(residuals, kde=True, ax=ax2, color="#8c564b", bins=50)
    ax2.set_title("Distribuce reziduí (Normalita chyb)", fontweight="bold")
    ax2.set_xlabel("Chyba predikce ($)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    residuals_plot_path = plots_dir / "04_actual_vs_predicted_residuals.png"
    plt.savefig(residuals_plot_path, dpi=300)
    plt.close()
    print(f"Diagnostický graf reziduí uložen do: {residuals_plot_path}")

    # B. Moderní transformace cílové proměnné: Log-Transform (TransformedTargetRegressor)
    # Ceny nemovitostí mají pravostranně sešikmenou distribuci (log-normal).
    # Modelování log(price) dramaticky stabilizuje rozptyl a zvyšuje kvalitu.
    log_model = TransformedTargetRegressor(
        regressor=Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression()),
            ]
        ),
        func=np.log1p,
        inverse_func=np.expm1,
    )
    log_model.fit(X_train, y_train)
    y_pred_log = log_model.predict(X_test)
    r2_log = r2_score(y_test, y_pred_log)
    rmse_log = np.sqrt(mean_squared_error(y_test, y_pred_log))
    mae_log = mean_absolute_error(y_test, y_pred_log)

    print("\n--- 1. Výsledek s Log-Transformací cíle (TransformedTargetRegressor) ---")
    print(f"R2 skóre (Log-Transform): {r2_log:.4f} (zlepšení o {r2_log - r2_test_scaled:+.4f})")
    print(f"RMSE (Log-Transform):     ${rmse_log:,.2f}")
    print(f"MAE (Log-Transform):      ${mae_log:,.2f} (pokles průměrné absolutní chyby!)")

    # C. Srovnání s moderním tabulárním gradient boostingem (HistGradientBoostingRegressor)
    hgb_model = HistGradientBoostingRegressor(random_state=42)
    hgb_model.fit(X_train, y_train)
    y_pred_hgb = hgb_model.predict(X_test)
    r2_hgb = r2_score(y_test, y_pred_hgb)
    rmse_hgb = np.sqrt(mean_squared_error(y_test, y_pred_hgb))
    mae_hgb = mean_absolute_error(y_test, y_pred_hgb)

    print("\n--- 2. Srovnání s moderním Gradient Boostingem (HistGradientBoostingRegressor) ---")
    print(f"R2 skóre (Gradient Boosting): {r2_hgb:.4f} (oproti OLS R2 {r2_test_scaled:.4f})")
    print(f"RMSE (Gradient Boosting):     ${rmse_hgb:,.2f}")
    print(f"MAE (Gradient Boosting):      ${mae_hgb:,.2f}")

    print("\n" + "=" * 70)
    print("VŠECHNY KROKY CVIČENÍ 1 DOKONČENY ÚSPĚŠNĚ!")
    print("=" * 70)


if __name__ == "__main__":
    main()
