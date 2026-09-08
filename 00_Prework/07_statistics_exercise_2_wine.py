"""
Statistics Exercise 2 - Wine Dataset Analysis
Machine Learning Course - CodersLab

Úkoly:
1. Načtení datasetu Wine ze scikit-learn do pandas DataFrame.
2. Výpočet průměru (mean) a mediánu (median) pro všechny proměnné.
3. Výpočet minima, maxima a směrodatné odchylky pro proměnné 'alcohol' a 'malic_acid'.
4. Výpočet 1. a 3. kvartilu (Q1 = 0.25, Q3 = 0.75) pro proměnné 'total_phenols' a 'proline'.
5. Výpočet korelační matice a její vizualizace pomocí Seaborn a Plotly s čitelnými popisky.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.datasets import load_wine

# Vytvoření složky pro výstupy grafů
script_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(script_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# ==============================================================================
# 0. Načtení dat
# ==============================================================================
wine_data = load_wine()
wine_df = pd.DataFrame(data=wine_data.data, columns=wine_data.feature_names)
print(f"Dataset Wine načten. Rozměry: {wine_df.shape[0]} vzorků, {wine_df.shape[1]} vlastností.\n")

# ==============================================================================
# Úkol 1: Průměr a medián pro každou proměnnou
# ==============================================================================
print("=" * 70)
print("ÚKOL 1: Průměr a medián pro všechny proměnné")
print("=" * 70)
mean_median_df = pd.DataFrame({
    'Mean': wine_df.mean(),
    'Median': wine_df.median()
}).round(3)
print(mean_median_df)

# ==============================================================================
# Úkol 2: Min, Max a Směrodatná odchylka pro alcohol a malic_acid
# ==============================================================================
print("\n" + "=" * 70)
print("ÚKOL 2: Min, Max a Std pro proměnné 'alcohol' a 'malic_acid'")
print("=" * 70)
alc_malic_stats = wine_df[['alcohol', 'malic_acid']].agg(['min', 'max', 'std']).T
alc_malic_stats.columns = ['Minimum', 'Maximum', 'Standard Deviation']
print(alc_malic_stats.round(3))

# ==============================================================================
# Úkol 3: 1. a 3. kvartil pro total_phenols a proline
# ==============================================================================
print("\n" + "=" * 70)
print("ÚKOL 3: 1. a 3. kvartil (Q1 a Q3) pro 'total_phenols' a 'proline'")
print("=" * 70)
quantiles_df = wine_df[['total_phenols', 'proline']].quantile([0.25, 0.75]).T
quantiles_df.columns = ['1st Quartile (Q1 / 25%)', '3rd Quartile (Q3 / 75%)']
print(quantiles_df.round(3))

# ==============================================================================
# Úkol 4: Korelační matice a vizualizace (Seaborn & Plotly)
# ==============================================================================
print("\n" + "=" * 70)
print("ÚKOL 4: Korelační matice a generování grafů")
print("=" * 70)
corr_matrix = wine_df.corr().round(2)
print("Korelační matice (náhled):")
print(corr_matrix.iloc[:5, :5])

# --- 4a. Vizualizace pomocí Seaborn (uložení do PNG) ---
plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8},
    annot_kws={'size': 9}
)
plt.title("Correlation Matrix of Wine Characteristics", fontsize=15, fontweight="bold", pad=15)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()

seaborn_path = os.path.join(plots_dir, "04_wine_correlation_seaborn.png")
plt.savefig(seaborn_path, dpi=200)
plt.close()
print(f"Seaborn graf uložen do: {seaborn_path}")

# --- 4b. Interaktivní vizualizace pomocí Plotly (uložení do HTML) ---
fig = px.imshow(
    corr_matrix,
    text_auto=".2f",
    aspect="auto",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    title="Interactive Correlation Matrix - Wine Dataset",
    labels=dict(x="Features", y="Features", color="Correlation")
)
fig.update_layout(
    width=950,
    height=850,
    xaxis_tickangle=-45,
    template="plotly_white",
    title_font_size=16
)

plotly_path = os.path.join(plots_dir, "04_wine_correlation_plotly.html")
fig.write_html(plotly_path)
print(f"Plotly interaktivní graf uložen do: {plotly_path}")
