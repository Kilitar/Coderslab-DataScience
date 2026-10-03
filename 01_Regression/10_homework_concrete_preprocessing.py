"""
Homework 1: Příprava dat pro regresní modely – Pevnost betonu (Concrete Compressive Strength)
=============================================================================================
Tento skript řeší zadání úvodní části domácího úkolu pro regresní modely:
1. Načtení datového souboru concrete_data.csv do Pandas DataFrame.
2. Zobrazení prvních 10 řádků a kontrola struktury.
3. Kontrola chybějících hodnot (NaN/null), datových typů a duplicit (odstranění 25 duplicit).
4. Analýza rozdělení jednotlivých proměnných (histogramy, šikmost, základní statistiky).
5. Analýza závislostí vůči cílové proměnné csMPa (scatter plots, trendy).
6. Korelační matice mezi všemi proměnnými (multikolinearita a vliv na pevnost).
7. Škálování nezávislých proměnných: Standardizace (StandardScaler) vs. Normalizace (MinMaxScaler)
   a uložení výsledného datasetu concrete_data_preprocessed.csv.
8. Export předpočtených výsledků do JSON cache a uložení diagnostických PNG grafů.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Nastavení cest
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV_PATH = DATA_DIR / "concrete_data.csv"
PREP_CSV_PATH = DATA_DIR / "concrete_data_preprocessed.csv"
PRECOMPUTED_JSON_PATH = DATA_DIR / "concrete_preprocessing_precomputed.json"

# Názvy sloupců dle zadání
COL_NAMES = [
    "cement",
    "slag",
    "flyash",
    "water",
    "superplasticizer",
    "coarseaggregate",
    "fineaggregate",
    "age",
    "csMPa"
]

FEATURE_COLS = COL_NAMES[:-1]
TARGET_COL = COL_NAMES[-1]

COL_DESCRIPTIONS = {
    "cement": "Obsah cementu (kg/m³)",
    "slag": "Obsah vysokopecní strusky (kg/m³)",
    "flyash": "Obsah popílku (kg/m³)",
    "water": "Obsah záměsové vody (kg/m³)",
    "superplasticizer": "Obsah superplastifikátoru (kg/m³)",
    "coarseaggregate": "Obsah hrubého kameniva / štěrku (kg/m³)",
    "fineaggregate": "Obsah jemného kameniva / písku (kg/m³)",
    "age": "Stáří betonu (dny zrání, 1 až 365)",
    "csMPa": "Pevnost v tlaku betonu (MPa, megapascals) [Target]"
}


def load_and_clean_data():
    print("=" * 70)
    print("1. Načtení dat a přejmenování sloupců")
    print("=" * 70)
    
    # Načtení se záložním kódováním
    try:
        df_raw = pd.read_csv(RAW_CSV_PATH, encoding="latin1")
    except Exception:
        df_raw = pd.read_csv(RAW_CSV_PATH, encoding="utf-8")
    
    raw_shape = df_raw.shape
    print(f"Původní rozměry: {raw_shape[0]} řádků, {raw_shape[1]} sloupců")
    
    # Nastavení standardních názvů sloupců
    df_raw.columns = COL_NAMES
    
    print("\nPrvních 5 řádků:")
    print(df_raw.head())
    
    print("\n" + "=" * 70)
    print("2. Kontrola chybějících hodnot a datových typů")
    print("=" * 70)
    missing_dict = df_raw.isna().sum().to_dict()
    dtypes_dict = {col: str(dtype) for col, dtype in df_raw.dtypes.items()}
    print("Počet chybějících hodnot:", missing_dict)
    print("Datové typy:", dtypes_dict)
    
    print("\n" + "=" * 70)
    print("3. Analýza a odstranění duplicit")
    print("=" * 70)
    n_duplicates = int(df_raw.duplicated().sum())
    print(f"Nalezeno duplicitních řádků: {n_duplicates}")
    
    duplicate_rows_sample = df_raw[df_raw.duplicated(keep=False)].sort_values(by=COL_NAMES).head(10).to_dict(orient="records")
    
    df_clean = df_raw.drop_duplicates().copy().reset_index(drop=True)
    clean_shape = df_clean.shape
    print(f"Rozměry po odstranění duplicit: {clean_shape[0]} řádků, {clean_shape[1]} sloupců")
    
    return df_raw, df_clean, n_duplicates, duplicate_rows_sample


def compute_statistics(df_clean):
    print("\n" + "=" * 70)
    print("4. Statistický souhrn a rozdělení proměnných")
    print("=" * 70)
    desc = df_clean.describe().round(3)
    print(desc)
    
    stats_dict = {}
    for col in COL_NAMES:
        series = df_clean[col]
        stats_dict[col] = {
            "name": col,
            "description": COL_DESCRIPTIONS[col],
            "count": int(series.count()),
            "mean": float(round(series.mean(), 3)),
            "std": float(round(series.std(), 3)),
            "min": float(round(series.min(), 3)),
            "q25": float(round(series.quantile(0.25), 3)),
            "median": float(round(series.median(), 3)),
            "q75": float(round(series.quantile(0.75), 3)),
            "max": float(round(series.max(), 3)),
            "skewness": float(round(series.skew(), 3)),
            "kurtosis": float(round(series.kurtosis(), 3)),
            "zeros_count": int((series == 0).sum()),
            "zeros_pct": float(round(((series == 0).sum() / len(series)) * 100, 2))
        }
    return stats_dict


def compute_correlations(df_clean):
    print("\n" + "=" * 70)
    print("5. Výpočet korelační matice (Pearson)")
    print("=" * 70)
    corr_matrix = df_clean.corr().round(4)
    print("Korelace s cílovou proměnnou csMPa:")
    print(corr_matrix[TARGET_COL].sort_values(ascending=False))
    
    corr_dict = corr_matrix.to_dict()
    corr_with_target = corr_matrix[TARGET_COL].to_dict()
    return corr_dict, corr_with_target


def perform_scaling(df_clean):
    print("\n" + "=" * 70)
    print("6. Škálování příznaků: Standardizace vs. Normalizace")
    print("=" * 70)
    
    X = df_clean[FEATURE_COLS]
    y = df_clean[TARGET_COL]
    
    # 1. Standardizace (StandardScaler: z = (x - mu) / sigma)
    std_scaler = StandardScaler()
    X_std = std_scaler.fit_transform(X)
    df_std = pd.DataFrame(X_std, columns=FEATURE_COLS)
    df_std[TARGET_COL] = y.round(2)
    
    # Uložení výsledného datasetu concrete_data_preprocessed.csv
    df_std.to_csv(PREP_CSV_PATH, index=False)
    print(f"Uložen předzpracovaný dataset: {PREP_CSV_PATH}")
    print(f"Tvar předzpracovaného datasetu: {df_std.shape}")
    
    # 2. MinMax Normalizace (MinMaxScaler: x_norm = (x - min) / (max - min))
    minmax_scaler = MinMaxScaler()
    X_minmax = minmax_scaler.fit_transform(X)
    df_minmax = pd.DataFrame(X_minmax, columns=FEATURE_COLS)
    df_minmax[TARGET_COL] = y.round(2)
    
    # Srovnávací statistiky po škálování
    std_stats = {
        col: {
            "mean": float(round(df_std[col].mean(), 4)),
            "std": float(round(df_std[col].std(), 4)),
            "min": float(round(df_std[col].min(), 4)),
            "max": float(round(df_std[col].max(), 4))
        }
        for col in FEATURE_COLS
    }
    
    minmax_stats = {
        col: {
            "mean": float(round(df_minmax[col].mean(), 4)),
            "std": float(round(df_minmax[col].std(), 4)),
            "min": float(round(df_minmax[col].min(), 4)),
            "max": float(round(df_minmax[col].max(), 4))
        }
        for col in FEATURE_COLS
    }
    
    return df_std, df_minmax, std_stats, minmax_stats


def generate_diagnostic_plots(df_clean, df_std, df_minmax):
    print("\n" + "=" * 70)
    print("7. Generování diagnostických grafů")
    print("=" * 70)
    
    # 1. Histogramy rozdělení všech 9 proměnných
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    for idx, col in enumerate(COL_NAMES):
        ax = axes[idx]
        color = "#1f77b4" if col != TARGET_COL else "#d62728"
        sns.histplot(df_clean[col], kde=True, ax=ax, color=color, bins=25, edgecolor="black", alpha=0.6)
        ax.set_title(f"{col}\n({COL_DESCRIPTIONS[col].split('(')[0].strip()})", fontsize=11, fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Počet vzorků")
        ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    hist_path = PLOTS_DIR / "concrete_histograms.png"
    fig.savefig(hist_path, dpi=200)
    plt.close(fig)
    print(f"Uložen graf histogramů: {hist_path}")
    
    # 2. Scatter plots vůči csMPa
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    for idx, col in enumerate(FEATURE_COLS):
        ax = axes[idx]
        corr_val = df_clean[col].corr(df_clean[TARGET_COL])
        sns.regplot(
            data=df_clean,
            x=col,
            y=TARGET_COL,
            ax=ax,
            scatter_kws={"alpha": 0.4, "color": "#1f77b4", "s": 20},
            line_kws={"color": "#e377c2", "linewidth": 2}
        )
        ax.set_title(f"{col} vs csMPa\n(r = {corr_val:+.3f})", fontsize=11, fontweight="bold")
        ax.set_xlabel(f"{col} (kg/m³)")
        ax.set_ylabel("csMPa (MPa)")
        ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    scatter_path = PLOTS_DIR / "concrete_scatter_vs_csmpa.png"
    fig.savefig(scatter_path, dpi=200)
    plt.close(fig)
    print(f"Uložen graf scatter závislostí: {scatter_path}")
    
    # 3. Korelační heatmapa
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df_clean.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt="+.2f",
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .8},
        ax=ax
    )
    ax.set_title("Korelační matice proměnných betonu (Pearson r)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    corr_path = PLOTS_DIR / "concrete_correlation_heatmap.png"
    fig.savefig(corr_path, dpi=200)
    plt.close(fig)
    print(f"Uložen graf korelační matice: {corr_path}")
    
    # 4. Srovnání škálování (Raw vs StandardScaler vs MinMaxScaler) na vybraných příznacích
    sample_cols = ["cement", "water", "superplasticizer", "age"]
    fig, axes = plt.subplots(len(sample_cols), 3, figsize=(15, 11))
    for i, col in enumerate(sample_cols):
        # Raw
        sns.kdeplot(df_clean[col], ax=axes[i, 0], color="#2ca02c", fill=True, alpha=0.3)
        axes[i, 0].set_title(f"Původní (Raw): {col}")
        axes[i, 0].grid(True, linestyle="--", alpha=0.5)
        
        # StandardScaler
        sns.kdeplot(df_std[col], ax=axes[i, 1], color="#1f77b4", fill=True, alpha=0.3)
        axes[i, 1].set_title(f"StandardScaler (z-score): {col}")
        axes[i, 1].grid(True, linestyle="--", alpha=0.5)
        
        # MinMaxScaler
        sns.kdeplot(df_minmax[col], ax=axes[i, 2], color="#ff7f0e", fill=True, alpha=0.3)
        axes[i, 2].set_title(f"MinMaxScaler [0, 1]: {col}")
        axes[i, 2].grid(True, linestyle="--", alpha=0.5)
        
    plt.tight_layout()
    scaling_path = PLOTS_DIR / "concrete_scaling_comparison.png"
    fig.savefig(scaling_path, dpi=200)
    plt.close(fig)
    print(f"Uložen graf srovnání škálování: {scaling_path}")


def main():
    df_raw, df_clean, n_duplicates, duplicate_rows_sample = load_and_clean_data()
    stats_dict = compute_statistics(df_clean)
    corr_dict, corr_with_target = compute_correlations(df_clean)
    df_std, df_minmax, std_stats, minmax_stats = perform_scaling(df_clean)
    generate_diagnostic_plots(df_clean, df_std, df_minmax)
    
    # Sestavení předpočtené JSON cache pro Streamlit
    precomputed = {
        "metadata": {
            "source": "I-Cheng Yeh, Modeling of strength of high performance concrete using artificial neural networks (1998)",
            "task": "Homework: Preparing data for regression models (Concrete Compressive Strength)",
            "raw_rows": int(len(df_raw)),
            "clean_rows": int(len(df_clean)),
            "num_features": len(FEATURE_COLS),
            "columns": COL_NAMES,
            "feature_cols": FEATURE_COLS,
            "target_col": TARGET_COL,
            "column_descriptions": COL_DESCRIPTIONS
        },
        "quality_audit": {
            "missing_values": df_raw.isna().sum().to_dict(),
            "has_missing": bool(df_raw.isna().sum().sum() > 0),
            "duplicates_count": n_duplicates,
            "duplicates_sample": duplicate_rows_sample,
            "dtypes": {col: str(dtype) for col, dtype in df_raw.dtypes.items()}
        },
        "statistics": stats_dict,
        "correlations": {
            "full_matrix": corr_dict,
            "with_target": corr_with_target
        },
        "scaling_comparison": {
            "standard_scaler": std_stats,
            "minmax_scaler": minmax_stats,
            "method_chosen": "StandardScaler",
            "justification": (
                "StandardScaler zachovává tvar rozdělení a nulové vycentrování (mean=0, std=1), "
                "což je ideální pro lineární modely s L1/L2 regularizací (Ridge, Lasso, ElasticNet) "
                "i neuronové sítě. Na rozdíl od MinMaxScaler není náchylný na kompresi intervalu odlehlými hodnotami."
            )
        },
        "sample_head_raw": df_clean.head(10).round(2).to_dict(orient="records"),
        "sample_head_scaled": df_std.head(10).round(2).to_dict(orient="records")
    }
    
    with open(PRECOMPUTED_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(precomputed, f, indent=2, ensure_ascii=False)
    print(f"\nUložena JSON cache: {PRECOMPUTED_JSON_PATH}")
    print("HOTOVO!")


if __name__ == "__main__":
    main()
