"""
Main Prework Project: Příprava dat pro Python model (Cancer Dataset)
Machine Learning Course - CodersLab

Tento projekt demonstruje kompletní proces přípravy a předzpracování dat
(Data Preparation & Preprocessing Pipeline) pro model strojového učení:
1. Načtení a prvotní inspekce dat (Pandas, .head(), .info(), .isnull()).
2. Odstranění redundantních sloupců ('id', 'Unnamed: 32', 'Unnamed: 0').
3. Řešení chybějících hodnot:
   - Výpočet procentuálního podílu chybějících hodnot
   - Odstranění řádků s chybějícími hodnotami (.dropna()) -> 549 řádků
   - Ukázka imputace pomocí SimpleImputer (strategie mean).
4. Kontrola a odstranění duplicit (.duplicated()).
5. Kódování kategorických proměnných ('diagnosis'):
   - Oprava překlepů / normalizace velikosti písmen ('b' -> 'B' / 'm' -> 'M')
   - Metoda 1: .replace()
   - Metoda 2 [DOPORUČENÁ]: pd.get_dummies(..., drop_first=True)
   - Metoda 3: OneHotEncoder ze scikit-learn.
6. Analýza numerických vztahů a distribucí:
   - Deskriptivní statistiky (.describe())
   - Korelační matice a heatmapa (Seaborn)
   - Analýza rozdělení vybraných proměnných (histogram & KDE distribuce)
   - Normalizace dat (vlastní funkce vs. MinMaxScaler).
7. Rozdělení dat na trénovací, validační a testovací sadu:
   - 2-stupňové rozdělení (70 % train, 30 % test) přes train_test_split
   - 3-stupňové rozdělení (70 % train, 15 % valid, 15 % test)
   - Ukázka přístupu s knihovnou fast_ml.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def normalize_custom(col):
    """Vlastní normalizační funkce (Min-Max škálování do rozsahu 0 až 1)."""
    min_val = col.min()
    max_val = col.max()
    if max_val - min_val == 0:
        return col
    return (col - min_val) / (max_val - min_val)


def main():
    # Nastavení cest
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data", "cancer_data_course.csv")
    plots_dir = os.path.join(script_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print("=" * 80)
    print("MAIN PREWORK PROJECT: PŘÍPRAVA DAT PRO MODEL (CANCER DATASET)")
    print("=" * 80)

    # ==============================================================================
    # 1. Načtení datasetu a prvotní inspekce
    # ==============================================================================
    cancer_df = pd.read_csv(data_path)
    print(f"\n1. Dataset načten:")
    print(f"   - Cesta: {data_path}")
    print(f"   - Počet řádků: {cancer_df.shape[0]}, počet sloupců: {cancer_df.shape[1]}")
    print("\nPrvních 5 řádků:")
    print(cancer_df.head())

    # ==============================================================================
    # 2. Odstranění redundantních proměnných
    # ==============================================================================
    # Sloupec 'id' je pouze identifikátor, 'Unnamed: 32' (případně 'Unnamed: 0')
    # jsou artefakty exportu obsahující pouze NaN nebo index.
    redundant_cols = [col for col in ["id", "Unnamed: 32", "Unnamed: 0"] if col in cancer_df.columns]
    cancer_df = cancer_df.drop(columns=redundant_cols)
    print("\n" + "=" * 80)
    print(f"2. Odstraněny redundantní sloupce: {redundant_cols}")
    print(f"   - Nové rozměry datasetu: {cancer_df.shape}")

    # ==============================================================================
    # 3. Práce s chybějícími hodnotami
    # ==============================================================================
    print("\n" + "=" * 80)
    print("3. Analýza chybějících hodnot:")
    null_counts = cancer_df.isnull().sum()
    null_percentage = (null_counts / len(cancer_df)) * 100
    missing_summary = pd.DataFrame({
        "Počet chybějících": null_counts,
        "Procento chybějících (%)": null_percentage.round(2)
    })
    cols_with_nulls = missing_summary[missing_summary["Počet chybějících"] > 0]
    print(f"Počet sloupců s chybějícími hodnotami: {len(cols_with_nulls)}")
    print(cols_with_nulls.head(10))

    # Varianta A: Odstranění řádků s chybějícími hodnotami (přístup zvolený v kurzu)
    cancer_df_cleaned = cancer_df.dropna(axis=0).reset_index(drop=True)
    print(f"\nVarianta A (dropna): Počet řádků po odstranění chybějících hodnot: {len(cancer_df_cleaned)} (původně {len(cancer_df)})")

    # Varianta B (Ukázka): Imputace pomocí SimpleImputer (strategie průměru - mean)
    imputer = SimpleImputer(strategy="mean")
    numeric_cols = cancer_df.select_dtypes(include=[np.number]).columns
    imputed_numeric = imputer.fit_transform(cancer_df[numeric_cols])
    print(f"Varianta B (SimpleImputer): Úspěšně nafitován a transformován pro {len(numeric_cols)} numerických sloupců.")

    # Pokračujeme s vyčištěným datasetem (549 řádků)
    cancer_df = cancer_df_cleaned

    # ==============================================================================
    # 4. Kontrola a odstranění duplicit
    # ==============================================================================
    print("\n" + "=" * 80)
    print("4. Kontrola duplicit:")
    duplicates_count = cancer_df.duplicated().sum()
    print(f"   - Počet duplicitních řádků: {duplicates_count}")
    if duplicates_count > 0:
        cancer_df = cancer_df.drop_duplicates().reset_index(drop=True)
        print("   - Duplicity byly odstraněny.")

    # ==============================================================================
    # 5. Kódování kategorických proměnných ('diagnosis')
    # ==============================================================================
    print("\n" + "=" * 80)
    print("5. Analýza a kódování kategorické proměnné 'diagnosis':")
    print("   - Četnosti před sjednocením:")
    print(cancer_df["diagnosis"].value_counts())

    # Oprava překlepu: hodnota 'b' na 'B' (resp. normalizace na velká písmena)
    cancer_df["diagnosis"] = cancer_df["diagnosis"].replace("b", "B")
    print("\n   - Četnosti po sjednocení ('b' -> 'B'):")
    print(cancer_df["diagnosis"].value_counts())

    # [DOPORUČENÁ METODA]: pd.get_dummies(..., drop_first=True)
    # Vytvoří sloupec diagnosis_M (1 = zhoubný/malignant, 0 = nezhoubný/benign)
    cancer_encoded_df = pd.get_dummies(cancer_df, columns=["diagnosis"], drop_first=True, dtype=int)
    target_col = [c for c in cancer_encoded_df.columns if "diagnosis" in c][0]
    print(f"\n   - Kódování provedeno pomocí pd.get_dummies(drop_first=True)")
    print(f"   - Název binární cílové proměnné: '{target_col}'")
    print(f"   - Rozložení: 0 (Benign) = {(cancer_encoded_df[target_col] == 0).sum()}, 1 (Malignant) = {(cancer_encoded_df[target_col] == 1).sum()}")

    # ==============================================================================
    # 6. Analýza vztahů v numerických datech a vizualizace
    # ==============================================================================
    print("\n" + "=" * 80)
    print("6. Analýza numerických dat a vizualizace:")
    desc_stats = cancer_encoded_df.describe().T[["mean", "std", "min", "max"]]
    print("Základní statistiky pro prvních 5 proměnných:")
    print(desc_stats.head())

    # a) Korelační matice a heatmapa
    corr_matrix = cancer_encoded_df.corr()
    plt.figure(figsize=(18, 16))
    sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        annot=False,
        linewidths=0.5,
        cbar_kws={'label': 'Korelační koeficient'}
    )
    plt.title("Korelační matice vlastností nádorových buněk", fontsize=16, fontweight="bold", pad=15)
    plt.tight_layout()
    corr_plot_path = os.path.join(plots_dir, "09_cancer_correlation_heatmap.png")
    plt.savefig(corr_plot_path, dpi=200)
    plt.close()
    print(f"   - Korelační heatmapa uložena do: {corr_plot_path}")

    # b) Distribuce vybrané proměnné (radius_mean)
    plt.figure(figsize=(10, 6))
    sns.histplot(cancer_encoded_df["radius_mean"], kde=True, color="#1f77b4", bins=30)
    plt.title("Distribuce proměnné 'radius_mean' (Histogram + KDE)", fontsize=14, fontweight="bold")
    plt.xlabel("radius_mean")
    plt.ylabel("Četnost")
    plt.tight_layout()
    dist_plot_path = os.path.join(plots_dir, "09_radius_mean_distribution.png")
    plt.savefig(dist_plot_path, dpi=200)
    plt.close()
    print(f"   - Graf distribuce radius_mean uložen do: {dist_plot_path}")

    # ==============================================================================
    # 7. Normalizace dat (Min-Max Scaling)
    # ==============================================================================
    print("\n" + "=" * 80)
    print("7. Škálování a normalizace dat (MinMaxScaler):")
    scaler = MinMaxScaler()
    normalized_array = scaler.fit_transform(cancer_encoded_df)
    cancer_normalized_df = pd.DataFrame(normalized_array, columns=scaler.get_feature_names_out())
    print("   - Všechny proměnné normalizovány do intervalu [0, 1].")
    print(f"   - Minima všech sloupců: min = {cancer_normalized_df.min().min():.1f}, maxima = {cancer_normalized_df.max().max():.1f}")

    # ==============================================================================
    # 8. Rozdělení na trénovací, validační a testovací množinu
    # ==============================================================================
    print("\n" + "=" * 80)
    print("8. Rozdělení dat na podmnožiny:")
    X = cancer_normalized_df.drop(columns=[target_col])
    y = cancer_normalized_df[target_col]

    # Standardní 2-cestné rozdělení: 70 % train, 30 % test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    print(f"\n   A) 2-cestné rozdělení (70 % train / 30 % test):")
    print(f"      - X_train: {X_train.shape}, y_train: {y_train.shape} (70 %)")
    print(f"      - X_test:  {X_test.shape},  y_test:  {y_test.shape}  (30 %)")

    # 3-cestné rozdělení: 70 % train, 15 % valid, 15 % test (rozdělením testovací sady 50:50)
    X_test_final, X_valid, y_test_final, y_valid = train_test_split(
        X_test, y_test, test_size=0.50, random_state=42
    )
    print(f"\n   B) 3-cestné rozdělení (70 % train / 15 % valid / 15 % test):")
    print(f"      - X_train: {X_train.shape} (70 %)")
    print(f"      - X_valid: {X_valid.shape} (15 %)")
    print(f"      - X_test:  {X_test_final.shape} (15 %)")

    # Kontrola přítomnosti knihovny fast_ml
    print("\n   C) Alternativa: Knihovna fast_ml:")
    try:
        from fast_ml.model_development import train_valid_test_split
        X_tr, y_tr, X_val, y_val, X_te, y_te = train_valid_test_split(
            cancer_normalized_df, target=target_col, train_size=0.70, valid_size=0.15, test_size=0.15
        )
        print("      - Rozděleno přes fast_ml:", X_tr.shape, X_val.shape, X_te.shape)
    except ImportError:
        print("      - Knihovna fast_ml není nainstalována (volitelná: 'pip install fast_ml').")
        print("      - Běžná osvědčená metoda přes Scikit-learn train_test_split výše plně postačuje.")

    print("\n" + "=" * 80)
    print("DATOVÝ PIPELINE PŘÍPRAVY DAT DOKONČEN ÚSPĚŠNĚ!")
    print("=" * 80)


if __name__ == "__main__":
    main()
