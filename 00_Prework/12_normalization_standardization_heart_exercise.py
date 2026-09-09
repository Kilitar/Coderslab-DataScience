"""
Prework Cvičení 12: Škálování dat - Normalizace a Standardizace (Heart Dataset)
Machine Learning Course - CodersLab

Úkoly:
1. Načtení datasetu vytvořeného v předchozím cvičení (heart_data_exercise_2.csv).
2. Přístup 1: Normalizace dat (Min-Max Scaling do intervalu [0, 1]):
   - Použití třídy MinMaxScaler ze Scikit-learn nebo vlastní funkce v Pandas.
   - Aplikace na spojité numerické proměnné ('age', 'restbp', 'chol', 'maxhr', 'oldpeak', 'slope', 'ca').
   - Uložení výsledného DataFrame do proměnné `heart_data_normalized`.
3. Přístup 2: Standardizace dat (Z-score škálování na průměr = 0 a směrodatnou odchylku = 1):
   - Použití třídy StandardScaler ze Scikit-learn nebo vzorce (x - mean) / std.
   - Aplikace na spojité numerické proměnné.
   - Uložení výsledného DataFrame do proměnné `heart_data_standardized`.
4. Uložení obou výsledných datasetů do samostatných souborů .csv bez indexu:
   - heart_data_normalized.csv
   - heart_data_standardized.csv
"""

import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def normalize_pandas(df, columns):
    """Normalizace vybraných sloupců pomocí vzorce Min-Max: (x - min) / (max - min)."""
    df_copy = df.copy()
    for col in columns:
        col_min = df_copy[col].min()
        col_max = df_copy[col].max()
        df_copy[col] = (df_copy[col] - col_min) / (col_max - col_min)
    return df_copy


def standardize_pandas(df, columns):
    """Standardizace vybraných sloupců pomocí Z-score: (x - mean) / std."""
    df_copy = df.copy()
    for col in columns:
        col_mean = df_copy[col].mean()
        col_std = df_copy[col].std(ddof=0)  # ddof=0 pro shodu se StandardScaler
        df_copy[col] = (df_copy[col] - col_mean) / col_std
    return df_copy


def main():
    # Nastavení relativních cest k datům
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv_path = os.path.join(script_dir, "data", "heart_data_exercise_2.csv")
    output_norm_path = os.path.join(script_dir, "data", "heart_data_normalized.csv")
    output_std_path = os.path.join(script_dir, "data", "heart_data_standardized.csv")

    print("=" * 80)
    print("PREWORK CVIČENÍ: NORMALIZACE A STANDARDIZACE DAT (HEART DATASET)")
    print("=" * 80)

    # ==============================================================================
    # 1. Načtení zakódovaného datasetu
    # ==============================================================================
    df = pd.read_csv(input_csv_path)
    print(f"\n1. Data načtena z: {input_csv_path}")
    print(f"   - Rozměry: {df.shape[0]} řádků, {df.shape[1]} sloupců")

    # Identifikace spojitých numerických sloupců ke škálování
    # Binární dummy proměnné (0/1) není žádoucí znovu škálovat
    numeric_cols = ["age", "restbp", "chol", "maxhr", "oldpeak", "slope", "ca"]
    dummy_cols = [c for c in df.columns if c not in numeric_cols]

    # Uspořádání sloupců: nejprve dummy proměnné, následně spojité proměnné
    ordered_cols = dummy_cols + numeric_cols

    print(f"\n   Spojité numerické proměnné ke škálování ({len(numeric_cols)}): {numeric_cols}")
    print(f"   Binární dummy proměnné ({len(dummy_cols)}): {dummy_cols}")

    print("\nStatistiky numerických proměnných před škálováním:")
    print(df[numeric_cols].describe().T[["mean", "std", "min", "max"]].round(2))

    # ==============================================================================
    # 2. Přístup 1: Normalizace dat (MinMaxScaler do rozsahu [0, 1])
    # ==============================================================================
    print("\n" + "=" * 80)
    print("2. Normalizace dat (MinMaxScaler):")

    min_max = MinMaxScaler()
    heart_data_normalized = df.copy()
    heart_data_normalized[numeric_cols] = min_max.fit_transform(heart_data_normalized[numeric_cols])

    # Uspořádání sloupců dle zavedené struktury kurzu
    heart_data_normalized = heart_data_normalized[ordered_cols]

    print("   - Proměnné úspěšně normalizovány do rozsahu [0, 1].")
    print(f"   - Minima v heart_data_normalized: {heart_data_normalized[numeric_cols].min().min():.2f}")
    print(f"   - Maxima v heart_data_normalized: {heart_data_normalized[numeric_cols].max().max():.2f}")
    print("\nNáhled normalizovaných dat (prvních 5 řádků vybraných proměnných):")
    print(heart_data_normalized[numeric_cols].head())

    # ==============================================================================
    # 3. Přístup 2: Standardizace dat (StandardScaler na mean=0, std=1)
    # ==============================================================================
    print("\n" + "=" * 80)
    print("3. Standardizace dat (StandardScaler):")

    std_scaler = StandardScaler()
    heart_data_standardized = df.copy()
    heart_data_standardized[numeric_cols] = std_scaler.fit_transform(heart_data_standardized[numeric_cols])

    # Uspořádání sloupců dle zavedené struktury kurzu
    heart_data_standardized = heart_data_standardized[ordered_cols]

    print("   - Proměnné úspěšně standardizovány (Z-score).")
    print("   - Průměry standardizovaných proměnných:")
    print(heart_data_standardized[numeric_cols].mean().round(4))
    print("   - Směrodatné odchylky standardizovaných proměnných:")
    print(heart_data_standardized[numeric_cols].std().round(4))
    print("\nNáhled standardizovaných dat (prvních 5 řádků vybraných proměnných):")
    print(heart_data_standardized[numeric_cols].head())

    # ==============================================================================
    # 4. Uložení obou datasetů do souborů .csv bez indexu
    # ==============================================================================
    print("\n" + "=" * 80)
    print("4. Uložení obou výsledných datasetů bez indexu:")

    heart_data_normalized.to_csv(output_norm_path, index=False)
    print(f"   - Normalizovaný dataset uložen do: {output_norm_path}")
    print(f"     (Tvar: {heart_data_normalized.shape})")

    heart_data_standardized.to_csv(output_std_path, index=False)
    print(f"   - Standardizovaný dataset uložen do: {output_std_path}")
    print(f"     (Tvar: {heart_data_standardized.shape})")

    print("=" * 80)
    print("CVIČENÍ ŠKÁLOVÁNÍ DAT (NORMALIZACE & STANDARDIZACE) DOKONČENO!")
    print("=" * 80)


if __name__ == "__main__":
    main()
