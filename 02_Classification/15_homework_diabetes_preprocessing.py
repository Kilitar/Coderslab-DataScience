# ==============================================================================
# Homework: Příprava dat pro klasifikační modely (Diabetes Dataset)
# ==============================================================================
# Úkol:
# 1. Načtení datové sady diabetes.csv do Pandas DataFrame.
# 2. Normalizace názvů sloupců (malá písmena, slova oddělená podtržítkem).
# 3. Kontrola chybějících hodnot (standardních NaN/null).
# 4. Analýza četnosti tříd v cílové proměnné 'outcome'.
# 5. Analýza distribuce proměnných (histogramy, šikmost).
# 6. Identifikace a nahrazení biologicky nemožných nul (glucose, blood_pressure,
#    skin_thickness, insulin, bmi) hodnotami NaN.
# 7. Imputace chybějících hodnot:
#    - Průměr pro přibližně normální rozdělení (glucose, blood_pressure).
#    - Medián pro šikmá rozdělení (skin_thickness, insulin, bmi).
# 8. Analýza korelační matice.
# 9. Škálování příznaků pomocí standardizace (StandardScaler).
# 10. Uložení upraveného a škálovaného datasetu do diabetes_scaled.csv.
# ==============================================================================

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

# Cesty k datům
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

RAW_CSV_PATH = os.path.join(DATA_DIR, "diabetes.csv")
SCALED_CSV_PATH = os.path.join(DATA_DIR, "diabetes_scaled.csv")
JSON_OUTPUT_PATH = os.path.join(DATA_DIR, "diabetes_preprocessing_precomputed.json")

# Záložní cesta k původnímu souboru, pokud v 02_Classification/data ještě není
ALT_RAW_CSV_PATH = os.path.join(os.path.dirname(BASE_DIR), "data", "MAL_downloadable materials_session 1", "Homework", "diabetes.csv")

def run_diabetes_preprocessing():
    print("=" * 80)
    print("KROK 1: Načtení datové sady")
    print("=" * 80)
    if os.path.exists(RAW_CSV_PATH):
        df_raw = pd.read_csv(RAW_CSV_PATH)
    elif os.path.exists(ALT_RAW_CSV_PATH):
        df_raw = pd.read_csv(ALT_RAW_CSV_PATH)
        df_raw.to_csv(RAW_CSV_PATH, index=False)
    else:
        raise FileNotFoundError(f"Soubor diabetes.csv nebyl nalezen ani na {RAW_CSV_PATH}, ani na {ALT_RAW_CSV_PATH}")

    print(f"Původní tvar dat: {df_raw.shape[0]} řádků, {df_raw.shape[1]} sloupců")
    print(f"Původní názvy sloupců: {list(df_raw.columns)}")

    print("\n" + "=" * 80)
    print("KROK 2: Normalizace názvů sloupců")
    print("=" * 80)
    column_mapping = {
        'Pregnancies': 'pregnancies',
        'Glucose': 'glucose',
        'BloodPressure': 'blood_pressure',
        'SkinThickness': 'skin_thickness',
        'Insulin': 'insulin',
        'BMI': 'bmi',
        'DiabetesPedigreeFunction': 'diabetes_pedigree_function',
        'Age': 'age',
        'Outcome': 'outcome'
    }
    df = df_raw.rename(columns=column_mapping)
    # Zajištění převodu všech na lower_case se snake_case
    df.columns = [c.strip().lower() for c in df.columns]
    print(f"Normalizované sloupce: {list(df.columns)}")

    print("\n" + "=" * 80)
    print("KROK 3: Kontrola standardních chybějících hodnot (NaN)")
    print("=" * 80)
    initial_nulls = df.isnull().sum().to_dict()
    print("Počet explicitních NaN hodnot v původním CSV:")
    for col, count in initial_nulls.items():
        print(f"  - {col}: {count}")

    print("\n" + "=" * 80)
    print("KROK 4: Četnost cílové proměnné 'outcome'")
    print("=" * 80)
    class_counts = df['outcome'].value_counts().to_dict()
    class_ratios = df['outcome'].value_counts(normalize=True).to_dict()
    print(f"Třída 0 (Bez diabetu): {class_counts.get(0, 0)} ({class_ratios.get(0, 0) * 100:.2f} %)")
    print(f"Třída 1 (Diabetes):     {class_counts.get(1, 0)} ({class_ratios.get(1, 0) * 100:.2f} %)")

    print("\n" + "=" * 80)
    print("KROK 5 & 6: Detekce biologicky nemožných nul a nahrazení za NaN")
    print("=" * 80)
    # Žijící člověk nemůže mít nulovou glykémii, nulový krevní tlak, nulovou tloušťku kožní řasy,
    # nulový 2hodinový inzulin ani BMI rovné 0.
    zero_invalid_cols = ['glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi']
    zero_counts = {}
    for col in df.columns:
        zero_counts[col] = int((df[col] == 0).sum())

    print("Počet nulových hodnot v jednotlivých sloupcích:")
    for col, zc in zero_counts.items():
        pct = (zc / len(df)) * 100
        note = "<- Biologicky nemožné (chybějící údaj!)" if col in zero_invalid_cols else "(Platná nula)"
        print(f"  - {col:25s}: {zc:4d} nul ({pct:5.2f} %) {note}")

    df_nan = df.copy()
    for col in zero_invalid_cols:
        df_nan[col] = df_nan[col].replace(0, np.nan)

    created_nans = {col: int(df_nan[col].isna().sum()) for col in zero_invalid_cols}

    print("\n" + "=" * 80)
    print("KROK 7: Analýza rozdělení a imputace chybějících hodnot")
    print("=" * 80)
    skewness_dict = {}
    imputation_strategy = {}
    imputation_values = {}

    for col in zero_invalid_cols:
        valid_series = df_nan[col].dropna()
        skew_val = float(valid_series.skew())
        skewness_dict[col] = skew_val
        
        # Pravidlo ze zadání:
        # Přibližně normální rozdělení (|skew| < 0.5) -> průměr
        # Šikmé rozdělení (|skew| >= 0.5) -> medián
        if abs(skew_val) < 0.5:
            strategy = "mean"
            fill_val = float(valid_series.mean())
        else:
            strategy = "median"
            fill_val = float(valid_series.median())
            
        imputation_strategy[col] = strategy
        imputation_values[col] = fill_val
        print(f"Sloupec: {col:16s} | Šikmost (skew): {skew_val:+.3f} | Zvolená metoda: {strategy:6s} | Hodnota: {fill_val:.3f}")

    df_imputed = df_nan.copy()
    for col, fill_val in imputation_values.items():
        df_imputed[col] = df_imputed[col].fillna(fill_val)

    print(f"\nZbývající počet NaN po imputaci v celém datasetu: {df_imputed.isna().sum().sum()}")

    print("\n" + "=" * 80)
    print("KROK 8: Korelační analýza")
    print("=" * 80)
    corr_matrix = df_imputed.corr()
    print("Korelace s cílovou proměnnou 'outcome' (seřazeno sestupně):")
    outcome_corr = corr_matrix['outcome'].sort_values(ascending=False).to_dict()
    for col, val in outcome_corr.items():
        if col != 'outcome':
            print(f"  - {col:25s}: {val:+.4f}")

    print("\n" + "=" * 80)
    print("KROK 9: Škálování příznaků (Standardizace)")
    print("=" * 80)
    feature_cols = [c for c in df_imputed.columns if c != 'outcome']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_imputed[feature_cols])
    
    df_scaled = pd.DataFrame(scaled_features, columns=feature_cols)
    df_scaled['outcome'] = df_imputed['outcome'].values

    print("Statistiky po standardizaci (výběr prvních 3 příznaků):")
    for col in feature_cols[:3]:
        print(f"  - {col}: průměr = {df_scaled[col].mean():.4f}, směrodatná odchylka = {df_scaled[col].std():.4f}")

    print("\n" + "=" * 80)
    print("KROK 10: Uložení upraveného a škálovaného datasetu")
    print("=" * 80)
    df_scaled.to_csv(SCALED_CSV_PATH, index=False)
    print(f"Dataset úspěšně uložen do: {SCALED_CSV_PATH}")

    # Uložíme i do root data/ složky pro případ, že další úlohy hledají data v rootu
    root_scaled_path = os.path.join(os.path.dirname(BASE_DIR), "data", "diabetes_scaled.csv")
    df_scaled.to_csv(root_scaled_path, index=False)
    print(f"Kopie uložena také do: {root_scaled_path}")

    # Příprava dat pro interaktivní aplikaci
    precomputed_data = {
        "raw_shape": list(df_raw.shape),
        "columns_original": list(df_raw.columns),
        "columns_normalized": list(df.columns),
        "initial_nulls": initial_nulls,
        "class_distribution": {
            "0": int(class_counts.get(0, 0)),
            "1": int(class_counts.get(1, 0)),
            "ratio_0": float(class_ratios.get(0, 0)),
            "ratio_1": float(class_ratios.get(1, 0))
        },
        "zero_counts": zero_counts,
        "zero_invalid_cols": zero_invalid_cols,
        "created_nans": created_nans,
        "skewness": skewness_dict,
        "imputation_strategy": imputation_strategy,
        "imputation_values": imputation_values,
        "correlation_with_outcome": outcome_corr,
        "correlation_matrix": {col: corr_matrix[col].to_dict() for col in corr_matrix.columns},
        "scaler_means": {col: float(mean) for col, mean in zip(feature_cols, scaler.mean_)},
        "scaler_scales": {col: float(scale) for col, scale in zip(feature_cols, scaler.scale_)},
        "sample_head_raw": df.head(5).to_dict(orient="records"),
        "sample_head_scaled": df_scaled.head(5).to_dict(orient="records"),
        "descriptive_stats_before": df.describe().to_dict(),
        "descriptive_stats_imputed": df_imputed.describe().to_dict(),
        "descriptive_stats_scaled": df_scaled.describe().to_dict()
    }

    with open(JSON_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(precomputed_data, f, ensure_ascii=False, indent=2)
    print(f"Precomputed metadata uložena do: {JSON_OUTPUT_PATH}")

    return df_scaled, precomputed_data

if __name__ == "__main__":
    run_diabetes_preprocessing()
