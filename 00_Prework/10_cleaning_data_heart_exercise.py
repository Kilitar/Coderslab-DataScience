"""
Prework Cvičení 10: Čištění dat (Cleaning Data - Heart Disease Dataset)
Machine Learning Course - CodersLab

Úkoly:
1. Načtení dat ze souboru heart_data.csv pomocí knihovny Pandas.
2. Zobrazení posledních 15 pozorování načteného DataFrame (.tail(15)).
3. Odstranění redundantních proměnných pro predikci onemocnění srdce (sloupec 'minhr', který obsahuje 100 % chybějících hodnot).
4. Ověření přítomnosti duplicitních záznamů a jejich odstranění (.duplicated(), .drop_duplicates()).
5. Ověření přítomnosti chybějících hodnot (NaN):
   - Pro numerické proměnné ('chol', 'ca') doplnění chybějících hodnot mediánem (.median()).
   - Pro kategorické proměnné ('thal') doplnění modem (.mode()).
6. Kontrola a oprava překlepů a chyb v kategorických proměnných:
   - Sjednocení velikosti písmen na malá písmena (.str.lower()).
   - Oprava překlepů v 'chestpain' ('nontypica', 'asymptomaticc', 'nontypiccal').
   - Oprava překlepu v 'thal' ('reversabble' -> 'reversable').
7. Uložení vyčištěného datasetu do .csv souboru bez indexu (index=False).
"""

import os
import pandas as pd


def main():
    # Nastavení relativních cest k datům
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv_path = os.path.join(script_dir, "data", "heart_data.csv")
    output_csv_path = os.path.join(script_dir, "data", "heart_data_exercise_1.csv")

    print("=" * 80)
    print("PREWORK CVIČENÍ: ČIŠTĚNÍ DAT (HEART DATASET)")
    print("=" * 80)

    # ==============================================================================
    # 1. Načtení dat
    # ==============================================================================
    df = pd.read_csv(input_csv_path)
    print(f"\n1. Data úspěšně načtena ze souboru: {input_csv_path}")
    print(f"   - Výchozí rozměry: {df.shape[0]} řádků, {df.shape[1]} sloupců")

    # ==============================================================================
    # 2. Zobrazení posledních 15 pozorování
    # ==============================================================================
    print("\n" + "=" * 80)
    print("2. Posledních 15 pozorování v načteném DataFrame (.tail(15)):")
    print("=" * 80)
    print(df.tail(15))

    # ==============================================================================
    # 3. Odstranění redundantních proměnných
    # ==============================================================================
    # Sloupec 'minhr' má 100 % hodnot NaN (313 z 313), je tedy zcela nepoužitelný pro model.
    print("\n" + "=" * 80)
    print("3. Analýza a odstranění redundantních sloupců:")
    null_summary = df.isnull().sum()
    empty_cols = null_summary[null_summary == len(df)].index.tolist()
    print(f"   - Sloupce se 100 % chybějících hodnot: {empty_cols}")

    df = df.drop(columns=["minhr"])
    print(f"   - Sloupec 'minhr' odstraněn. Nový tvar DataFrame: {df.shape}")

    # ==============================================================================
    # 4. Detekce a odstranění duplicit
    # ==============================================================================
    print("\n" + "=" * 80)
    print("4. Kontrola a odstranění duplicit:")
    num_duplicates = df.duplicated().sum()
    print(f"   - Nalezeno duplicitních řádků: {num_duplicates}")

    if num_duplicates > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"   - Duplicity odstraněny. Počet unikátních řádků: {len(df)}")

    # ==============================================================================
    # 5. Ověření a ošetření chybějících hodnot (Missing Values)
    # ==============================================================================
    print("\n" + "=" * 80)
    print("5. Analýza a doplnění chybějících hodnot (NaN):")
    missing_before = df.isnull().sum()
    print("   - Sloupce s chybějícími hodnotami:")
    print(missing_before[missing_before > 0])

    # Numerické proměnné: 'chol' a 'ca' doplníme mediánem
    numeric_cols_with_nulls = ["chol", "ca"]
    for col in numeric_cols_with_nulls:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"   - Sloupec '{col}': chybějící hodnoty doplněny mediánem ({median_val})")

    # Kategorické proměnné: 'thal' doplníme nejčastější hodnotou (mode)
    thal_mode = df["thal"].mode()[0]
    df["thal"] = df["thal"].fillna(thal_mode)
    print(f"   - Sloupec 'thal': chybějící hodnoty doplněny modem ('{thal_mode}')")

    print(f"   - Celkový počet zbývajících chybějících hodnot: {df.isnull().sum().sum()}")

    # ==============================================================================
    # 6. Kontrola a oprava chyb v kategorických proměnných
    # ==============================================================================
    print("\n" + "=" * 80)
    print("6. Kontrola a oprava překlepů a nestejné velikosti písmen:")

    cat_cols = ["sex", "chestpain", "fbs", "exang", "thal", "ahd"]

    # 6a. Sjednocení na malá písmena (převod např. 'Male'/'Female', 'Yes'/'No', 'FIXED'/'Fixed')
    for col in cat_cols:
        df[col] = df[col].astype(str).str.lower()

    # 6b. Oprava specifických překlepů
    chestpain_fixes = {
        "nontypica": "nontypical",
        "asymptomaticc": "asymptomatic",
        "nontypiccal": "nontypical"
    }
    df["chestpain"] = df["chestpain"].replace(chestpain_fixes)
    print(f"   - 'chestpain' opraveny překlepy: {list(chestpain_fixes.keys())}")

    thal_fixes = {
        "reversabble": "reversable"
    }
    df["thal"] = df["thal"].replace(thal_fixes)
    print(f"   - 'thal' opraven překlep: 'reversabble' -> 'reversable'")

    print("\n   Unikátní hodnoty po vyčištění:")
    for col in cat_cols:
        print(f"   - {col:10s}: {sorted(df[col].unique())}")

    # ==============================================================================
    # 7. Uložení vyčištěného datasetu do .csv souboru bez indexu
    # ==============================================================================
    print("\n" + "=" * 80)
    print("7. Uložení vyčištěného datasetu:")
    df.to_csv(output_csv_path, index=False)
    print(f"   - Soubor úspěšně uložen do: {output_csv_path}")
    print(f"   - Finální rozměry: {df.shape[0]} řádků, {df.shape[1]} sloupců")
    print("=" * 80)
    print("CVIČENÍ ČIŠTĚNÍ DAT DOKONČENO!")
    print("=" * 80)


if __name__ == "__main__":
    main()
