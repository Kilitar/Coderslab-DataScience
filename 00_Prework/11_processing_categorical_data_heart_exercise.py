"""
Prework Cvičení 11: Zpracování kategorických proměnných (Processing Categorical Data - Heart Dataset)
Machine Learning Course - CodersLab

Úkoly:
1. Načtení datasetu vytvořeného v předchozím cvičení čištění dat (heart_data_exercise_1.csv).
2. Identifikace kategorických proměnných v načteném DataFrame:
   - Textové kategorické proměnné: 'sex', 'chestpain', 'fbs', 'exang', 'thal', 'ahd'
   - Diskrétní kategorická proměnná s ordinálním/nominálním kódováním: 'restecg' (hodnoty 0, 1, 2)
3. Převod kategorických proměnných na numerické hodnoty pomocí vhodné metody:
   - Použití metody pd.get_dummies(..., drop_first=True) pro One-Hot Encoding s eliminací dummy variable trap.
4. Uložení výsledného datasetu do .csv souboru bez indexu (heart_data_exercise_2.csv).
"""

import os
import pandas as pd


def main():
    # Nastavení relativních cest k datům
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv_path = os.path.join(script_dir, "data", "heart_data_exercise_1.csv")
    output_csv_path = os.path.join(script_dir, "data", "heart_data_exercise_2.csv")

    print("=" * 80)
    print("PREWORK CVIČENÍ: ZPRACOVÁNÍ KATEGORICKÝCH DAT (HEART DATASET)")
    print("=" * 80)

    # ==============================================================================
    # 1. Načtení vyčištěného datasetu
    # ==============================================================================
    df = pd.read_csv(input_csv_path)
    print(f"\n1. Vyčištěná data načtena z: {input_csv_path}")
    print(f"   - Rozměry: {df.shape[0]} řádků, {df.shape[1]} sloupců")
    print("\nPrvních 5 řádků před kódováním:")
    print(df.head())

    # ==============================================================================
    # 2. Identifikace kategorických proměnných
    # ==============================================================================
    print("\n" + "=" * 80)
    print("2. Identifikace kategorických proměnných:")

    # Textové a diskrétní kategorické proměnné
    # 'restecg' nabývá diskrétních hodnot 0, 1, 2 reprezentujících výsledky EKG vyšetření
    cat_cols = ["sex", "chestpain", "fbs", "restecg", "exang", "thal", "ahd"]

    print(f"\n   Kategorické proměnné určené ke konverzi ({len(cat_cols)}):")
    for col in cat_cols:
        unique_vals = sorted(df[col].unique())
        print(f"     * {col:12s} ({len(unique_vals)} unikátních hodnot): {unique_vals}")

    # ==============================================================================
    # 3. Převod kategorických proměnných na numerické hodnoty (One-Hot Encoding)
    # ==============================================================================
    print("\n" + "=" * 80)
    print("3. Kódování proměnných pomocí pd.get_dummies(..., drop_first=True):")

    # Použití drop_first=True zabraňuje multikolinearitě (tzv. Dummy Variable Trap):
    # Pro k kategorií vznikne k - 1 binárních sloupců (0 / 1).
    encoded_df = pd.get_dummies(
        df,
        columns=cat_cols,
        drop_first=True,
        dtype=int
    )

    print(f"   - Původní počet sloupců: {df.shape[1]}")
    print(f"   - Nový počet sloupců po kódování: {encoded_df.shape[1]}")
    print("\nNové sloupce vzniklé kódováním:")
    new_cols = [col for col in encoded_df.columns if col not in df.columns]
    for col in new_cols:
        print(f"   - {col}")

    # ==============================================================================
    # 4. Kontrola výsledného DataFrame
    # ==============================================================================
    print("\n" + "=" * 80)
    print("4. Náhled zakódovaných dat (.head()):")
    print(encoded_df.head())
    print(f"\nDatové typy po kódování (všechny sloupce jsou nyní numerické):")
    print(encoded_df.dtypes.value_counts())

    # ==============================================================================
    # 5. Uložení výsledného datasetu do .csv souboru bez indexu
    # ==============================================================================
    print("\n" + "=" * 80)
    print("5. Uložení zakódovaného datasetu:")
    encoded_df.to_csv(output_csv_path, index=False)
    print(f"   - Soubor úspěšně uložen do: {output_csv_path}")
    print(f"   - Finální rozměry: {encoded_df.shape[0]} řádků, {encoded_df.shape[1]} sloupců")
    print("=" * 80)
    print("CVIČENÍ ZPRACOVÁNÍ KATEGORICKÝCH DAT DOKONČENO!")
    print("=" * 80)


if __name__ == "__main__":
    main()
