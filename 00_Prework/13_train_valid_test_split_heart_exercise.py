"""
Prework Cvičení 13: Rozdělení dat na trénovací, validační a testovací sadu (Heart Dataset)
Machine Learning Course - CodersLab

Úkoly:
1. Načtení jednoho z výsledných datasetů ze cvičení škálování (heart_data_normalized.csv).
2. Návrh vhodného poměru rozdělení:
   - Doporučený poměr pro dataset této velikosti (303 vzorků) je 70 % trénovací, 15 % validační a 15 % testovací sada.
   - Trénovací sada (70 %): slouží k učení parametrů modelu.
   - Validační sada (15 %): slouží k ladění hyperparametrů a prevenci přeučení (overfittingu).
   - Testovací sada (15 %): slouží k finálnímu, nestrannému vyhodnocení zobecňující schopnosti modelu.
3. Provedení rozdělení pomocí funkce train_test_split ze Scikit-learn s využitím stratifikace podle cílové proměnné 'ahd_yes'.
4. Vypsání detailních informací o počtu řádků, sloupců a procentuálním zastoupení každé sady.
5. Uložení každé sady do samostatného souboru .csv bez indexu:
   - heart_data_train.csv
   - heart_data_valid.csv
   - heart_data_test.csv
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    # Nastavení relativních cest k souborům
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv_path = os.path.join(script_dir, "data", "heart_data_normalized.csv")
    train_csv_path = os.path.join(script_dir, "data", "heart_data_train.csv")
    valid_csv_path = os.path.join(script_dir, "data", "heart_data_valid.csv")
    test_csv_path = os.path.join(script_dir, "data", "heart_data_test.csv")

    print("=" * 80)
    print("PREWORK CVIČENÍ: ROZDĚLENÍ DAT (TRAIN / VALID / TEST SPLIT)")
    print("=" * 80)

    # ==============================================================================
    # 1. Načtení datasetu
    # ==============================================================================
    df = pd.read_csv(input_csv_path)
    total_rows, total_cols = df.shape
    print(f"\n1. Načten normalizovaný dataset: {input_csv_path}")
    print(f"   - Celkový počet řádků:   {total_rows}")
    print(f"   - Celkový počet sloupců: {total_cols}")
    print(f"   - Cílová proměnná: 'ahd_yes' (0 = bez onemocnění, 1 = srdeční onemocnění)")

    # ==============================================================================
    # 2. Návrh poměru rozdělení (Proportions proposal)
    # ==============================================================================
    print("\n" + "=" * 80)
    print("2. Návrh poměru rozdělení dat:")
    print("   - Trénovací sada (Training set):   70 %  (~212 vzorků)")
    print("   - Validační sada (Validation set): 15 %  (~45 vzorků)")
    print("   - Testovací sada  (Test set):       15 %  (~46 vzorků)")
    print("\n   Důvod:")
    print("   - 70 % poskytuje modelu dostatek vzorků pro spolehlivou konvergenci parametrů.")
    print("   - 15 % validační sada umožňuje nestranné porovnávání různých modelů a hyperparametrů.")
    print("   - 15 % testovací sada slouží jako 'nedotčená' finální prověrka.")
    print("   - Použijeme stratifikaci (stratify=df['ahd_yes']), aby všechny 3 sady měly stejný podíl diagnóz.")

    # ==============================================================================
    # 3. Rozdělení datasetu (dvoustupňový train_test_split)
    # ==============================================================================
    # 1. Krok: Oddělení trénovací sady (70 %) a dočasné sady (30 %)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["ahd_yes"]
    )

    # 2. Krok: Rozdělení dočasné sady 50:50 na validační (15 %) a testovací (15 %)
    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["ahd_yes"]
    )

    # ==============================================================================
    # 4. Zobrazení informací o jednotlivých sadách
    # ==============================================================================
    print("\n" + "=" * 80)
    print("3. Výsledky rozdělení (Počty řádků, sloupců a podíly):")
    print("=" * 80)

    subsets_info = [
        ("Trénovací sada (Train)", train_df),
        ("Validační sada (Valid)", valid_df),
        ("Testovací sada (Test)", test_df)
    ]

    for name, subset in subsets_info:
        rows, cols = subset.shape
        pct = (rows / total_rows) * 100
        dist_0 = (subset["ahd_yes"] == 0).sum()
        dist_1 = (subset["ahd_yes"] == 1).sum()
        pct_1 = (dist_1 / rows) * 100
        print(f"\n{name}:")
        print(f"   - Počet řádků:   {rows:3d} ({pct:5.1f} % z celku)")
        print(f"   - Počet sloupců: {cols:3d}")
        print(f"   - Třídy ahd_yes: 0 (No) = {dist_0:3d}, 1 (Yes) = {dist_1:3d} (zastoupení pozitivních: {pct_1:.1f} %)")

    print(f"\nKontrolní součet řádků: {len(train_df)} + {len(valid_df)} + {len(test_df)} = {len(train_df) + len(valid_df) + len(test_df)} (celkem {total_rows})")

    # ==============================================================================
    # 5. Uložení jednotlivých sad do souborů .csv bez indexu
    # ==============================================================================
    print("\n" + "=" * 80)
    print("4. Ukládání dílčích sad do .csv souborů:")
    print("=" * 80)

    train_df.to_csv(train_csv_path, index=False)
    print(f"   - Trénovací sada uložena do:  {train_csv_path}")

    valid_df.to_csv(valid_csv_path, index=False)
    print(f"   - Validační sada uložena do:  {valid_csv_path}")

    test_df.to_csv(test_csv_path, index=False)
    print(f"   - Testovací sada uložena do:  {test_csv_path}")

    print("=" * 80)
    print("CVIČENÍ ROZDĚLENÍ DAT DOKONČENO!")
    print("=" * 80)


if __name__ == "__main__":
    main()
