"""
Python (Pandas) Exercise - Loans Dataset
Machine Learning Course - CodersLab

Kroky:
1. Načtení loans.csv jako pandas DataFrame.
2. Přejmenování všech sloupců na malá písmena.
3. Zobrazení prvních 10 řádků.
4. Kontrola datových typů a potřebné konverze.
5. Přehled chybějících hodnot (.isnull().sum()).
6. Nahrazení chybějících hodnot u kategoriálních proměnných nejčastější hodnotou (.mode()).
7. Vytvoření loans_granted (pouze schválené půjčky).
8. Vytvoření standalone_borrowers (schválené půjčky samostatným žadatelům).
9. Vytvoření loans_count_by_gender (počet půjček, průměrný příjem, průměrná výše půjčky dle pohlaví).
"""

import os
import pandas as pd

# Cesta k datovému souboru loans.csv
data_path = os.path.join(os.path.dirname(__file__), "data", "loans.csv")

# ==============================================================================
# Krok 1: Načtení .csv souboru
# ==============================================================================
print("=" * 70)
print("KROK 1: Načtení souboru loans.csv")
print("=" * 70)
df = pd.read_csv(data_path)
print(f"Dataset úspěšně načten. Rozměry: {df.shape[0]} řádků, {df.shape[1]} sloupců.")

# ==============================================================================
# Krok 2: Přejmenování sloupců na malá písmena
# ==============================================================================
print("\n" + "=" * 70)
print("KROK 2: Přejmenování sloupců na malá písmena")
print("=" * 70)
df.columns = df.columns.str.lower()
print("Nové názvy sloupců:")
print(df.columns.tolist())

# ==============================================================================
# Krok 3: Zobrazení prvních 10 pozorování
# ==============================================================================
print("\n" + "=" * 70)
print("KROK 3: Prvních 10 řádků datasetu")
print("=" * 70)
print(df.head(10))

# ==============================================================================
# Krok 4: Kontrola typů proměnných a případné konverze
# ==============================================================================
print("\n" + "=" * 70)
print("KROK 4: Kontrola datových typů")
print("=" * 70)
print("Datové typy před úpravou:")
print(df.dtypes)

# Kategoriální sloupce převedeme na typ 'category' pro optimalizaci paměti a sémantickou přesnost
categorical_columns = ['gender', 'married', 'education', 'self_employed', 'area', 'status']
for col in categorical_columns:
    df[col] = df[col].astype('category')

print("\nDatové typy po konverzi na category:")
print(df.dtypes)

# ==============================================================================
# Krok 5: Přehled chybějících hodnot (.isnull())
# ==============================================================================
print("\n" + "=" * 70)
print("KROK 5: Počet chybějících hodnot v jednotlivých sloupcích")
print("=" * 70)
missing_summary = df.isnull().sum()
print(missing_summary)

# ==============================================================================
# Krok 6: Nahrazení chybějících hodnot u kategoriálních proměnných hodnotou módu
# ==============================================================================
print("\n" + "=" * 70)
print("KROK 6: Imputace chybějících hodnot nejčastější hodnotou (.mode())")
print("=" * 70)
# Kategoriální proměnné s chybějícími hodnotami: gender, married, dependents, self_employed
cat_cols_to_impute = ['gender', 'married', 'dependents', 'self_employed']

for col in cat_cols_to_impute:
    col_mode = df[col].mode()[0]
    missing_count = df[col].isnull().sum()
    df[col] = df[col].fillna(col_mode)
    print(f"Sloupec '{col}': nahrazeno {missing_count} prázdných hodnot módem '{col_mode}'.")

# Také credit_history je binární příznak (0/1), kde lze chybějící hodnoty nahradit módem (1.0)
credit_mode = df['credit_history'].mode()[0]
df['credit_history'] = df['credit_history'].fillna(credit_mode)

print("\nKontrola chybějících hodnot po imputaci:")
print(df[cat_cols_to_impute + ['credit_history']].isnull().sum())

# ==============================================================================
# Krok 7: loans_granted - pouze schválené půjčky (status == 'Y')
# ==============================================================================
print("\n" + "=" * 70)
print("KROK 7: Schválené půjčky (loans_granted)")
print("=" * 70)
loans_granted = df[df['status'] == 'Y'].copy()
print(f"Počet schválených půjček: {len(loans_granted)} z celkových {len(df)}")
print(loans_granted.head(3))

# ==============================================================================
# Krok 8: standalone_borrowers - půjčky samostatným žadatelům
# ==============================================================================
print("\n" + "=" * 70)
print("KROK 8: Samostatní žadatelé se schválenou půjčkou (standalone_borrowers)")
print("=" * 70)
# Samostatný žadatel (standalone applicant) nemá spolužadatele -> coapplicant_income == 0
standalone_borrowers = loans_granted[loans_granted['coapplicant_income'] == 0].copy()
print(f"Počet schválených půjček samostatným žadatelům: {len(standalone_borrowers)}")
print(standalone_borrowers.head(3))

# ==============================================================================
# Krok 9: loans_count_by_gender - statistiky podle pohlaví
# ==============================================================================
print("\n" + "=" * 70)
print("KROK 9: Statistiky půjček podle pohlaví (loans_count_by_gender)")
print("=" * 70)
# Agregace: počet půjček, průměrný příjem žadatele, průměrná výše půjčky
loans_count_by_gender = loans_granted.groupby('gender', observed=False).agg(
    loans_count=('loan_amount', 'count'),
    avg_applicant_income=('applicant_income', 'mean'),
    avg_loan_amount=('loan_amount', 'mean')
).reset_index()

# Zaokrouhlení průměrů pro přehlednost
loans_count_by_gender['avg_applicant_income'] = loans_count_by_gender['avg_applicant_income'].round(2)
loans_count_by_gender['avg_loan_amount'] = loans_count_by_gender['avg_loan_amount'].round(2)

print("Výsledný DataFrame loans_count_by_gender (pro schválené půjčky):")
print(loans_count_by_gender)

# Alternativní výpočet pro celý dataset (pokud by zadání myslelo všechny podané žádosti):
loans_count_by_gender_all = df.groupby('gender', observed=False).agg(
    loans_count=('loan_amount', 'count'),
    avg_applicant_income=('applicant_income', 'mean'),
    avg_loan_amount=('loan_amount', 'mean')
).reset_index().round(2)

print("\nAlternativa pro všechny podané žádosti (celý dataset df):")
print(loans_count_by_gender_all)
