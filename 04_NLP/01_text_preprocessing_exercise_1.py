"""
Processing textual data - Exercise 1
====================================
Zadání:
1. Stáhněte datový soubor pojmenovaný imdb_reviews.csv.
2. Načtěte data ze souboru do proměnné imdb_reviews pomocí knihovny Pandas.
3. Napište funkci, která jako argument přijme text recenze a vrátí text v očištěné podobě:
   - Všechna slova budou složena z malých písmen (lowercased),
   - V textu zůstanou pouze písmena (Nápověda: použijte regulární výraz),
   - Z textu budou odstraněna stop-slova (můžete použít jakýkoliv způsob, který jste se naučili).
4. Do DataFrame přidejte sloupec review_cleaned s výsledkem volání funkce z kroku 3 na sloupec obsahující texty recenzí. Použijte metodu .apply() z knihovny Pandas.
5. Uložte DataFrame do souboru .csv jako imdb_reviews_preprocessed_1.csv.
"""

import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords

# --- KROK 1 & 2: Načtení datového souboru ---
DATA_PATH = "data/imdb_reviews.csv"
if not os.path.exists(DATA_PATH):
    DATA_PATH = "data/MAL_downloadable materials_session 2/Day 4/imdb_reviews.csv"

print(f"1. Načítání souboru z: {DATA_PATH}")
imdb_reviews = pd.read_csv(DATA_PATH)
print(f"Načteno {len(imdb_reviews):,} recenzí. Sloupce: {imdb_reviews.columns.tolist()}")
print("Distribuce sentimentu:")
print(imdb_reviews["sentiment"].value_counts())

# Příprava stop-slov z knihovny NLTK
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))

# --- KROK 3: Funkce pro čištění textu ---
def clean_review(text: str) -> str:
    """
    Vyčistí text recenze podle zadání:
    1. Převod na malá písmena.
    2. Ponechání pouze písmen pomocí regulárního výrazu.
    3. Odstranění stop-slov.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Ponechání pouze písmen (odstranění čísel, HTML tagů, interpunkce)
    # Zvolíme náhradu nealfabetických znaků za mezeru (nebo vyhledání slovních bloků)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    
    # 3. Dělení na slova a filtrace stop-slov
    words = text.split()
    filtered_words = [w for w in words if w not in stop_words]
    
    return " ".join(filtered_words)

# Ukázka na prvním řádku
print("\n--- Ukázka čištění na 1. recenzi ---")
sample_raw = imdb_reviews["review"].iloc[0]
sample_clean = clean_review(sample_raw)
print("PŮVODNÍ (prvních 120 znaků):", sample_raw[:120])
print("OČIŠTĚNÝ (prvních 120 znaků):", sample_clean[:120])

# --- KROK 4: Aplikace na celý sloupec DataFrame ---
print("\n4. Aplikuji funkci clean_review na celý DataFrame pomocí .apply()...")
imdb_reviews["review_cleaned"] = imdb_reviews["review"].apply(clean_review)

# --- KROK 5: Uložení do CSV ---
OUTPUT_PATH = "data/imdb_reviews_preprocessed_1.csv"
print(f"\n5. Ukládám výsledek do souboru: {OUTPUT_PATH}")
imdb_reviews.to_csv(OUTPUT_PATH, index=False)
print("Hotovo! Soubor úspěšně vytvořen.")
