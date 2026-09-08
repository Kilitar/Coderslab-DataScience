"""
Google Colab Exercise - Prework
Machine Learning Course - CodersLab
"""

import os
import pandas as pd

# Krok 1: Připojení Google Drive (Mounting Google Drive)
# Při spuštění v Google Colab vás Colab požádá o potvrzení přístupu k Disku Google.
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
    base_dir = "/content/drive/MyDrive/Machine Learning/Google Colab"
except ImportError:
    IN_COLAB = False
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "data")
    print("Skript běží mimo Google Colab (lokálně). Používám lokální cestu.")

# Krok 2: Ověření verze knihovny Pandas
print(f"Instalovaná verze Pandas: {pd.__version__}")

# Krok 3: Načtení dat ze souboru student_scores.csv
input_file = os.path.join(base_dir, "student_scores.csv")

if not os.path.exists(input_file) and IN_COLAB:
    # Alternativní cesta pokud je v názvu mezera 'My Drive'
    alt_file = "/content/drive/My Drive/Machine Learning/Google Colab/student_scores.csv"
    if os.path.exists(alt_file):
        input_file = alt_file

print(f"Načítám data z: {input_file}")
df = pd.read_csv(input_file)

print("\nNáhled načtených dat (prvních 5 řádků):")
print(df.head())
print(f"\nRozměry datasetu (řádky, sloupce): {df.shape}")

# Krok 4: Přidání nového sloupce TotalScore
# TotalScore je součtem sloupců MathScore, ReadingScore a WritingScore
df['TotalScore'] = df['MathScore'] + df['ReadingScore'] + df['WritingScore']

print("\nNáhled se sloupcem TotalScore:")
print(df[['MathScore', 'ReadingScore', 'WritingScore', 'TotalScore']].head())

# Krok 5: Uložení upraveného DataFrame na Google Drive jako colab_exercise.csv
output_file = os.path.join(base_dir, "colab_exercise.csv")
df.to_csv(output_file, index=False)
print(f"\nSoubor byl úspěšně uložen do: {output_file}")
