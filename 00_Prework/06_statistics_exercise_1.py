"""
Statistics Exercise 1 - Real Estate Prices in Manhattan
Machine Learning Course - CodersLab

Úkoly:
1. Extrakce cen z dictionary do NumPy array.
2. Výpočet základních statistických ukazatelů pomocí knihovny NumPy:
   - Průměr (Mean)
   - Medián (Median)
   - Minimální a maximální hodnota (Min, Max)
   - Rozptyl (Variance)
   - Směrodatná odchylka (Standard Deviation)
3. Uložení hodnot do proměnných a vypsání pomocí print().
"""

import numpy as np

# Vstupní data
real_estate_prices = {
    'Broadway': 1200000,
    'Park Avenue': 3500000,
    'Madison Avenue': 2800000,
    'Fifth Avenue': 5000000,
    'Lexington Avenue': 2100000,
    'West End Avenue': 1800000,
    'Central Park West': 4500000,
    'Riverside Drive': 1600000,
    'Waverly Place': 1100000,
    'Bleecker Street': 900000,
    'Christopher Street': 950000,
    'Houston Street': 1300000,
    'Canal Street': 800000,
    'Bowery': 700000,
    'Delancey Street': 1000000
}

# 1. Extrakce hodnot cen do NumPy array
prices = np.array(list(real_estate_prices.values()))

# 2. Výpočet statistických metrik pomocí funkcí knihovny NumPy
mean_price = np.mean(prices)
median_price = np.median(prices)
min_price = np.min(prices)
max_price = np.max(prices)
var_price = np.var(prices)
std_price = np.std(prices)

# 3. Zobrazení výsledků
print("=" * 60)
print("STATISTICKÉ UKAZATELE CEN NEMOVITOSTÍ V MANHATTANU")
print("=" * 60)
print(f"NumPy pole cen (délka: {len(prices)}):")
print(prices)
print("-" * 60)
print(f"Průměr (Mean):               ${mean_price:,.2f}")
print(f"Medián (Median):             ${median_price:,.2f}")
print(f"Minimální hodnota (Min):     ${min_price:,.2f}")
print(f"Maximální hodnota (Max):     ${max_price:,.2f}")
print(f"Rozptyl (Variance):          ${var_price:,.2f}")
print(f"Směrodatná odchylka (Std):   ${std_price:,.2f}")
print("=" * 60)
