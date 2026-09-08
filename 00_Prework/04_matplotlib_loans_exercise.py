"""
Python (Matplotlib) Exercise - Loans Dataset Visualization
Machine Learning Course - CodersLab

Grafy:
1. Sloupcový graf počtu žadatelů podle pohlaví (barva light blue + číselné hodnoty nad sloupci).
2. Sloupcový graf počtu žadatelů podle statusu půjčky (schváleno / neschváleno, barva orange).
3. Histogram rozdělení proměnné loan_amount (50 binů, barva red).
"""

import os
import matplotlib.pyplot as plt
import pandas as pd

# 1. Příprava dat (stejný postup jako v předchozí úloze)
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "data", "loans.csv")
plots_dir = os.path.join(script_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

loans_df = pd.read_csv(data_path)
loans_df.columns = loans_df.columns.str.lower()

# Nahrazení chybějících hodnot u kategoriálních proměnných módem
for col in ['gender', 'married', 'dependents', 'self_employed']:
    loans_df[col] = loans_df[col].fillna(loans_df[col].mode()[0])

print("Příprava dat hotova. Rozměry loans_df:", loans_df.shape)


# ==============================================================================
# Graf 1: Počet žadatelů podle pohlaví (Gender)
# ==============================================================================
plt.figure(figsize=(7, 5))
gender_counts = loans_df['gender'].value_counts()

bars1 = plt.bar(
    gender_counts.index,
    gender_counts.values,
    color="lightblue",
    edgecolor="steelblue",
    width=0.5
)

plt.title("Number of Borrowers by Gender", fontsize=14, fontweight="bold", pad=12)
plt.xlabel("Gender", fontsize=11, fontweight="semibold")
plt.ylabel("Number of Borrowers", fontsize=11, fontweight="semibold")
plt.grid(axis="y", linestyle="--", alpha=0.5)

# Přidání číselných hodnot nad sloupce (optional)
plt.bar_label(bars1, padding=3, fontsize=11, fontweight="bold")
plt.tight_layout()

plot1_path = os.path.join(plots_dir, "01_borrowers_by_gender.png")
plt.savefig(plot1_path, dpi=150)
print(f"Graf 1 uložen do: {plot1_path}")
plt.close()


# ==============================================================================
# Graf 2: Počet žadatelů podle statusu půjčky (Loan Status)
# ==============================================================================
plt.figure(figsize=(7, 5))
status_counts = loans_df['status'].value_counts()

# Převedení 'Y' / 'N' na srozumitelnější popisky 'Granted (Y)' / 'Not Granted (N)'
status_labels = ['Granted (Y)' if s == 'Y' else 'Not Granted (N)' for s in status_counts.index]

bars2 = plt.bar(
    status_labels,
    status_counts.values,
    color="orange",
    edgecolor="darkorange",
    width=0.5
)

plt.title("Number of Borrowers by Loan Status", fontsize=14, fontweight="bold", pad=12)
plt.xlabel("Loan Status", fontsize=11, fontweight="semibold")
plt.ylabel("Number of Borrowers", fontsize=11, fontweight="semibold")
plt.grid(axis="y", linestyle="--", alpha=0.5)

# Číselné hodnoty nad sloupci
plt.bar_label(bars2, padding=3, fontsize=11, fontweight="bold")
plt.tight_layout()

plot2_path = os.path.join(plots_dir, "02_borrowers_by_status.png")
plt.savefig(plot2_path, dpi=150)
print(f"Graf 2 uložen do: {plot2_path}")
plt.close()


# ==============================================================================
# Graf 3: Distribuce proměnné loan_amount (Histogram, 50 binů)
# ==============================================================================
plt.figure(figsize=(9, 5))

# Přepočet loan_amount na miliony pro lepší čitelnost osy X
plt.hist(
    loans_df['loan_amount'],
    bins=50,
    color="red",
    edgecolor="darkred",
    alpha=0.75
)

plt.title("Distribution of Loan Amount", fontsize=14, fontweight="bold", pad=12)
plt.xlabel("Loan Amount", fontsize=11, fontweight="semibold")
plt.ylabel("Frequency (Number of Loans)", fontsize=11, fontweight="semibold")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

plot3_path = os.path.join(plots_dir, "03_distribution_loan_amount.png")
plt.savefig(plot3_path, dpi=150)
print(f"Graf 3 uložen do: {plot3_path}")
plt.close()

print("\nVšechny grafy byly úspěšně vygenerovány a uloženy.")
