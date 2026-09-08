"""
Python (Plotly) Exercise - Loans Dataset Interactive Visualizations
Machine Learning Course - CodersLab

Grafy:
1. Sloupcový graf průměrné výše půjčky podle typu oblasti (area) a statusu půjčky (status).
   - Dva sloupce vedle sebe (barmode='group') odlišené barvou podle statusu.
2. Bodový graf (scatter plot) vztahu mezi příjmem žadatele (applicant_income) a výší půjčky (loan_amount).
   - Barva bodu podle proměnné married.
3. Korelační matice spojitých numerických proměnných.
   - Barevná paleta 'Oranges'.
"""

import os
import pandas as pd
import plotly.express as px

# 1. Příprava dat
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "data", "loans.csv")
plots_dir = os.path.join(script_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

loans_df = pd.read_csv(data_path)
loans_df.columns = loans_df.columns.str.lower()

# Imputace módem u kategoriálních proměnných
for col in ['gender', 'married', 'dependents', 'self_employed']:
    loans_df[col] = loans_df[col].fillna(loans_df[col].mode()[0])

# ==============================================================================
# Graf 1: Průměrná výše půjčky podle oblasti a statusu půjčky
# ==============================================================================
# Výpočet průměrné výše půjčky pro každou kombinaci area a status
avg_loan_by_area_status = loans_df.groupby(['area', 'status'], as_index=False)['loan_amount'].mean()

fig1 = px.bar(
    avg_loan_by_area_status,
    x='area',
    y='loan_amount',
    color='status',
    barmode='group',
    title='Average Property Loan Amount by Area Type and Loan Status',
    labels={
        'area': 'Property Area',
        'loan_amount': 'Average Loan Amount (Kč)',
        'status': 'Loan Status'
    },
    color_discrete_map={'Y': '#2ca02c', 'N': '#d62728'},
    text_auto='.2s'
)

fig1.update_layout(
    xaxis_title='Property Area',
    yaxis_title='Average Loan Amount',
    legend_title_text='Loan Status (Y/N)',
    template='plotly_white'
)

html1_path = os.path.join(plots_dir, "01_plotly_avg_loan_by_area.html")
fig1.write_html(html1_path)
print(f"Graf 1 uložen do HTML: {html1_path}")


# ==============================================================================
# Graf 2: Vztah mezi příjmem žadatele a výší půjčky (barevně podle married)
# ==============================================================================
fig2 = px.scatter(
    loans_df,
    x='applicant_income',
    y='loan_amount',
    color='married',
    title='Relationship Between Borrower Income and Loan Amount by Marital Status',
    labels={
        'applicant_income': 'Borrower Income (Applicant Income)',
        'loan_amount': 'Loan Amount',
        'married': 'Married'
    },
    hover_data=['education', 'self_employed', 'status'],
    opacity=0.7,
    template='plotly_white'
)

fig2.update_layout(
    xaxis_title='Borrower Income (Applicant Income)',
    yaxis_title='Loan Amount',
    legend_title_text='Married (Yes/No)'
)

html2_path = os.path.join(plots_dir, "02_plotly_income_vs_loan.html")
fig2.write_html(html2_path)
print(f"Graf 2 uložen do HTML: {html2_path}")


# ==============================================================================
# Graf 3: Korelační matice spojitých numerických proměnných (paleta Oranges)
# ==============================================================================
continuous_vars = ['applicant_income', 'coapplicant_income', 'loan_amount']
corr_matrix = loans_df[continuous_vars].corr().round(2)

fig3 = px.imshow(
    corr_matrix,
    text_auto=True,
    aspect='auto',
    color_continuous_scale='Oranges',
    title='Correlation Matrix of Continuous Numerical Variables',
    labels=dict(x='Continuous Variables', y='Continuous Variables', color='Correlation')
)

fig3.update_layout(
    xaxis_title='Numerical Variables',
    yaxis_title='Numerical Variables',
    template='plotly_white'
)

html3_path = os.path.join(plots_dir, "03_plotly_correlation_matrix.html")
fig3.write_html(html3_path)
print(f"Graf 3 uložen do HTML: {html3_path}")

print("\nVšechny Plotly vizualizace byly úspěšně vygenerovány.")
