import streamlit as st

st.set_page_config(
    page_title="Data Science & ML Portfolio | Coderslab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Definice stránek
p_home = st.Page("views/00_home.py", title="Přehled kurzu a sylabus", icon="🏠", default=True)
p_theory = st.Page("views/01_theory.py", title="Teorie 1: Lineární regrese & OLS", icon="📚")
p_metrics_theory = st.Page("views/01_metrics_theory.py", title="Teorie 2: Metriky regrese", icon="🎯")
p_reg_theory = st.Page("views/01_regularization_theory.py", title="Teorie 3: Regularizace (Lasso & Ridge)", icon="🎛️")

# Cvičení 1 – Reality (Model + Časová analýza + Notebook)
p_kc_model = st.Page("views/01_kc_housing.py", title="Cvičení 1: Reality – Model & Dashboard", icon="🏡")
p_kc_time = st.Page("views/01_kc_time_analysis.py", title="Cvičení 1: Reality – Časová analýza & Sezónnost", icon="⏱️")
p_kc_nb = st.Page("views/01_kc_notebook.py", title="Cvičení 1: Reality – Jupyter Notebook (.ipynb)", icon="📓")

# Cvičení 2 – Diamanty (Model + Gemologická analýza + Notebook)
p_diamonds_model = st.Page("views/01_diamonds.py", title="Cvičení 2: Diamanty – Model & Dashboard", icon="💎")
p_diamonds_gemology = st.Page("views/01_diamonds_gemology_analysis.py", title="Cvičení 2: Diamanty – Gemologická analýza & 4C", icon="🔬")
p_diamonds_nb = st.Page("views/01_diamonds_notebook.py", title="Cvičení 2: Diamanty – Jupyter Notebook (.ipynb)", icon="📓")

# Cvičení 3 – Metriky reality (Výsledky zadání + Expertní analýza + Notebook)
p_metrics_kc_model = st.Page("views/01_metrics_exercise_1.py", title="Cvičení 3: Metriky reality – Výsledky zadání", icon="🎯")
p_metrics_kc_critique = st.Page("views/01_metrics_critique_analysis.py", title="Cvičení 3: Metriky reality – Expertní analýza & MLOps", icon="📐")
p_metrics_kc_nb = st.Page("views/01_metrics_notebook.py", title="Cvičení 3: Metriky reality – Jupyter Notebook (.ipynb)", icon="📓")

# Cvičení 4 – Metriky diamantů (Výsledky zadání + R2 paradox + Notebook)
p_metrics_diam_model = st.Page("views/01_diamonds_metrics.py", title="Cvičení 4: Metriky diamantů – Výsledky zadání", icon="💎")
p_metrics_diam_critique = st.Page("views/01_diamonds_metrics_critique.py", title="Cvičení 4: Metriky diamantů – Expertní analýza & R² paradox", icon="🔬")
p_metrics_diam_nb = st.Page("views/01_diamonds_metrics_nb.py", title="Cvičení 4: Metriky diamantů – Jupyter Notebook (.ipynb)", icon="📓")

# Cvičení 5 – Regularizace reality (Výsledky zadání + Expertní analýza + Notebook)
p_reg_kc_model = st.Page("views/01_regularization_exercise_1.py", title="Cvičení 5: Regularizace reality – Výsledky zadání", icon="🎛️")
p_reg_kc_critique = st.Page("views/01_regularization_critique.py", title="Cvičení 5: Regularizace reality – Expertní analýza & MLOps", icon="🔬")
p_reg_kc_nb = st.Page("views/01_regularization_nb.py", title="Cvičení 5: Regularizace reality – Jupyter Notebook (.ipynb)", icon="📓")

# Cvičení 6 – Regularizace diamantů (Výsledky zadání + Expertní analýza + Notebook)
p_reg_diam_model = st.Page("views/01_diamonds_regularization.py", title="Cvičení 6: Regularizace diamantů – Výsledky zadání", icon="💎")
p_reg_diam_critique = st.Page("views/01_diamonds_regularization_critique.py", title="Cvičení 6: Regularizace diamantů – Expertní analýza & 4C", icon="🔬")
p_reg_diam_nb = st.Page("views/01_diamonds_regularization_nb.py", title="Cvičení 6: Regularizace diamantů – Jupyter Notebook (.ipynb)", icon="📓")

# Hierarchická navigace přehledně členěná
nav = st.navigation(
    {
        "Úvod & Teoretický základ": [
            p_home,
            p_theory,
            p_metrics_theory,
            p_reg_theory,
        ],
        "01. Lineární regrese – Praktická cvičení": [
            p_kc_model,
            p_kc_time,
            p_kc_nb,
            p_diamonds_model,
            p_diamonds_gemology,
            p_diamonds_nb,
            p_metrics_kc_model,
            p_metrics_kc_critique,
            p_metrics_kc_nb,
            p_metrics_diam_model,
            p_metrics_diam_critique,
            p_metrics_diam_nb,
            p_reg_kc_model,
            p_reg_kc_critique,
            p_reg_kc_nb,
            p_reg_diam_model,
            p_reg_diam_critique,
            p_reg_diam_nb,
        ],
    }
)

nav.run()

