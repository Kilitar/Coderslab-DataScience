import streamlit as st

st.set_page_config(
    page_title="Data Science & ML Portfolio | Coderslab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Definice stránek
p_home = st.Page("views/00_home.py", title="Přehled kurzu a sylabus", icon="🏠", default=True)

# Téma 1: Lineární regrese & OLS
p_theory = st.Page("views/01_theory.py", title="Teorie: Lineární regrese & OLS", icon="📖")
p_kc_model = st.Page("views/01_kc_housing.py", title="Cvičení 1: Reality – Výsledky zadání", icon="🎯")
p_kc_time = st.Page("views/01_kc_time_analysis.py", title="Cvičení 1: Reality – Expertní analýza", icon="🔬")
p_kc_nb = st.Page("views/01_kc_notebook.py", title="Cvičení 1: Reality – Notebook", icon="🐍")
p_diamonds_model = st.Page("views/01_diamonds.py", title="Cvičení 2: Diamanty – Výsledky zadání", icon="🎯")
p_diamonds_gemology = st.Page("views/01_diamonds_gemology_analysis.py", title="Cvičení 2: Diamanty – Expertní analýza", icon="🔬")
p_diamonds_nb = st.Page("views/01_diamonds_notebook.py", title="Cvičení 2: Diamanty – Notebook", icon="🐍")

# Téma 2: Metriky regresních modelů
p_metrics_theory = st.Page("views/01_metrics_theory.py", title="Teorie: Metriky regrese", icon="📖")
p_metrics_kc_model = st.Page("views/01_metrics_exercise_1.py", title="Cvičení 3: Metriky reality – Výsledky zadání", icon="🎯")
p_metrics_kc_critique = st.Page("views/01_metrics_critique_analysis.py", title="Cvičení 3: Metriky reality – Expertní analýza", icon="🔬")
p_metrics_kc_nb = st.Page("views/01_metrics_notebook.py", title="Cvičení 3: Metriky reality – Notebook", icon="🐍")
p_metrics_diam_model = st.Page("views/01_diamonds_metrics.py", title="Cvičení 4: Metriky diamantů – Výsledky zadání", icon="🎯")
p_metrics_diam_critique = st.Page("views/01_diamonds_metrics_critique.py", title="Cvičení 4: Metriky diamantů – Expertní analýza", icon="🔬")
p_metrics_diam_nb = st.Page("views/01_diamonds_metrics_nb.py", title="Cvičení 4: Metriky diamantů – Notebook", icon="🐍")

# Téma 3: Regularizace (Lasso, Ridge, Elastic Net)
p_reg_theory = st.Page("views/01_regularization_theory.py", title="Teorie: Regularizace (Lasso & Ridge)", icon="📖")
p_elastic_theory = st.Page("views/01_elastic_net_theory.py", title="Teorie: Elastic Net regrese", icon="📖")
p_reg_kc_model = st.Page("views/01_regularization_exercise_1.py", title="Cvičení 5: Regularizace reality – Výsledky zadání", icon="🎯")
p_reg_kc_critique = st.Page("views/01_regularization_critique.py", title="Cvičení 5: Regularizace reality – Expertní analýza", icon="🔬")
p_reg_kc_nb = st.Page("views/01_regularization_nb.py", title="Cvičení 5: Regularizace reality – Notebook", icon="🐍")
p_reg_diam_model = st.Page("views/01_diamonds_regularization.py", title="Cvičení 6: Regularizace diamantů – Výsledky zadání", icon="🎯")
p_reg_diam_critique = st.Page("views/01_diamonds_regularization_critique.py", title="Cvičení 6: Regularizace diamantů – Expertní analýza", icon="🔬")
p_reg_diam_nb = st.Page("views/01_diamonds_regularization_nb.py", title="Cvičení 6: Regularizace diamantů – Notebook", icon="🐍")

# Téma 4: Polynomiální regrese
p_poly_theory = st.Page("views/01_polynomial_theory.py", title="Teorie: Polynomiální regrese", icon="📖")

# Hierarchická navigace přehledně členěná dle tematických bloků
nav = st.navigation(
    {
        "Úvod & Přehled kurzu": [
            p_home,
        ],
        "01. Lineární regrese (OLS)": [
            p_theory,
            p_kc_model,
            p_kc_time,
            p_kc_nb,
            p_diamonds_model,
            p_diamonds_gemology,
            p_diamonds_nb,
        ],
        "02. Metriky regresních modelů": [
            p_metrics_theory,
            p_metrics_kc_model,
            p_metrics_kc_critique,
            p_metrics_kc_nb,
            p_metrics_diam_model,
            p_metrics_diam_critique,
            p_metrics_diam_nb,
        ],
        "03. Regularizace (Lasso, Ridge, Elastic Net)": [
            p_reg_theory,
            p_elastic_theory,
            p_reg_kc_model,
            p_reg_kc_critique,
            p_reg_kc_nb,
            p_reg_diam_model,
            p_reg_diam_critique,
            p_reg_diam_nb,
        ],
        "04. Polynomiální regrese": [
            p_poly_theory,
        ],
    }
)

nav.run()

