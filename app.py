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
p_metrics = st.Page("views/01_metrics_theory.py", title="Teorie 2: Metriky regrese", icon="🎯")

# Cvičení 1 – Reality (Model + Časová analýza + Notebook)
p_kc_model = st.Page("views/01_kc_housing.py", title="Cvičení 1: Reality – Model & Dashboard", icon="🏡")
p_kc_time = st.Page("views/01_kc_time_analysis.py", title="Cvičení 1: Reality – Časová analýza & Sezónnost", icon="⏱️")
p_kc_nb = st.Page("views/01_kc_notebook.py", title="Cvičení 1: Reality – Jupyter Notebook (.ipynb)", icon="📓")

# Cvičení 2 – Diamanty (Model + Gemologická analýza + Notebook)
p_diamonds_model = st.Page("views/01_diamonds.py", title="Cvičení 2: Diamanty – Model & Dashboard", icon="💎")
p_diamonds_gemology = st.Page("views/01_diamonds_gemology_analysis.py", title="Cvičení 2: Diamanty – Gemologická analýza & 4C", icon="🔬")
p_diamonds_nb = st.Page("views/01_diamonds_notebook.py", title="Cvičení 2: Diamanty – Jupyter Notebook (.ipynb)", icon="📓")

# Hierarchická navigace přehledně členěná
nav = st.navigation(
    {
        "Úvod & Teoretický základ": [
            p_home,
            p_theory,
            p_metrics,
        ],
        "01. Lineární regrese – Praktická cvičení": [
            p_kc_model,
            p_kc_time,
            p_kc_nb,
            p_diamonds_model,
            p_diamonds_gemology,
            p_diamonds_nb,
        ],
    }
)

nav.run()
