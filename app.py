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

# Cvičení 1 – Dvojice (Model + Notebook)
p_kc_model = st.Page("views/01_kc_housing.py", title="Cvičení 1: Nemovitosti King County", icon="🏡")
p_kc_nb = st.Page("views/01_kc_notebook.py", title="Cvičení 1: Jupyter Notebook (.ipynb)", icon="📓")

# Cvičení 2 – Dvojice (Model + Notebook)
p_diamonds_model = st.Page("views/01_diamonds.py", title="Cvičení 2: Klenotník a diamanty", icon="💎")
p_diamonds_nb = st.Page("views/01_diamonds_notebook.py", title="Cvičení 2: Jupyter Notebook (.ipynb)", icon="📓")

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
            p_kc_nb,
            p_diamonds_model,
            p_diamonds_nb,
        ],
    }
)

nav.run()
