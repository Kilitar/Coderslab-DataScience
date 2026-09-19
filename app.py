import streamlit as st

st.set_page_config(
    page_title="Data Science & ML Portfolio | Coderslab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Definice jednotlivých stránek
p_home = st.Page("views/00_home.py", title="Přehled kurzu", icon="🏠", default=True)
p_kc = st.Page("views/01_kc_housing.py", title="Cvičení 1: Nemovitosti King County", icon="🏡")
p_diamonds = st.Page("views/01_diamonds.py", title="Cvičení 2: Klenotník a diamanty", icon="💎")
p_notebooks = st.Page("views/01_notebooks.py", title="Jupyter sešity (.ipynb)", icon="📓")
p_theory = st.Page("views/01_theory.py", title="Teorie 1: Lineární regrese", icon="📚")
p_metrics = st.Page("views/01_metrics_theory.py", title="Teorie 2: Metriky regrese", icon="🎯")

# Hierarchická navigace:
nav = st.navigation(
    {
        "": [p_home],
        "01. Lineární regrese": [
            p_kc,
            p_diamonds,
            p_notebooks,
            p_theory,
            p_metrics,
        ],
    }
)

nav.run()
