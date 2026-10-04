"""
View pro zobrazení Jupyter Notebooku: Cvičení 2 – XGBoost Regrese diamantů
"""

from pathlib import Path
import streamlit as st
from views.notebook_renderer import render_jupyter_notebook


def render_xgboost_diamonds_nb_view():
    st.title("🐍 Cvičení 2: XGBoost Regrese diamantů – Jupyter Notebook")
    st.markdown(
        r"""
        Interaktivní náhled a stažení Jupyter Notebooku pro **Cvičení 2 (XGBoost Regrese – diamonds.csv / diamonds_preprocessed.csv)**.
        Notebook demonstruje rozdělení dat 70:30, nastavení `RandomizedSearchCV` s metrikou `neg_mean_absolute_error`, 
        vyhodnocení testovacího MAE a porovnání s nativním laděním XGBoost.
        """
    )

    base_dir = Path(__file__).resolve().parent.parent
    nb_path = base_dir / "03_Advanced_ML_Neural_Networks" / "07_xgboost_diamonds_exercise_2.ipynb"

    if nb_path.exists():
        with open(nb_path, "r", encoding="utf-8") as f:
            nb_json_str = f.read()

        st.download_button(
            label="📥 Stáhnout Jupyter Notebook (.ipynb)",
            data=nb_json_str,
            file_name="07_xgboost_diamonds_exercise_2.ipynb",
            mime="application/x-ipynb+json",
        )
        st.markdown("---")
        render_jupyter_notebook(nb_path)
    else:
        st.error(f"Soubor notebooku nebyl nalezen na cestě: `{nb_path}`")


if __name__ == "__main__":
    render_xgboost_diamonds_nb_view()
