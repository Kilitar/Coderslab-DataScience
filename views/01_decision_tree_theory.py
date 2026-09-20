from pathlib import Path
import streamlit as st

base_dir = Path(__file__).resolve().parent.parent
reg_dir = base_dir / "01_Regression"
theory_md_path = reg_dir / "theory" / "06_decision_tree_regression_theory.md"

st.title("🌳 Teorie 6: Rozhodovací strom v regresi")
st.caption("Struktura stromu, rekurzivní binární dělení, kritéria MSE/MAE, kontrola přeučení a přemostění k ansámblům (09/2026).")

if theory_md_path.exists():
    with open(theory_md_path, "r", encoding="utf-8") as f:
        content = f.read()
    st.markdown(content)
else:
    st.warning("Dokument 06_decision_tree_regression_theory.md nebyl nalezen.")
