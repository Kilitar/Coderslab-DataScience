from pathlib import Path
import streamlit as st

base_dir = Path(__file__).resolve().parent.parent
theory_md_path = base_dir / "02_Classification" / "theory" / "04_logistic_regression_theory.md"

st.title("📚 Logistická regrese: Teoretický rozbor")
st.caption("Sigmoidální funkce, odvození logitu a šancí (Odds), ztrátová funkce Log Loss, metoda maximální věrohodnosti (MLE) a moderní MLOps standardy (Stav k 09/2026).")

if theory_md_path.exists():
    with open(theory_md_path, "r", encoding="utf-8") as f:
        content = f.read()
    st.markdown(content)
else:
    st.warning("Teoretický dokument 04_logistic_regression_theory.md nebyl nalezen.")
