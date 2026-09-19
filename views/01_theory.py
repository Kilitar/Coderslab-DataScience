from pathlib import Path
import streamlit as st

base_dir = Path(__file__).resolve().parent.parent
reg_dir = base_dir / "01_Regression"
theory_md_path = reg_dir / "theory" / "01_linear_regression_theory.md"

st.title("📚 Teoretický rozbor a Moderní ML (Stav k 09/2026)")
st.caption("Matematické základy OLS a Gradient Descent, kritické poznámky ke kurzu a moderní AI/ML techniky.")

if theory_md_path.exists():
    with open(theory_md_path, "r", encoding="utf-8") as f:
        content = f.read()
    st.markdown(content)
else:
    st.warning("Teoretický dokument 01_linear_regression_theory.md nebyl nalezen.")
