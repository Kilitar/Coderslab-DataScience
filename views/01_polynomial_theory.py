from pathlib import Path
import streamlit as st

base_dir = Path(__file__).resolve().parent.parent
reg_dir = base_dir / "01_Regression"
theory_md_path = reg_dir / "theory" / "05_polynomial_regression_theory.md"

st.title("📈 Teorie 5: Polynomiální regrese")
st.caption("Modelování nelinearit, volba stupně polynomu, Rungeho fenomén a moderní alternativy (Splines, GAM/EBM) (09/2026).")

if theory_md_path.exists():
    with open(theory_md_path, "r", encoding="utf-8") as f:
        content = f.read()
    st.markdown(content)
else:
    st.warning("Dokument 05_polynomial_regression_theory.md nebyl nalezen.")
