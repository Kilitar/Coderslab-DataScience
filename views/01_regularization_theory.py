from pathlib import Path
import streamlit as st

base_dir = Path(__file__).resolve().parent.parent
reg_dir = base_dir / "01_Regression"
theory_md_path = reg_dir / "theory" / "03_regularization_theory.md"

st.title("🎛️ Teorie 3: Regularizace v lineární regresi")
st.caption("Kompromis mezi vychýlením a rozptylem (Bias-Variance), Lasso (L1), Ridge (L2), Elastic Net a kritické zhodnocení (09/2026).")

if theory_md_path.exists():
    with open(theory_md_path, "r", encoding="utf-8") as f:
        content = f.read()
    st.markdown(content)
else:
    st.warning("Dokument 03_regularization_theory.md nebyl nalezen.")
