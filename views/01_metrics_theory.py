from pathlib import Path
import streamlit as st

base_dir = Path(__file__).resolve().parent.parent
reg_dir = base_dir / "01_Regression"
theory_md_path = reg_dir / "theory" / "02_regression_metrics_theory.md"

st.title("🎯 Teorie 2: Metriky regresních modelů")
st.caption("Jak správně určit kvalitu modelu: MAE, MSE, RMSE, R², Adjusted R² a moderní produkční metriky (09/2026).")

if theory_md_path.exists():
    with open(theory_md_path, "r", encoding="utf-8") as f:
        content = f.read()
    st.markdown(content)
else:
    st.warning("Dokument 02_regression_metrics_theory.md nebyl nalezen.")
