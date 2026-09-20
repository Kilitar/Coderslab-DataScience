from pathlib import Path
import streamlit as st

base_dir = Path(__file__).resolve().parent.parent
reg_dir = base_dir / "01_Regression"
theory_md_path = reg_dir / "theory" / "04_elastic_net_theory.md"

st.title("🎛️ Teorie 4: Elastic Net regrese")
st.caption("Kombinace L1 a L2 regularizace, řešení limitů Lassa, Grouping Effect a moderní ML praktiky (09/2026).")

if theory_md_path.exists():
    with open(theory_md_path, "r", encoding="utf-8") as f:
        content = f.read()
    st.markdown(content)
else:
    st.warning("Dokument 04_elastic_net_theory.md nebyl nalezen.")
