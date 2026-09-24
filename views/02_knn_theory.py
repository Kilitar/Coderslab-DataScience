from pathlib import Path
import streamlit as st

base_dir = Path(__file__).resolve().parent.parent
theory_md_path = base_dir / "02_Classification" / "theory" / "01_knn_theory.md"

st.title("📚 K-Nearest Neighbors (k-NN): Teoretický základ")
st.caption("Matematické principy metrik vzdáleností, geometrie Voronoiových buněk, volba parametru k a kletba dimenzionality (Stav k 09/2026).")

if theory_md_path.exists():
    with open(theory_md_path, "r", encoding="utf-8") as f:
        content = f.read()
    st.markdown(content)
else:
    st.warning("Teoretický dokument 01_knn_theory.md nebyl nalezen.")
