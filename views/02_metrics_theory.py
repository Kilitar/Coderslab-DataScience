from pathlib import Path
import streamlit as st

base_dir = Path(__file__).resolve().parent.parent
theory_md_path = base_dir / "02_Classification" / "theory" / "03_classification_metrics_theory.md"

st.title("📚 Metriky klasifikačních modelů: Teoretický rozbor")
st.caption("Matice záměn, přesnost (Accuracy), preciznost (Precision), senzitivita (Recall), F1-skóre, ROC-AUC křivka a Log Loss (Stav k 09/2026).")

if theory_md_path.exists():
    with open(theory_md_path, "r", encoding="utf-8") as f:
        content = f.read()
    st.markdown(content)
else:
    st.warning("Teoretický dokument 03_classification_metrics_theory.md nebyl nalezen.")
