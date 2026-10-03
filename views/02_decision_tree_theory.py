from pathlib import Path
import streamlit as st

base_dir = Path(__file__).resolve().parent.parent
theory_md_path = base_dir / "02_Classification" / "theory" / "05_decision_tree_classification_theory.md"

st.title("📚 Rozhodovací stromy: Teoretický rozbor")
st.caption("Rozdělovací kritéria (Gini Impurity, Shannonova entropie, Information Gain), CART algoritmus, problém přeučení (overfitting), post-pruning a TreeSHAP.")

if theory_md_path.exists():
    with open(theory_md_path, "r", encoding="utf-8") as f:
        content = f.read()
    st.markdown(content)
else:
    st.warning("Teoretický dokument 05_decision_tree_classification_theory.md nebyl nalezen.")
