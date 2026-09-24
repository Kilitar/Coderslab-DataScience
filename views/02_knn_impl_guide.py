from pathlib import Path
import streamlit as st

base_dir = Path(__file__).resolve().parent.parent
guide_md_path = base_dir / "02_Classification" / "theory" / "02_knn_implementation_guide.md"

st.title("🛠️ k-NN v Scikit-learn: Praktická implementace & Rozbor")
st.caption("Parametry KNeighborsClassifier, stromové indexy kd-tree a ball-tree, inspekce .kneighbors() a kritika školního řešení.")

if guide_md_path.exists():
    with open(guide_md_path, "r", encoding="utf-8") as f:
        content = f.read()
    st.markdown(content)
else:
    st.warning("Dokument 02_knn_implementation_guide.md nebyl nalezen.")
