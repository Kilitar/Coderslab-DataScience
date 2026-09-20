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

    img_path = reg_dir / "plots" / "16_regularization_geometric_contours.png"
    marker = "![Geometrie regularizace](https://raw.githubusercontent.com/Kilitar/Coderslab-DataScience/main/01_Regression/plots/16_regularization_geometric_contours.png)"
    if marker in content and img_path.exists():
        part1, part2 = content.split(marker, 1)
        st.markdown(part1)
        st.image(
            str(img_path),
            caption="Geometrická interpretace: Porovnání ohraničení přípustné oblasti (L2 kruh, L1 kosočtverec, L1+L2 zaoblený kosočtverec s grouping efektem) a dotyku s elipsami ztrátové funkce.",
            width="stretch"
        )
        st.markdown(part2)
    else:
        st.markdown(content)
else:
    st.warning("Dokument 04_elastic_net_theory.md nebyl nalezen.")

