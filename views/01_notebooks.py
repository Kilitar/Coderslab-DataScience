import base64
import json
from pathlib import Path
import streamlit as st

base_dir = Path(__file__).resolve().parent.parent
reg_dir = base_dir / "01_Regression"

def render_jupyter_notebook(nb_path: Path):
    if not nb_path.exists():
        st.warning(f"Soubor {nb_path.name} nebyl v repozitáři nalezen.")
        return

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_json = json.load(f)

    with open(nb_path, "rb") as f:
        bytes_data = f.read()

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        st.download_button(
            label=f"📥 Stáhnout soubor {nb_path.name}",
            data=bytes_data,
            file_name=nb_path.name,
            mime="application/x-ipynb+json",
            use_container_width=True,
        )
    with col_info:
        st.caption(f"Umístění: `01_Regression/{nb_path.name}` | Počet buněk: {len(nb_json.get('cells', []))}")

    st.markdown("---")

    for cell in nb_json.get("cells", []):
        cell_type = cell.get("cell_type")
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue

        if cell_type == "markdown":
            st.markdown(source)
        elif cell_type == "code":
            exec_count = cell.get("execution_count")
            exec_str = f"[{exec_count}]" if exec_count is not None else "[ ]"
            st.markdown(f"**Vstupní kód buňky `In {exec_str}`:**")
            st.code(source, language="python")

            outputs = cell.get("outputs", [])
            if outputs:
                with st.expander(f"Výstup buňky {exec_str}", expanded=True):
                    for out in outputs:
                        out_type = out.get("output_type")
                        if out_type == "stream":
                            text = "".join(out.get("text", []))
                            st.text(text)
                        elif out_type in ("execute_result", "display_data"):
                            data = out.get("data", {})
                            if "image/png" in data:
                                img_bytes = base64.b64decode(data["image/png"])
                                st.image(img_bytes)
                            elif "text/plain" in data:
                                st.text("".join(data["text/plain"]))
        st.write("")

st.title("📓 Jupyter Notebooky (.ipynb)")
st.caption("Kompletní interaktivní zobrazení vypracovaných sešitů buňku po buňce včetně kódů a reálných výstupů.")

selected_nb = st.selectbox(
    "Zvolte cvičení pro zobrazení sešitu:",
    [
        "01. Cvičení 1: Lineární regrese – Nemovitosti King County (01_linear_regression_exercise_1.ipynb)",
        "02. Cvičení 2: Lineární regrese – Klenotník a diamanty (02_linear_regression_exercise_2.ipynb)",
    ],
)

if "01_linear_regression_exercise_1" in selected_nb:
    nb_file = reg_dir / "01_linear_regression_exercise_1.ipynb"
else:
    nb_file = reg_dir / "02_linear_regression_exercise_2.ipynb"

render_jupyter_notebook(nb_file)
