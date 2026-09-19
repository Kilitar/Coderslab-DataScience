import base64
import io
import json
import sys
from pathlib import Path
import streamlit as st

GITHUB_REPO = "Kilitar/Coderslab-DataScience"

def render_jupyter_notebook(nb_rel_path: str, title: str, description: str):
    """
    Vykreslí Jupyter notebook (.ipynb) s čistým zobrazením buněk,
    reálnými výstupy a možností spuštění v Google Colab nebo lokálně.
    """
    st.title(title)
    st.caption(description)

    base_dir = Path(__file__).resolve().parent.parent
    nb_path = base_dir / nb_rel_path

    if not nb_path.exists():
        st.error(f"Soubor `{nb_rel_path}` nebyl v repozitáři nalezen.")
        return

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_json = json.load(f)

    with open(nb_path, "rb") as f:
        bytes_data = f.read()

    # Akční panel s možnostmi spuštění a stažení
    col_colab, col_dl, col_info = st.columns([1.2, 1.2, 2.6])
    
    colab_url = f"https://colab.research.google.com/github/{GITHUB_REPO}/blob/main/{nb_rel_path.replace('\\', '/')}"
    github_url = f"https://github.com/{GITHUB_REPO}/blob/main/{nb_rel_path.replace('\\', '/')}"

    with col_colab:
        st.markdown(
            f"""
            <a href="{colab_url}" target="_blank">
                <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" style="height: 38px; vertical-align: middle;">
            </a>
            """,
            unsafe_allow_html=True
        )

    with col_dl:
        st.download_button(
            label="📥 Stáhnout .ipynb",
            data=bytes_data,
            file_name=nb_path.name,
            mime="application/x-ipynb+json",
            use_container_width=True,
        )

    with col_info:
        st.markdown(
            f"🔗 [Zobrazit na GitHubu]({github_url}) &nbsp;|&nbsp; "
            f"Buněk: **{len(nb_json.get('cells', []))}** &nbsp;|&nbsp; "
            f"Jádro: **Python 3.14**"
        )

    st.markdown("---")

    # Informační pruh o možnostech spouštění
    st.info(
        """
        💡 **Jak sešit spustit a interagovat s buňkami:**
        1. **Ihned v prohlížeči (Živý běh):** Klikněte na tlačítko **Open in Colab** výše – sešit se otevře v bezplatném cloudovém prostředí Google Colab se všemi knihovnami a GPU/CPU výpočty.
        2. **Lokálně ve VS Code / JupyterLab:** Stáhněte soubor tlačítkem **Stáhnout .ipynb**.
        3. **Níže v aplikaci:** Zobrazeny jsou kompletní reálné výstupy, tabulky a vygenerované grafy z posledního běhu.
        """
    )

    # Vykreslení buněk sešitu
    for idx, cell in enumerate(nb_json.get("cells", [])):
        cell_type = cell.get("cell_type")
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue

        if cell_type == "markdown":
            st.markdown(source)
            st.write("")
        elif cell_type == "code":
            exec_count = cell.get("execution_count")
            exec_str = f"[{exec_count}]" if exec_count is not None else "[ ]"
            
            st.markdown(f"##### Buňka `In {exec_str}`:")
            st.code(source, language="python")

            outputs = cell.get("outputs", [])
            if outputs:
                with st.container():
                    for out in outputs:
                        out_type = out.get("output_type")
                        if out_type == "stream":
                            text = "".join(out.get("text", []))
                            st.caption("📋 Výstup konzole:")
                            st.code(text, language="text")
                        elif out_type in ("execute_result", "display_data"):
                            data = out.get("data", {})
                            if "image/png" in data:
                                img_bytes = base64.b64decode(data["image/png"])
                                st.image(img_bytes, caption=f"Graf z buňky In {exec_str}", use_container_width=True)
                            elif "text/html" in data:
                                html_str = "".join(data["text/html"])
                                st.components.v1.html(html_str, height=250, scrolling=True)
                            elif "text/plain" in data:
                                st.code("".join(data["text/plain"]), language="text")
            st.markdown("---")
