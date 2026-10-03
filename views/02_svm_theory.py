import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# 1. Načtení dat a teorie
base_dir = Path(__file__).resolve().parent.parent
theory_md_path = base_dir / "02_Classification" / "theory" / "06_svm_theory.md"
json_path = base_dir / "02_Classification" / "data" / "svm_theory_precomputed.json"
plot_kernel_path = base_dir / "02_Classification" / "plots" / "svm_kernel_comparison.png"
plot_grid_path = base_dir / "02_Classification" / "plots" / "svm_c_gamma_grid.png"

st.title("📚 Support Vector Machines (SVM): Teoretický rozbor")
st.caption("Podpůrné vektory, maximální okraj (Margin), nadroviny v n-dimenzích, analogie s dekou, Jádrový trik (Kernel Trick) a multiclass strategie OvO.")

# Záložky pro přehlednost: Teorie a Interaktivní laboratoř
tab_theory, tab_lab = st.tabs(["📖 Teoretický výklad", "🔬 Interaktivní laboratoř & Vizualizace"])

with tab_theory:
    if theory_md_path.exists():
        with open(theory_md_path, "r", encoding="utf-8") as f:
            content = f.read()
        st.markdown(content)
    else:
        st.warning("Teoretický dokument 06_svm_theory.md nebyl nalezen.")

with tab_lab:
    st.subheader("1. Vizuální demonstrace jader: Lineární vs. Nelineární data")
    st.markdown(r"""
Data uspořádaná do soustředných kružnic nelze v původním 2D prostoru oddělit přímkou:
- **Lineární jádro (`linear`):** Selhává (přesnost pouze $\approx 42\,\%$), potřebuje téměř všechny body jako podpůrné vektory.
- **Polynomiální jádro (`poly`, stupeň 2) a RBF jádro (`rbf`):** Dosahují dokonalé **$100\,\%$ přesnosti** při použití pouhých 24 až 36 podpůrných vektorů!
    """)

    if plot_kernel_path.exists():
        st.image(str(plot_kernel_path), caption="Srovnání rozhodovacích hranic a podpůrných vektorů pro 4 základní jádra SVM", width="stretch")

    st.divider()

    # 3D Blanket Analogy
    st.subheader("2. Interaktivní 3D model: „Analogie s dekou“ (The Blanket Analogy)")
    st.markdown(r"""
Podle přednášky v kurzu si představte nelineární 2D data jako barevné míčky na dece. Když deku prudce zvedneme, míčky vyletí do vzduchu (3. dimenze $z = x_1^2 + x_2^2$).
V tomto 3D prostoru je pak snadné vložit rovnou dělící plochu (deku), která červené body oddělí od modrých!
""")

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            svm_data = json.load(f)

        pts_3d = pd.DataFrame(svm_data["blanket_analogy_3d"])
        fig_3d = px.scatter_3d(
            pts_3d,
            x="x1",
            y="x2",
            z="z",
            color=pts_3d["class"].astype(str),
            color_discrete_map={"0": "#1f77b4", "1": "#d62728"},
            labels={"x1": "Příznak x1", "x2": "Příznak x2", "z": "Transformace z = x1² + x2²", "color": "Třída"},
            title="Mapování 2D dat do 3D prostoru (Lineární separabilita nadrovinou z = 0.4)"
        )
        fig_3d.update_traces(marker=dict(size=4, opacity=0.85))

        # Dělící rovina (deka) z = 0.4
        x_grid = np.linspace(-1.2, 1.2, 10)
        y_grid = np.linspace(-1.2, 1.2, 10)
        X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
        Z_mesh = np.full_like(X_mesh, 0.4)

        fig_3d.add_trace(go.Surface(
            x=X_mesh, y=Y_mesh, z=Z_mesh,
            colorscale=[[0, "rgba(0,200,0,0.3)"], [1, "rgba(0,200,0,0.3)"]],
            showscale=False,
            name="Dělící rovina (Deka: z = 0.4)"
        ))

        fig_3d.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            scene=dict(
                xaxis_title="x1",
                yaxis_title="x2",
                zaxis_title="z = x1² + x2²"
            )
        )
        st.plotly_chart(fig_3d, width="stretch")

    st.divider()

    # Vliv C a gamma
    st.subheader("3. Vliv hyperparametrů C a gamma na RBF jádro")
    st.markdown(r"""
- **Hyperparametr $C$ (penalizace za chyby):**  
  Větší $C$ $\implies$ tvrdší okraj, nulová tolerance chyb na trénovacích datech $\implies$ riziko **přeučení (overfitting)**.
- **Hyperparametr $\gamma$ (dosah vlivu bodů):**  
  Větší $\gamma$ $\implies$ každý bod vytváří strmou izolovanou Gaussovskou špičku $\implies$ extrémně členitá hranice.
""")

    if plot_grid_path.exists():
        st.image(str(plot_grid_path), caption="Mřížka 4x4: Vývoj tvaru hranice RBF jádra pro různé hodnoty C a gamma", width="stretch")

    st.divider()

    # OvO Kalkulátor
    st.subheader("4. Kalkulátor strategie One-vs-One (OvO) pro vícedenní klasifikaci")
    st.markdown(r"""
SVM je od základu binární model. Pro $K$ tříd vytváří strategie **One-vs-One (OvO)** binární klasifikátor pro každou dvojici:
$$N_{\text{models}} = \frac{K(K - 1)}{2}$$
""")

    k_classes = st.slider("Zvolte počet tříd ($K$):", min_value=2, max_value=20, value=3, step=1)
    n_models = int(k_classes * (k_classes - 1) / 2)

    c1, c2 = st.columns(2)
    c1.metric("Počet tříd ($K$)", f"{k_classes}")
    c2.metric("Počet potřebných binárních modelů", f"{n_models}", help="Tolik modelů natrénuje Scikit-learn v pozadí.")

    if k_classes == 3:
        st.info("🐧 Příklad z našeho kurzu: Pro **3 druhy tučňáků** (Adelie, Chinstrap, Gentoo) vytváří `SVC` přesně **3 binární modely**.")
    elif k_classes == 10:
        st.info("🔢 Příklad ze slajdů kurzu: Pro rozpoznávání **10 číslic** (0–9) vytváří `SVC` celkem **45 binárních modelů**.")
