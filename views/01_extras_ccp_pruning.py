from pathlib import Path
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.title("🔬 Expertní analýza: Cost-Complexity Pruning (ccp_alpha)")
st.caption("Matematicky čisté prořezávání rozhodovacích stromů v Scikit-learn: Hledání optimálního parametru alpha pro minimalizaci chyby na testovacích datech.")

@st.cache_data
def load_ccp_data():
    base_dir = Path(__file__).resolve().parent.parent
    p = base_dir / "01_Regression" / "data" / "day1_extras_precomputed.json"
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data["ccp_results"])

df_ccp = load_ccp_data()

st.markdown(r"""
V Cvičeních 8 a 9 jsme strom omezovali předem (**Pre-pruning**) pomocí `max_depth` a `min_samples_leaf`.  
Algoritmus CART však nabízí mnohem elegantnější přístup: **Post-pruning (Minimal Cost-Complexity Pruning)**.

Účelová funkce penalizuje strom za celkový počet listů $|T|$:
$$R_\alpha(T) = R(T) + \alpha |T|$$
kde $R(T)$ je celková chyba MSE stromu a $\alpha \ge 0$ je komplexní koeficient penalizace (obdoba Lasso/Ridge pro stromy).
""")

fig_ccp = go.Figure()
fig_ccp.add_trace(go.Scatter(
    x=df_ccp["alpha"],
    y=df_ccp["r2_train"],
    mode="lines+markers",
    name="Trénovací R² (Train Score)",
    line=dict(color="#3B82F6", width=2.5)
))
fig_ccp.add_trace(go.Scatter(
    x=df_ccp["alpha"],
    y=df_ccp["r2_test"],
    mode="lines+markers",
    name="Testovací R² (Test Score / Generalizace)",
    line=dict(color="#10B981", width=3)
))

# Zvýraznění optimální alfy
opt_idx = df_ccp["r2_test"].idxmax()
opt_row = df_ccp.loc[opt_idx]

fig_ccp.add_vline(
    x=opt_row["alpha"],
    line_dash="dash",
    line_color="#EF4444",
    annotation_text=f"Optimum (alpha={opt_row['alpha']:.1f}, R²={opt_row['r2_test']:.4f})"
)

fig_ccp.update_layout(
    title="Vývoj Train a Test R² v závislosti na parametru ccp_alpha",
    xaxis_title="Hodnota penalizačního parametru ccp_alpha",
    yaxis_title="Koeficient determinace R²",
    height=500,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_ccp, width="stretch")

st.warning(
    "⚠️ **Metodická poznámka:** Křivka testovacího $R^2$ zde slouží k názorné pedagogické demonstraci bodu prořezání. "
    "V profesionální praxi se optimální `ccp_alpha` ladí výhradně křížovou validací na trénovacích datech (`GridSearchCV`), "
    "aby testovací sada zůstala nedotčená pro finální nestranné hodnocení."
)

st.markdown("### 🎚️ Prozkoumejte detaily prořezání:")
sel_alpha_slider = st.select_slider(
    "Vyberte hodnotu ccp_alpha:",
    options=df_ccp["alpha"].tolist(),
    value=opt_row["alpha"]
)

curr_row = df_ccp[df_ccp["alpha"] == sel_alpha_slider].iloc[0]

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Počet zbylých listů", f"{int(curr_row['n_leaves']):,}")
with c2:
    st.metric("Hloubka stromu", f"{int(curr_row['depth'])}")
with c3:
    st.metric("Testovací R²", f"{curr_row['r2_test']:.4f}")
with c4:
    st.metric("Testovací RMSE", f"{curr_row['rmse_test']:,.1f} USD")

st.info(
    "💡 **Poznatek:** Při $\\alpha = 0$ je strom obrovský a přeučený. S rostoucím $\\alpha$ "
    "odpadávají čistě šumové větve a **testovací $R^2$ roste až k optimu**. Pokud $\\alpha$ přeženeme, "
    "strom se smrskne na pár uzlů a nastává nedoučení (underfitting)."
)
