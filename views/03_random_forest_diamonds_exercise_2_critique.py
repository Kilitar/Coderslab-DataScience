import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# 1. Načtení předpočtených dat
base_dir = Path(__file__).resolve().parent.parent
json_path = base_dir / "03_Advanced_ML_Neural_Networks" / "data" / "diamonds_rf_exercise_2_precomputed.json"

st.title("🔬 Cvičení 2: Diamanty – Expertní analýza & Diagnostika")
st.caption("Kritické gemologické zhodnocení: Heteroskedasticita chyb, extrapolační strop u luxusních diamantů a dekorelace fyzikálních rozměrů.")

if not json_path.exists():
    st.error("Předpočtená data `diamonds_rf_exercise_2_precomputed.json` nebyla nalezena. Spusťte nejprve výpočetní skript.")
    st.stop()

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

meta = data["metadata"]
opt_model = data["optimal_model_test"]
feat_data = data["feature_importances"]
bin_data = pd.DataFrame(data["price_bin_performance"])

# =============================================================================
# 1. HETEROSKEDASTICITA CHYB DLE CENOVÝCH HLADIN
# =============================================================================
st.subheader("1. Kde model chybuje? Heteroskedasticita a růst chyb u drahých diamantů")
st.markdown(
    r"""
    Zadání vyčísluje celkovou testovací MAE na **268.21 USD**. Tento průměr však skrývá obrovskou nerovnoměrnost 
    v přesnosti ocenění mezi běžnými a investičními kameny:
    """
)

fig_bins = go.Figure()
fig_bins.add_trace(go.Bar(
    x=bin_data["Cenová kategorie"],
    y=bin_data["Průměrná MAE (USD)"],
    name="Průměrná MAE (USD)",
    text=[f"{v:.1f} USD" for v in bin_data["Průměrná MAE (USD)"]],
    textposition="auto",
    marker_color="#f59e0b"
))
fig_bins.add_trace(go.Bar(
    x=bin_data["Cenová kategorie"],
    y=bin_data["Medián MAE (USD)"],
    name="Medián MAE (USD)",
    text=[f"{v:.1f} USD" for v in bin_data["Medián MAE (USD)"]],
    textposition="auto",
    marker_color="#3b82f6"
))
fig_bins.update_layout(
    barmode="group",
    title="Chyba ocenění diamantu (MAE) podle cenových segmentů trhu",
    yaxis_title="Chyba predikce (USD)",
    height=340,
    margin=dict(l=30, r=30, t=35, b=30),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_bins, width="stretch")

st.info(
    r"""
    💡 **Analytické zjištění:**
    - U běžných diamantů do 1 000 USD je model mimořádně přesný (průměrná chyba je pouhých **62.5 USD** a medián **38 USD**).
    - U luxusních diamantů nad 9 000 USD však průměrná chyba exploduje na **více než 860 USD**!
    - **Důvod:** Exponenciální charakter cenové křivky diamantů. V profesionální praxi se doporučuje cíl modelovat jako $\log(\text{price})$ 
      a následně exponencielou vrátit do původního měřítka, což stabilizuje rozptyl reziduí napříč všemi cenami.
    """
)

st.divider()

# =============================================================================
# 2. PROBLÉM EXTRAPOLACE U VYSOKÝCH KARÁTŮ (EXTRAPOLATION CEILING)
# =============================================================================
st.subheader("2. Extrapolační strop: Proč Random Forest selže u velkých diamantů?")
st.markdown(
    r"""
    V trénovací sadě končí nejdražší diamant na hodnotě **18 823 USD** a hmotnosti okolo 5 karátů.
    - Každý list rozhodovacího stromu predikuje **konstantní průměr** pozorování, která do něj spadla.
    - Pokud přijde unikátní investiční diamant o váze 8 karátů s reálnou hodnotou 50 000 USD, 
      **Random Forest pro něj nikdy nepředpoví více než 18 823 USD**!
    - **Závěr:** Pro úlohy s otevřeným cenovým stropem nebo extrapolací v čase musí být Random Forest doplněn 
      o lineární nebo exponenciální trendovou složku.
    """
)

st.divider()

# =============================================================================
# 3. DŮLEŽITOST PŘÍZNAKŮ: MULTIKOLINEARITA A PERMUTAČNÍ VÝZNAM
# =============================================================================
st.subheader("3. Důležitost příznaků: MDI vs. Permutační důležitost na testovacích datech")
st.markdown(
    r"""
    Fyzické rozměry diamantu $x, y, z$ mají s hmotností v karátech (`carat`) korelaci $r > 0.97$. 
    Podívejme se, jak se s touto kolinearitou vyrovnal Random Forest:
    """
)

df_feat_diam = pd.DataFrame({
    "Příznak": feat_data["features"],
    "MDI Importance (Trénink)": feat_data["mdi"],
    "Permutation Importance (Test)": feat_data["permutation_mean"],
    "Permutation Std": feat_data["permutation_std"]
}).sort_values(by="Permutation Importance (Test)", ascending=True)

fig_feat_diam = go.Figure()
fig_feat_diam.add_trace(go.Bar(
    y=df_feat_diam["Příznak"],
    x=df_feat_diam["MDI Importance (Trénink)"],
    name="MDI (Gini / MSE pokles)",
    orientation="h",
    marker_color="#94a3b8"
))
fig_feat_diam.add_trace(go.Bar(
    y=df_feat_diam["Příznak"],
    x=df_feat_diam["Permutation Importance (Test)"],
    name="Permutační významnost (Test MAE)",
    orientation="h",
    marker_color="#10b981"
))
fig_feat_diam.update_layout(
    barmode="group",
    title="Důležitost příznaků diamantů: MDI vs. Permutační dopad na testovací chybu",
    xaxis_title="Relativní významnost příznaku",
    height=440,
    margin=dict(l=30, r=30, t=35, b=30),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_feat_diam, width="stretch")

st.success(
    r"""
    #### 💎 Gemologická interpretace:
    1. **Hmotnost `carat` a rozměry `y`, `x`:** Zodpovídají za více než 85 % veškeré predikční síly. Zamíchání karátů zvedne chybu modelu o stovky USD.
    2. **Čistota (`clarity`) a Barva (`color`):** Jsou druhým nejdůležitějším faktorem určujícím prémiovou přirážku ke kameni stejné velikosti.
    3. **Kvalita brusu (`cut`) a tabulka (`table`):** Mají nejnižší přímý permutační dopad, protože špatný brus diamant spíše znehodnotí v okrajových případech, ale na základní cenotvorbu má menší vliv než karáty a čistota.
    """
)
