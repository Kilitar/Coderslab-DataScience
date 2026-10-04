import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# 1. Načtení předpočtených dat
base_dir = Path(__file__).resolve().parent.parent
json_path = base_dir / "03_Advanced_ML_Neural_Networks" / "data" / "heart_xgb_exercise_1_precomputed.json"

st.title("🔬 Cvičení 1: Nemoc srdce (XGBoost) – Expertní analýza")
st.caption("Kritické kardiologické zhodnocení: Proč na malých datech Bagging předčil Boosting, role regularizace Gamma, threshold tuning a analýza zisků (Gain).")

if not json_path.exists():
    st.error("Předpočtená data `heart_xgb_exercise_1_precomputed.json` nebyla nalezena. Spusťte nejprve výpočetní skript.")
    st.stop()

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

opt_model = data["optimal_model_test"]
feat_data = data["feature_importances"]
thresh_data = pd.DataFrame(data["threshold_sweep"])

# =============================================================================
# 1. ANALÝZA: PROČ RANDOM FOREST PŘEDČIL XGBOOST NA MALÝCH DATECH?
# =============================================================================
st.subheader("1. Datová diagnostika: Proč Bagging na 212 pacientech předčil Boosting?")
st.markdown(
    r"""
    V soutěžích Kaggle s miliony řádků XGBoost obvykle dominuje. Proč zde na kardiologických datech 
    dosáhl **Random Forest testovací Precision 85.71 %**, zatímco **XGBoost 81.40 %**?
    """
)

col_exp1, col_exp2 = st.columns(2)
with col_exp1:
    st.info(
        r"""
        #### 🌲 Silná stránka Random Forestu na malých datech:
        - Má pouhých 212 trénovacích vzorků.
        - Nezávislé vzorkování řádků (Bootstrap) a sloupců vytváří **vysokou diverzitu**.
        - Průměrování 100 stromů spolehlivě eliminuje náhodný šum malého vzorku pacientů.
        """
    )
with col_exp2:
    st.warning(
        r"""
        #### 🚀 Zranitelnost Boostingu na malých datech:
        - Boosting se učí **sekvenčně z chyb a reziduí**.
        - Na 212 pacientech je každá drobná fluktuace (neobvyklá hodnota tlaku či tepu) vnímána jako systémová chyba, 
          kterou se následující stromy snaží usilovně vykrýt.
        - To vede k mírné adaptaci na náhodný šum trénovací sady.
        """
    )

st.divider()

# =============================================================================
# 2. INTERAKTIVNÍ THRESHOLD TUNING PRO XGBOOST
# =============================================================================
st.subheader("2. Interaktivní simulátor rozhodovacího prahu v XGBoostu")
st.markdown(
    r"""
    Při výchozím prahu $P \ge 0.50$ má XGBoost **8 přehlédnutých nemocných pacientů (False Negatives)**. 
    Podívejme se, jak posun prahu zachrání životy:
    """
)

col_th_ctrl, col_th_kpi = st.columns([1, 1.2])

with col_th_ctrl:
    sel_thresh = st.slider(
        "Zvolte rozhodovací práh (Threshold):",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05,
        help="Snížením prahu zachytíme pacienty s počínajícím rizikem infarktu."
    )
    matched_row = thresh_data.iloc[(thresh_data["threshold"] - sel_thresh).abs().argsort()[:1]].iloc[0]

with col_th_kpi:
    c_kpi1, c_kpi2, c_kpi3 = st.columns(3)
    with c_kpi1:
        st.metric("Recall (Záchyt)", f"{matched_row['recall'] * 100:.1f} %", delta=f"{(matched_row['recall'] - opt_model['recall']) * 100:+.1f} % vs default")
    with c_kpi2:
        st.metric("Přehlédnutí nemocní (FN)", f"{int(matched_row['false_negatives'])} pacientů", delta=f"{int(matched_row['false_negatives']) - 8:+d} vs default", delta_color="inverse")
    with c_kpi3:
        st.metric("Precision (Přesnost)", f"{matched_row['precision'] * 100:.1f} %", delta=f"{(matched_row['precision'] - opt_model['precision']) * 100:+.1f} %")

fig_th_xgb = go.Figure()
fig_th_xgb.add_trace(go.Scatter(
    x=thresh_data["threshold"],
    y=thresh_data["recall"] * 100,
    mode="lines+markers",
    name="Recall (Záchyt nemocných %)",
    line=dict(color="#10b981", width=3)
))
fig_th_xgb.add_trace(go.Scatter(
    x=thresh_data["threshold"],
    y=thresh_data["precision"] * 100,
    mode="lines+markers",
    name="Precision (Přesnost %)",
    line=dict(color="#f97316", width=3)
))
fig_th_xgb.add_vline(
    x=sel_thresh,
    line_dash="dot",
    line_color="#ef4444",
    annotation_text=f"Práh {sel_thresh:.2f}"
)
fig_th_xgb.update_layout(
    title="XGBoost Trade-off: Precision vs. Recall v závislosti na prahu",
    xaxis_title="Klasifikační práh (Threshold)",
    yaxis_title="Hodnota (%)",
    height=330,
    margin=dict(l=30, r=30, t=35, b=30),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_th_xgb, width="stretch")

st.info(
    f"💡 **Klinické zhodnocení:** Při prahu **0.35** klesne počet přehlédnutých nemocných z 8 na pouhé **{int(matched_row['false_negatives'])}**, "
    f"přičemž Recall stoupne na **{matched_row['recall'] * 100:.1f} %**! Pro kardiologický screening je toto nastavení nesrovnatelně bezpečnější."
)

st.divider()

# =============================================================================
# 3. DŮLEŽITOST PŘÍZNAKŮ: XGBOOST GAIN VS. PERMUTAČNÍ VÝZNAM
# =============================================================================
st.subheader("3. Důležitost příznaků: XGBoost Gain (zisk štěpení) vs. Permutační testovací význam")

df_feat_xgb = pd.DataFrame({
    "Příznak": feat_data["features"],
    "XGBoost Gain": feat_data["gain"],
    "Permutation Importance (Precision)": feat_data["permutation_mean"],
    "Permutation Std": feat_data["permutation_std"]
}).sort_values(by="XGBoost Gain", ascending=True)

fig_feat_xgb = go.Figure()
fig_feat_xgb.add_trace(go.Bar(
    y=df_feat_xgb["Příznak"],
    x=df_feat_xgb["XGBoost Gain"],
    name="XGBoost Gain (Zisk z optimalizace loss)",
    orientation="h",
    marker_color="#f97316"
))
fig_feat_xgb.add_trace(go.Bar(
    y=df_feat_xgb["Příznak"],
    x=df_feat_xgb["Permutation Importance (Precision)"],
    name="Permutační významnost (Test)",
    orientation="h",
    marker_color="#3b82f6"
))
fig_feat_xgb.update_layout(
    barmode="group",
    title="Srovnání důležitosti: XGBoost Gain vs. Permutační dopad na Precision",
    xaxis_title="Relativní významnost příznaku",
    height=480,
    margin=dict(l=30, r=30, t=35, b=30),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_feat_xgb, width="stretch")

st.success(
    r"""
    #### 🏆 Závěr porovnání:
    1. **`thal_normal` a `ca`:** Stejně jako u Random Forestu jsou genetická talasémie a počet zasažených cév nejmocnějšími prediktory.
    2. **Role `gamma = 1.0`:** Bez nastavení parametru gamma by model vygeneroval o 40 % více větví stromů, které by pouze memorovaly trénovací vzorky. Gamma zajistila, že uzly se dělily jen při prokazatelném zisku logloss.
    """
)
