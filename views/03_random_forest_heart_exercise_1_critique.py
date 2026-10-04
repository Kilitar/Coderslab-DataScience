import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# 1. Načtení předpočtených dat
base_dir = Path(__file__).resolve().parent.parent
json_path = base_dir / "03_Advanced_ML_Neural_Networks" / "data" / "heart_rf_exercise_1_precomputed.json"

st.title("🔬 Cvičení 1: Nemoc srdce – Expertní analýza & Diagnostika")
st.caption("Kritické kardiologické zhodnocení: Etické a klinické důsledky metriky Precision, interaktivní ladění prahu (Threshold Tuning), ROC/PR křivky a Permutační důležitost.")

if not json_path.exists():
    st.error("Předpočtená data `heart_rf_exercise_1_precomputed.json` nebyla nalezena. Spusťte nejprve výpočetní skript.")
    st.stop()

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

opt_model = data["optimal_model_test"]
feat_data = data["feature_importances"]
thresh_data = pd.DataFrame(data["threshold_sweep"])

# =============================================================================
# HORNÍ VÝSTRAHA: KLINICKÉ DILEMA
# =============================================================================
st.warning(
    r"""
    ### 🩺 Kardiologické dilema: Proč je optimalizace výhradně na Precision v medicíně riskantní?
    Zadání kurzu požadovalo maximalizaci metriky **Precision** ($\frac{\text{TP}}{\text{TP} + \text{FP}}$).
    - **Vysoká Precision** znamená, že pokud model označí pacienta jako nemocného, je to téměř jisté (málo falešných poplachů).
    - **Kritická odvrácená strana:** Model se stane příliš konzervativním. Raději pacienta neoznačí, pokud si není 100% jistý.
    - **Důsledek:** Vzniká **7 falešně negativních případů (False Negatives – FN)**. To znamená, že **7 pacientů se skutečným onemocněním srdce bylo posláno domů jako „zdraví“** s rizikem infarktu!
    """
)

st.divider()

# =============================================================================
# 1. INTERAKTIVNÍ SIMULÁTOR ROZHODOVACÍHO PRAHU (THRESHOLD TUNING)
# =============================================================================
st.subheader("1. Interaktivní simulátor rozhodovacího prahu (Threshold Tuning)")
st.markdown(
    r"""
    Standardní binární klasifikátor rozhoduje při prahu $P \ge 0.50$. V kardiologii však lékaři raději provedou jedno doplňující 
    bezpečné vyšetření navíc (vyšší FP), než aby přehlédli infarkt (minimalizace FN).
    
    Vyzkoušejte si, jak změna klasifikačního prahu ovlivní záchyt nemocných:
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
        help="Snížením prahu zachytíme více nemocných pacientů za cenu mírného nárůstu falešných poplachů."
    )
    # Vyhledání nejbližšího řádku v předpočteném threshold sweepu
    matched_row = thresh_data.iloc[(thresh_data["threshold"] - sel_thresh).abs().argsort()[:1]].iloc[0]

with col_th_kpi:
    c_kpi1, c_kpi2, c_kpi3 = st.columns(3)
    with c_kpi1:
        st.metric("Recall (Záchyt nemocných)", f"{matched_row['recall'] * 100:.1f} %", delta=f"{(matched_row['recall'] - opt_model['recall']) * 100:+.1f} % vs default")
    with c_kpi2:
        st.metric("Přehlédnutí nemocní (FN)", f"{int(matched_row['false_negatives'])} pacientů", delta=f"{int(matched_row['false_negatives']) - 7:+d} vs default", delta_color="inverse")
    with c_kpi3:
        st.metric("Precision (Přesnost diagnózy)", f"{matched_row['precision'] * 100:.1f} %", delta=f"{(matched_row['precision'] - opt_model['precision']) * 100:+.1f} %")

# Křivka Precision vs Recall vs Threshold
fig_th = go.Figure()
fig_th.add_trace(go.Scatter(
    x=thresh_data["threshold"],
    y=thresh_data["recall"] * 100,
    mode="lines+markers",
    name="Recall (Záchyt nemocných %)",
    line=dict(color="#10b981", width=3)
))
fig_th.add_trace(go.Scatter(
    x=thresh_data["threshold"],
    y=thresh_data["precision"] * 100,
    mode="lines+markers",
    name="Precision (Přesnost %)",
    line=dict(color="#3b82f6", width=3)
))
fig_th.add_trace(go.Scatter(
    x=thresh_data["threshold"],
    y=thresh_data["f1_score"] * 100,
    mode="lines",
    name="F1-Score (%)",
    line=dict(color="#f59e0b", width=2, dash="dash")
))
fig_th.add_vline(
    x=sel_thresh,
    line_dash="dot",
    line_color="#ef4444",
    annotation_text=f"Práh {sel_thresh:.2f}"
)
fig_th.update_layout(
    title="Trade-off: Precision, Recall a F1 v závislosti na klasifikačním prahu",
    xaxis_title="Klasifikační práh (Threshold)",
    yaxis_title="Hodnota (%)",
    height=340,
    margin=dict(l=30, r=30, t=35, b=30),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_th, width="stretch")

st.info(
    f"💡 **Klinické doporučení:** Při prahu **0.35** klesne počet přehlédnutých nemocných ze 7 na pouhé **{int(matched_row['false_negatives'])}**, "
    f"přičemž Recall stoupne na **{matched_row['recall'] * 100:.1f} %**. V medicíně se tak zachrání lidské životy s minimálními dodatečnými náklady na diagnostiku!"
)

st.divider()

# =============================================================================
# 2. DIAGNOSTICKÉ KŘIVKY: ROC-AUC A PRECISION-RECALL AUC
# =============================================================================
st.subheader("2. Komplexní diagnostika: ROC křivka vs. Precision-Recall křivka")

col_roc, col_pr = st.columns(2)

with col_roc:
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=opt_model["fpr"],
        y=opt_model["tpr"],
        mode="lines",
        name=f"Random Forest (AUC = {opt_model['roc_auc']:.3f})",
        line=dict(color="#3b82f6", width=3)
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Náhodný klasifikátor (AUC = 0.500)",
        line=dict(color="#9ca3af", dash="dash")
    ))
    fig_roc.update_layout(
        title="ROC křivka (Receiver Operating Characteristic)",
        xaxis_title="False Positive Rate (1 - Specificita)",
        yaxis_title="True Positive Rate (Senzitivita / Recall)",
        height=320,
        margin=dict(l=30, r=30, t=35, b=30)
    )
    st.plotly_chart(fig_roc, width="stretch")

with col_pr:
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(
        x=opt_model["pr_recall"],
        y=opt_model["pr_precision"],
        mode="lines",
        name=f"Random Forest (PR-AUC = {opt_model['pr_auc']:.3f})",
        line=dict(color="#10b981", width=3)
    ))
    fig_pr.update_layout(
        title="Precision-Recall křivka (PR Curve)",
        xaxis_title="Recall (Záchyt)",
        yaxis_title="Precision (Přesnost)",
        height=320,
        margin=dict(l=30, r=30, t=35, b=30)
    )
    st.plotly_chart(fig_pr, width="stretch")

st.divider()

# =============================================================================
# 3. DŮLEŽITOST PŘÍZNAKŮ: MDI VS. PERMUTAČNÍ DŮLEŽITOST
# =============================================================================
st.subheader("3. Odhalení zkreslení: MDI (Gini) vs. Permutační důležitost na testovacích datech")
st.markdown(
    r"""
    Výchozí `rf.feature_importances_` (MDI – Mean Decrease in Impurity) bývá zkreslená ve prospěch spojitých veličin 
    s mnoha unikátními hodnotami. Porovnejme ji s **Permutační důležitostí** na nezávislých testovacích datech:
    """
)

df_feat = pd.DataFrame({
    "Příznak": feat_data["features"],
    "MDI Importance (Trénink)": feat_data["mdi"],
    "Permutation Importance (Test)": feat_data["permutation_mean"],
    "Permutation Std": feat_data["permutation_std"]
}).sort_values(by="Permutation Importance (Test)", ascending=True)

fig_feat_comp = go.Figure()
fig_feat_comp.add_trace(go.Bar(
    y=df_feat["Příznak"],
    x=df_feat["MDI Importance (Trénink)"],
    name="MDI (Gini pokles nečistoty)",
    orientation="h",
    marker_color="#3b82f6"
))
fig_feat_comp.add_trace(go.Bar(
    y=df_feat["Příznak"],
    x=df_feat["Permutation Importance (Test)"],
    name="Permutační důležitost (Test)",
    orientation="h",
    marker_color="#10b981"
))
fig_feat_comp.update_layout(
    barmode="group",
    title="Srovnání důležitosti příznaků: MDI vs. Permutační testovací významnost",
    xaxis_title="Relativní významnost příznaku",
    height=480,
    margin=dict(l=30, r=30, t=35, b=30),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_feat_comp, width="stretch")

st.success(
    r"""
    #### 🏆 Klíčové lékařské zjištění:
    1. **Dominantní prediktor `ca` (počet zasažených hlavních cév při fluoroskopii):** Jednoznačně nejsilnější marker onemocnění srdce na obou škálách.
    2. **Talasémie (`thal_normal`, `thal_reversable`):** Genetická krevní anomálie je druhým nejsilnějším rizikovým faktorem.
    3. **Falešné bezpečí u spojitých proměnných:** MDI přisuzovala vysokou váhu cholesterolu (`chol`) a tlaku (`restbp`), avšak na testovací sadě jejich permutační význam klesá téměř k nule. Zátěžová ST deprese (`oldpeak`) a maximální tep (`maxhr`) jsou pro záchyt infarktu podstatně průkaznější!
    """
)
