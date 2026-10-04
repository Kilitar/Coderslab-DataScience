"""
Den 3 Extras: ⚖️ Cost-Sensitive Decision Threshold kalkulačka (Heart Disease)
=============================================================================
- Řešení asymetrických nákladů (False Negative vs. False Positive) u diagnózy srdce.
- Využívá předpočítané testovací pravděpodobnosti Random Forest a XGBoost.
- Exaktní výpočet celkových finančních/zdravotních nákladů v závislosti na prahu T.
- Srovnání standardního prahu T = 0.5 s nákladově optimálním prahem T*.
- Teorie Bayesova rozhodovacího prahu a iso-cost tečny na ROC křivce.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_curve

st.set_page_config(page_title="Cost-Sensitive Threshold kalkulačka", page_icon="⚖️", layout="wide")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "03_Advanced_ML_Neural_Networks" / "data" / "day3_extras_heart_probs.json"

st.markdown("""
# ⚖️ Cost-Sensitive Decision Threshold kalkulačka
### Optimalizace rozhodovacího prahu pro lékařskou diagnózu nemocí srdce
""")

st.info(r"""
Většina vývojářů v ML automaticky používá výchozí rozhodovací práh **T = 0.5** ($P \ge 0.5 \implies$ Nemoc). 
V medicíně a bankovnictví je ale takový přístup kritickou chybou! 
**Neodhalit nemocného pacienta (False Negative) je dramaticky nebezpečnější a dražší** než poslat zdravého na kontrolní vyšetření (False Positive). 
Tento nástroj ti umožní najít nákladově optimální práh $T^*$, který minimalizuje celkovou újmu.
""")

# Načtení dat
@st.cache_data
def load_heart_probs():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

data = load_heart_probs()
y_test = np.array(data["y_test"])
proba_rf = np.array(data["proba_rf"])
proba_xgb = np.array(data["proba_xgb"])

# --- Ovládací prvky v hlavní ploše (nezasahuje do navigace v levém docku) ---
with st.expander("⚙️ Výběr modelu, nákladová matice & rozhodovací práh", expanded=True):
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.markdown("#### 1. Model")
        model_choice = st.radio("Porovnávaný model:", ["XGBoost Classifier", "Random Forest Classifier"], index=0)
        current_probs = proba_xgb if model_choice == "XGBoost Classifier" else proba_rf
        st.markdown("#### 3. Ruční práh")
        interactive_threshold = st.slider("Rozhodovací práh (T):", min_value=0.05, max_value=0.95, value=0.50, step=0.01)

    with col_p2:
        st.markdown("#### 2a. Náklady na chyby")
        cost_fn = st.number_input(
            "Náklad False Negative (FN):",
            value=60000,
            step=5000,
            help="Nemocný pacient poslán domů bez léčby (riziko infarktu, hospitalizace, komplikací).",
        )
        cost_fp = st.number_input(
            "Náklad False Positive (FP):",
            value=5000,
            step=500,
            help="Zdravý pacient poslán na zbytečné kontrolní vyšetření (např. ultrazvuk, zátěžové EKG).",
        )

    with col_p3:
        st.markdown("#### 2b. Náklady na správná určení")
        cost_tp = st.number_input(
            "Náklad True Positive (TP):",
            value=2000,
            step=500,
            help="Včasné odhalení nemoci a nasazení standardní levné medikace.",
        )
        cost_tn = st.number_input(
            "Náklad True Negative (TN):",
            value=0,
            step=100,
            help="Správné propuštění zdravého pacienta bez dalších nákladů.",
        )

# --- Výpočet nákladů pro všechny prahy ---
thresholds = np.linspace(0.02, 0.98, 193)
costs = []
fps = []
fns = []
tps = []
tns = []

for t in thresholds:
    y_pred = (current_probs >= t).astype(int)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    tot_cost = fp * cost_fp + fn * cost_fn + tp * cost_tp + tn * cost_tn
    costs.append(tot_cost)
    fps.append(fp)
    fns.append(fn)
    tps.append(tp)
    tns.append(tn)

costs = np.array(costs)
best_idx = np.argmin(costs)
best_t = thresholds[best_idx]
best_cost = costs[best_idx]

# Náklady pro T = 0.5
idx_05 = np.argmin(np.abs(thresholds - 0.50))
cost_05 = costs[idx_05]

# Teoretický Bayesovský práh
denom = (cost_fp - cost_tn) + (cost_fn - cost_tp)
bayes_t = (cost_fp - cost_tn) / denom if denom > 0 else 0.5

# --- Horní lišta výsledků ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Výchozí náklady (T = 0.5)", f"{cost_05:,.0f} Kč".replace(",", " "))
c2.metric("Optimální náklady (T*)", f"{best_cost:,.0f} Kč".replace(",", " "), delta=f"{(best_cost - cost_05):,.0f} Kč".replace(",", " "), delta_color="inverse")
c3.metric("Optimální práh T*", f"{best_t:.2f}", help="Práh, který minimalizuje celkové škody na testovací množině.")
c4.metric("Teoretický Bayesův práh", f"{bayes_t:.3f}", help="T_bayes = C_FP / (C_FP + C_FN)")

st.divider()

# --- GRAF NÁKLADOVÉ KŘIVKY ---
col_g1, col_g2 = st.columns([1.5, 1.0])

with col_g1:
    fig_cost = go.Figure()
    fig_cost.add_trace(go.Scatter(
        x=thresholds, y=costs, mode="lines",
        name="Celkové náklady (Expected Cost)",
        line=dict(color="#2563eb", width=3),
    ))
    # Bod T = 0.5
    fig_cost.add_trace(go.Scatter(
        x=[0.5], y=[cost_05], mode="markers",
        name="Standardní práh (T = 0.5)",
        marker=dict(color="#64748b", size=10, symbol="circle"),
    ))
    # Optimální bod T*
    fig_cost.add_trace(go.Scatter(
        x=[best_t], y=[best_cost], mode="markers",
        name=f"Optimální práh T* = {best_t:.2f}",
        marker=dict(color="#10b981", size=14, symbol="star"),
    ))
    # Ručně zvolený bod
    manual_idx = np.argmin(np.abs(thresholds - interactive_threshold))
    fig_cost.add_trace(go.Scatter(
        x=[interactive_threshold], y=[costs[manual_idx]], mode="markers",
        name=f"Váš práh T = {interactive_threshold:.2f}",
        marker=dict(color="#ef4444", size=11, symbol="diamond"),
    ))

    fig_cost.update_layout(
        title="Křivka celkových nákladů v závislosti na rozhodovacím prahu T",
        height=380,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(title="Rozhodovací práh T", range=[0.0, 1.0]),
        yaxis=dict(title="Celkové náklady v Kč"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_cost, width="stretch")

with col_g2:
    st.subheader("ROC křivka s optimálním bodem")
    fpr, tpr, _ = roc_curve(y_test, current_probs)
    
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC křivka", line=dict(color="#6366f1", width=2.5)))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Náhoda", line=dict(color="gray", dash="dash")))
    
    # Bod pro nejlepší práh
    best_fpr = fps[best_idx] / (fps[best_idx] + tns[best_idx])
    best_tpr = tps[best_idx] / (tps[best_idx] + fns[best_idx])
    fig_roc.add_trace(go.Scatter(
        x=[best_fpr], y=[best_tpr], mode="markers",
        name=f"Optimum T* ({best_t:.2f})",
        marker=dict(color="#10b981", size=12, symbol="star"),
    ))
    
    fig_roc.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title="False Positive Rate (1 - Specificity)"),
        yaxis=dict(title="True Positive Rate (Recall / Sensitivity)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_roc, width="stretch")

st.divider()

# --- SROVNÁNÍ MATIC ZÁMĚN ---
st.subheader("🔬 Srovnání matic záměn: Standardní práh (0.50) vs. Optimální práh (T*)")

y_pred_05 = (current_probs >= 0.50).astype(int)
y_pred_opt = (current_probs >= best_t).astype(int)

cm_05 = confusion_matrix(y_test, y_pred_05, labels=[0, 1])
cm_opt = confusion_matrix(y_test, y_pred_opt, labels=[0, 1])

c_cm1, c_cm2 = st.columns(2)

with c_cm1:
    st.markdown(f"#### Standardní práh T = 0.50 (Náklady: {cost_05:,.0f} Kč)")
    df_cm_05 = pd.DataFrame(
        cm_05,
        index=["Skutečně Zdravý (0)", "Skutečně Nemocný (1)"],
        columns=["Predikce Zdravý (0)", "Predikce Nemocný (1)"],
    )
    st.dataframe(df_cm_05, width="stretch")
    st.caption(f"⚠️ **Neodhalení nemocní (FN):** {cm_05[1, 0]} pacientů × {cost_fn:,} Kč = {cm_05[1, 0] * cost_fn:,} Kč.")

with c_cm2:
    st.markdown(f"#### Optimální práh T* = {best_t:.2f} (Náklady: {best_cost:,.0f} Kč)")
    df_cm_opt = pd.DataFrame(
        cm_opt,
        index=["Skutečně Zdravý (0)", "Skutečně Nemocný (1)"],
        columns=["Predikce Zdravý (0)", "Predikce Nemocný (1)"],
    )
    st.dataframe(df_cm_opt, width="stretch")
    st.caption(f"✅ **Neodhalení nemocní (FN) klesli na:** {cm_opt[1, 0]} pacientů! Celková úspora: **{(cost_05 - best_cost):,.0f} Kč**.")
