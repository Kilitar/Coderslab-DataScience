import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# 1. Načtení předpočtených dat
base_dir = Path(__file__).resolve().parent.parent
json_path = base_dir / "02_Classification" / "data" / "lumbar_logistic_exercise_1_precomputed.json"

st.title("🎯 Cvičení 1: Bederní páteř – Logistická regrese & Ladění C")
st.caption("Splnění všech 8 kroků zadání: Načtení lumbar_df_normalized.csv, rozdělení 75/25, výchozí Precision a optimalizace regularizace C.")

if not json_path.exists():
    st.error("Předpočtená data `lumbar_logistic_exercise_1_precomputed.json` nebyla nalezena. Spusťte nejprve výpočetní skript.")
    st.stop()

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

meta = data["metadata"]
head_10 = pd.DataFrame(data["head_10"])
base_m = data["baseline_model"]
opt_m = data["optimal_model_c5"]
l1_m = data["l1_model"]
c_sweep = pd.DataFrame(data["c_sweep"])
coef_df = pd.DataFrame(data["coefficients"])

# =============================================================================
# KROK 1 & 2: NAČTENÍ DAT A KONTROLA PRVNÍCH 10 POZOROVÁNÍ
# =============================================================================
st.subheader("1. & 2. Načtení dat a kontrola prvních 10 pozorování (`head(10)`)")
st.markdown(r"""
Normalizovaný dataset biomechanických měření pánve obsahuje **310 pacientů** a **6 anatomických úhlů**:
- `pelvic_incidence`, `pelvic_tilt numeric`, `lumbar_lordosis_angle`, `sacral_slope`, `pelvic_radius`, `degree_spondylolisthesis`.
- Cílová proměnná `class`: `0` = Normal (zdravý pacient, 100 vzorků), `1` = Abnormal (patologie – výhřez / posun obratle, 210 vzorků).
""")

st.dataframe(head_10, width="stretch", hide_index=True)
st.caption(f"Celkem: {meta['samples_total']} pacientů | Trénovací sada (75 %): {meta['samples_train']} | Testovací sada (25 %): {meta['samples_test']} (`random_state=42`).")

st.divider()

# =============================================================================
# KROK 3 AŽ 7: VÝCHOZÍ MODEL (BASELINE C=1.0)
# =============================================================================
st.subheader("3.–7. Výchozí model logistické regrese bez ladění (Baseline)")
st.markdown(r"""
Model `LogisticRegression(random_state=42)` s výchozí $L_2$ regularizací ($C=1{,}0$, solver `lbfgs`) natrénovaný na `X_train`:
""")

m_col1, m_col2, m_col3, m_col4 = st.columns(4)
m_col1.metric("Výchozí Precision", f"{base_m['precision']*100:.2f} %")
m_col2.metric("Výchozí Accuracy", f"{base_m['accuracy']*100:.2f} %")
m_col3.metric("Výchozí Recall", f"{base_m['recall']*100:.2f} %")
m_col4.metric("Výchozí F1-score", f"{base_m['f1_score']*100:.2f} %")

col_cm_base, col_info_base = st.columns([1, 1])

with col_cm_base:
    st.markdown("#### Matice záměn výchozího modelu ($C=1{,}0$)")
    cm_b = np.array(base_m["confusion_matrix"])
    labels = ["Normal (0)", "Abnormal (1)"]
    fig_cm_b = px.imshow(
        cm_b,
        labels=dict(x="Predikovaná třída", y="Skutečná třída", color="Počet"),
        x=labels,
        y=labels,
        text_auto=True,
        color_continuous_scale="Blues"
    )
    fig_cm_b.update_layout(margin=dict(l=30, r=30, t=30, b=30), height=300)
    st.plotly_chart(fig_cm_b, width="stretch")

with col_info_base:
    st.markdown("#### Diagnostika výchozího stavu:")
    st.markdown(fr"""
    - **True Positives (TP):** `{cm_b[1][1]}` správně odhalených pacientů s patologií páteře.
    - **False Positives (FP – Falešné poplachy):** `{cm_b[0][1]}` zdravých lidí označeno za nemocné.
    - **False Negatives (FN – Přehlédnutí):** `{cm_b[1][0]}` nemocných propuštěno jako zdraví.
    - **Výchozí přesnost pozitivních predikcí (Precision):**  
      $$\text{{Precision}} = \frac{{\text{{TP}}}}{{\text{{TP}} + \text{{FP}}}} = \frac{{{cm_b[1][1]}}}{{{cm_b[1][1]} + {cm_b[0][1]}}} = \frac{{{cm_b[1][1]}}}{{{cm_b[1][1] + cm_b[0][1]}}} = {base_m['precision']*100:.2f}\,\%$$
    """)
    st.warning("13 zdravých pacientů (z 21 v testovací sadě) dostalo falešný poplach. Cílem cvičení je snížit počet FP a zvýšit Precision!")

st.divider()

# =============================================================================
# KROK 8: EXPERIMENT S REGULARIZACÍ C (LADĚNÍ HYPERPARAMETRŮ)
# =============================================================================
st.subheader("8. Experiment s hyperparametry: Ladění síly regularizace $C$")
st.markdown(r"""
Parametr $C$ je **inverzní k síle regularizace** ($C = \frac{1}{\lambda}$):
- **Malé $C$ ($C \to 0$):** Extrémně silná regularizace. Váhy jsou stlačeny k nule $\implies$ model je zploštělý a predikuje převážně většinovou třídu (Recall 100 %, ale Precision padá na $73{,}08\,\%$).
- **Optimální $C$ ($C = 5{,}0$):** Uvolnění vah umožní modelu přesněji vymezit hranici v prostoru anatomických úhlů $\implies$ **počet falešných poplachů (FP) klesá z 13 na pouhých 6** a **Precision roste na $88{,}89\,\%$**!
""")

# Interaktivní výběr C
c_options = [float(x) for x in c_sweep["C"].tolist()]
selected_c = st.select_slider(
    "Zvolte hodnotu hyperparametru C pro zobrazení modelu:",
    options=c_options,
    value=5.0
)

sel_row = c_sweep[c_sweep["C"] == selected_c].iloc[0]

c_col1, c_col2, c_col3, c_col4 = st.columns(4)
c_col1.metric("Precision (Přesnost)", f"{sel_row['Precision']*100:.2f} %" if "Precision" in sel_row else f"{sel_row['precision']*100:.2f} %", delta=f"{(sel_row['precision'] - base_m['precision'])*100:+.2f} % vs Baseline")
c_col2.metric("Accuracy (Celková)", f"{sel_row['accuracy']*100:.2f} %", delta=f"{(sel_row['accuracy'] - base_m['accuracy'])*100:+.2f} %")
c_col3.metric("Recall (Senzitivita)", f"{sel_row['recall']*100:.2f} %", delta=f"{(sel_row['recall'] - base_m['recall'])*100:+.2f} %")
c_col4.metric("Falešné poplachy (FP)", f"{int(sel_row['fp'])}", delta=f"{int(sel_row['fp'] - cm_b[0][1])} případů", delta_color="inverse")

# Křivka Precision vs C
fig_sweep = go.Figure()
fig_sweep.add_trace(go.Scatter(x=c_sweep["C"], y=c_sweep["precision"], mode="lines+markers", name="Precision (Přesnost)", line=dict(color="#1f77b4", width=3)))
fig_sweep.add_trace(go.Scatter(x=c_sweep["C"], y=c_sweep["accuracy"], mode="lines+markers", name="Accuracy", line=dict(color="#2ca02c", width=2, dash="dash")))
fig_sweep.add_trace(go.Scatter(x=c_sweep["C"], y=c_sweep["recall"], mode="lines+markers", name="Recall (Senzitivita)", line=dict(color="#ff7f0e", width=2, dash="dot")))

fig_sweep.add_vline(x=1.0, line_color="gray", line_dash="dash", annotation_text="Baseline C = 1.0 (80.0 %)")
fig_sweep.add_vline(x=5.0, line_color="red", line_dash="dot", annotation_text="Optimum C = 5.0 (88.9 %)")

fig_sweep.update_layout(
    title="Křivky klasifikačních metrik v závislosti na parametru C (log scale)",
    xaxis_title="Hyperparametr C (menší = silnější regularizace)",
    xaxis_type="log",
    yaxis_title="Hodnota metriky",
    yaxis=dict(range=[0.65, 1.02]),
    height=400,
    margin=dict(l=40, r=40, t=40, b=40)
)
st.plotly_chart(fig_sweep, width="stretch")

st.divider()

# =============================================================================
# KROK 8 (BONUS): SROVNÁNÍ MODELŮ A VÝBĚR PŘÍZNAKŮ L1 LASSO
# =============================================================================
st.subheader("Pokročilý rozbor: Srovnání Baseline ($C=1{,}0$), Optimal ($C=5{,}0$) a $L_1$ Lasso")

col_tab, col_summary = st.columns([1, 1])

with col_tab:
    st.markdown("#### Srovnání vah příznaků $\\beta$:")
    st.dataframe(coef_df, width="stretch", hide_index=True)

with col_summary:
    st.markdown("#### Shrnutí výsledků experimentu:")
    st.markdown(fr"""
    | Metrika / Model | Výchozí ($C=1.0$) | Optimální ($C=5.0$) | $L_1$ Lasso (`saga`) |
    | :--- | :--- | :--- | :--- |
    | **Precision** | **{base_m['precision']*100:.2f} %** | **{opt_m['precision']*100:.2f} %** | **{l1_m['precision']*100:.2f} %** |
    | **Accuracy** | {base_m['accuracy']*100:.2f} % | {opt_m['accuracy']*100:.2f} % | {l1_m['accuracy']*100:.2f} % |
    | **Recall** | {base_m['recall']*100:.2f} % | {opt_m['recall']*100:.2f} % | {l1_m['recall']*100:.2f} % |
    | **Falešné poplachy (FP)** | **13** | **6** *(pokles o 54 %)* | **4** *(pokles o 69 %)* |
    """)
    st.success(r"✅ **Úkol splněn:** Úpravou hyperparametru $C=5{,}0$ jsme dosáhli nárůstu Precision z $80{,}00\,\%$ na **$88{,}89\,\%$** (a s $L_1$ regularizací dokonce na **$92{,}16\,\%$**).")
