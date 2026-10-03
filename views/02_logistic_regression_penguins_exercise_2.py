import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# 1. Načtení předpočtených dat
base_dir = Path(__file__).resolve().parent.parent
json_path = base_dir / "02_Classification" / "data" / "penguins_logistic_exercise_2_precomputed.json"

st.title("🎯 Cvičení 2: Tučňáci – Logistická regrese & Multiclass")
st.caption("Splnění všech 8 kroků zadání: penguins_df_normalized.csv, dělení 70/30, class_weight='balanced', multiclass link a záchrana Precision z 59.6 % na 91.7 % laděním C.")

if not json_path.exists():
    st.error("Předpočtená data `penguins_logistic_exercise_2_precomputed.json` nebyla nalezena. Spusťte nejprve výpočetní skript.")
    st.stop()

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

meta = data["metadata"]
head_10 = pd.DataFrame(data["head_10"])
classes = meta["classes"]
base_m = data["baseline_model"]
opt_c_m = data["optimal_c_model"]
scaler_m = data["standard_scaler_model"]
c_sweep = pd.DataFrame(data["c_sweep"])

# =============================================================================
# KROK 1 & 2: NAČTENÍ DAT A KONTROLA PRVNÍCH 10 POZOROVÁNÍ
# =============================================================================
st.subheader("1. & 2. Načtení dat a kontrola prvních 10 pozorování (`head(10)`)")
st.markdown(r"""
Normalizovaný dataset tučňáků souostroví Palmer obsahuje **334 pozorování** a **6 příznaků**:
- Spojité morfologické míry: `culmen_length_mm`, `culmen_depth_mm`, `flipper_length_mm`, `body_mass_g`.
- Kategoriální kódované znaky: `island` (0, 1, 2), `sex` (1 = female, 2 = male).
- Cílová proměnná `species`: **Adelie** (146), **Chinstrap** (68), **Gentoo** (120).
""")

st.dataframe(head_10, width="stretch", hide_index=True)
st.caption(f"Celkem: {meta['samples_total']} tučňáků | Trénovací sada (70 %): {meta['samples_train']} | Testovací sada (30 %): {meta['samples_test']} (`random_state=42`).")

st.divider()

# =============================================================================
# KROK 3 AŽ 7: VÝCHOZÍ MODEL (MULTICLASS, BALANCED, C=1.0)
# =============================================================================
st.subheader("3.–7. Výchozí model logistické regrese (`class_weight='balanced'`)")
st.markdown(r"""
Model `LogisticRegression(class_weight='balanced', random_state=42)` s výchozím $C=1{,}0$:
- **Multiclass link:** V moderním Scikit-learn (1.8+) je výchozím nastavením multinomiální cross-entropy (Softmax).
- **Vyvážení tříd:** Parametr `class_weight='balanced'` penalizuje chyby na menšinové třídě *Chinstrap*.
""")

m_col1, m_col2, m_col3 = st.columns(3)
m_col1.metric("Weighted Precision", f"{base_m['precision_weighted']*100:.2f} %")
m_col2.metric("Macro Precision", f"{base_m['precision_macro']*100:.2f} %")
m_col3.metric("Celková Accuracy", f"{base_m['accuracy']*100:.2f} %")

col_cm_base, col_diag_base = st.columns([1, 1])

with col_cm_base:
    st.markdown("#### Matice záměn výchozího modelu ($C=1{,}0$)")
    cm_b = np.array(base_m["confusion_matrix"])
    fig_cm_b = px.imshow(
        cm_b,
        labels=dict(x="Predikovaný druh", y="Skutečný druh", color="Počet"),
        x=classes,
        y=classes,
        text_auto=True,
        color_continuous_scale="Blues"
    )
    fig_cm_b.update_layout(margin=dict(l=30, r=30, t=30, b=30), height=320)
    st.plotly_chart(fig_cm_b, width="stretch")

with col_diag_base:
    st.markdown("#### Proč výchozí model selhává?")
    st.markdown(r"""
    Výchozí Precision dosahuje pouhých **$59{,}61\,\%$**. Příčina je hluboce metodická:
    1. **Kolize řádkové normalizace:** V předchozím cvičení k-NN byla na dataset aplikována řádková $L_2$ normalizace (`Normalizer()`), která zmenšila spojité míry na hodnoty **$\approx 0{,}01$ až $0{,}05$**.
    2. **Kategoriální příznaky:** Sloupce `island` a `sex` však zůstaly v celých číslech **$0, 1, 2$**.
    3. **Výchozí penalizace $L_2$ ($C=1{,}0$):** Trestá velké váhy. Aby se model mohl řídit milimetry zobáku (hodnoty $0{,}01$), potřeboval by koeficienty $\approx 100$ až $500$. Regularizace mu to však při $C=1{,}0$ zakazuje!
    4. **Důsledek:** Model ignoruje zobák i křídla a plete si *Adelie* s *Chinstrap*.
    """)

st.divider()

# =============================================================================
# KROK 8: EXPERIMENT S REGULARIZACÍ C
# =============================================================================
st.subheader("8. Experiment s hyperparametry: Záchrana modelu uvolněním $C$")
st.markdown(r"""
Zadání požaduje vytvořit model s **vyšší Precision než u výchozího modelu ($59{,}61\,\%$)**.
Pokud oslabíme regularizaci (**zvýšíme parametr $C$** na $1\,000$ až $10\,000$), koeficienty anatomických měr konečně dorostou do potřebné síly:
""")

# Interaktivní výběr C
c_opts = [float(x) for x in c_sweep["C"].tolist()]
selected_c = st.select_slider(
    "Zvolte hodnotu hyperparametru C:",
    options=c_opts,
    value=10000.0
)

sel_row = c_sweep[c_sweep["C"] == selected_c].iloc[0]

c_col1, c_col2, c_col3 = st.columns(3)
c_col1.metric("Weighted Precision", f"{sel_row['precision_weighted']*100:.2f} %", delta=f"{(sel_row['precision_weighted'] - base_m['precision_weighted'])*100:+.2f} % vs Baseline")
c_col2.metric("Macro Precision", f"{sel_row['precision_macro']*100:.2f} %", delta=f"{(sel_row['precision_macro'] - base_m['precision_macro'])*100:+.2f} %")
c_col3.metric("Celková Accuracy", f"{sel_row['accuracy']*100:.2f} %", delta=f"{(sel_row['accuracy'] - base_m['accuracy'])*100:+.2f} %")

# Křivka Precision vs C
fig_sweep = go.Figure()
fig_sweep.add_trace(go.Scatter(x=c_sweep["C"], y=c_sweep["precision_weighted"], mode="lines+markers", name="Weighted Precision", line=dict(color="#1f77b4", width=3)))
fig_sweep.add_trace(go.Scatter(x=c_sweep["C"], y=c_sweep["accuracy"], mode="lines+markers", name="Accuracy", line=dict(color="#2ca02c", width=2, dash="dash")))

fig_sweep.add_vline(x=1.0, line_color="gray", line_dash="dash", annotation_text="Baseline C = 1.0 (59.6 %)")
fig_sweep.add_vline(x=10000.0, line_color="red", line_dash="dot", annotation_text="Optimum C = 10 000 (91.7 %)")

fig_sweep.update_layout(
    title="Dramatický nárůst Precision při uvolnění regularizace C (log scale)",
    xaxis_title="Hyperparametr C (menší = silnější regularizace)",
    xaxis_type="log",
    yaxis_title="Hodnota metriky",
    yaxis=dict(range=[0.50, 0.98]),
    height=400,
    margin=dict(l=40, r=40, t=40, b=40)
)
st.plotly_chart(fig_sweep, width="stretch")

st.divider()

# =============================================================================
# SROVNÁNÍ MODELŮ A EXPERTNÍ ŘEŠENÍ (STANDARDSCALER)
# =============================================================================
st.subheader("Srovnání modelů & Expertní řešení (`StandardScaler`)")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown(r"#### Matice záměn optimálního modelu ($C = 10\,000$):")
    cm_opt = np.array(sel_row["confusion_matrix"])
    fig_opt_cm = px.imshow(
        cm_opt,
        labels=dict(x="Predikovaný druh", y="Skutečný druh", color="Počet"),
        x=classes,
        y=classes,
        text_auto=True,
        color_continuous_scale="Blues"
    )
    fig_opt_cm.update_layout(margin=dict(l=30, r=30, t=30, b=30), height=320)
    st.plotly_chart(fig_opt_cm, width="stretch")

with col_right:
    st.markdown("#### Souhrnné srovnání přístupů:")
    st.markdown(fr"""
    | Model / Konfigurace | Weighted Precision | Macro Precision | Accuracy |
    | :--- | :--- | :--- | :--- |
    | **Výchozí ($C=1.0$)** | **{base_m['precision_weighted']*100:.2f} %** | {base_m['precision_macro']*100:.2f} % | {base_m['accuracy']*100:.2f} % |
    | **Optimální ($C=10000$)** | **{opt_c_m['precision_weighted']*100:.2f} %** | **{opt_c_m['precision_macro']*100:.2f} %** | **{opt_c_m['accuracy']*100:.2f} %** |
    | **Expertní Pipeline (`StandardScaler`)** | **{scaler_m['precision_weighted']*100:.2f} %** | **{scaler_m['precision_macro']*100:.2f} %** | **{scaler_m['accuracy']*100:.2f} %** |
    """)
    st.success(r"✅ **Závěr cvičení:** Laděním parametru $C=10\,000$ jsme zvýšili Precision o více než **$+32$ procentních bodů** (z $59{,}61\,\%$ na **$91{,}71\,\%$**). Pokud navíc nasadíme správný sloupcový `StandardScaler`, model dosáhne špičkových **$97{,}45\,\%$**.")
