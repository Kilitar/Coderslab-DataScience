import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# 1. Načtení předpočtených dat
base_dir = Path(__file__).resolve().parent.parent
json_path = base_dir / "02_Classification" / "data" / "lumbar_decision_tree_exercise_1_precomputed.json"
plot_d3_path = base_dir / "02_Classification" / "plots" / "lumbar_tree_structure_d3.png"
plot_stump_path = base_dir / "02_Classification" / "plots" / "lumbar_tree_structure_stump.png"

st.title("🎯 Cvičení 1: Bederní páteř – Rozhodovací strom & Ladění Precision")
st.caption("Splnění všech 8 kroků zadání kurzu: Načtení dat, výchozí DecisionTreeClassifier, vyhodnocení Precision a experimenty s laděním max_depth a min_samples_leaf.")

if not json_path.exists():
    st.error("Předpočtená data `lumbar_decision_tree_exercise_1_precomputed.json` nebyla nalezena. Spusťte nejprve výpočetní skript.")
    st.stop()

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

meta = data["metadata"]
head_10 = pd.DataFrame(data["head_10"])
base_m = data["baseline_model"]
stump_m = data["stump_model"]
d3_m = data["tuned_model_depth3"]
depth_sweep = pd.DataFrame(data["depth_sweep"])
split_sweep = pd.DataFrame(data["split_sweep"])
leaf_sweep = pd.DataFrame(data["leaf_sweep"])

# =============================================================================
# KROK 1 & 2: NAČTENÍ DAT A KONTROLA PRVNÍCH 10 POZOROVÁNÍ
# =============================================================================
st.subheader("1. & 2. Načtení dat a kontrola prvních 10 pozorování (`head(10)`)")
st.markdown(r"""
Normalizovaný dataset biomechanických měření pánve (`lumbar_normalized_df.csv`) obsahuje **310 pacientů** a **6 anatomických úhlů**:
- `pelvic_incidence`, `pelvic_tilt`, `lumbar_lordosis_angle`, `sacral_slope`, `pelvic_radius`, `degree_spondylolisthesis`.
- Cílová proměnná `class`: `0` = Normal (zdravý nález, 100 pacientů), `1` = Abnormal (výhřez ploténky / spondylolistéza, 210 pacientů).
""")

st.dataframe(head_10, width="stretch", hide_index=True)
st.caption(f"Celkem: {meta['samples_total']} pacientů | Trénovací sada (75 %): {meta['samples_train']} vzorků | Testovací sada (25 %): {meta['samples_test']} vzorků (`random_state=42`).")

st.divider()

# =============================================================================
# KROK 3 AŽ 7: VÝCHOZÍ MODEL (BASELINE BEZ HYPERPARAMETRŮ)
# =============================================================================
st.subheader("3.–7. Výchozí model rozhodovacího stromu (Baseline)")
st.markdown(r"""
Model `DecisionTreeClassifier(random_state=42)` bez omezení hloubky natrénovaný na trénovací sadě (75 % dat):
- Strom volně rostl až do **hloubky 10** s **28 listy**.
- Výsledné metriky na testovací sadě ($78$ pacientů):
""")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Výchozí Precision", f"{base_m['precision']*100:.2f} %", help="Klíčová metrika sledovaná zadáním: TP / (TP + FP)")
m2.metric("Výchozí Accuracy", f"{base_m['accuracy']*100:.2f} %")
m3.metric("Výchozí Recall", f"{base_m['recall']*100:.2f} %")
m4.metric("Výchozí F1-score", f"{base_m['f1_score']:.4f}")

col_cm_base, col_info_base = st.columns([1, 1])

with col_cm_base:
    st.markdown("#### Matice záměn výchozího stromu")
    cm_b = np.array(base_m["confusion_matrix"])
    cm_labels = ["Normal (0)", "Abnormal (1)"]
    fig_cm_base = px.imshow(
        cm_b,
        labels=dict(x="Predikovaná diagnóza", y="Skutečná diagnóza", color="Počet"),
        x=cm_labels,
        y=cm_labels,
        text_auto=True,
        color_continuous_scale="Blues"
    )
    fig_cm_base.update_layout(margin=dict(l=40, r=40, t=30, b=40), height=300)
    st.plotly_chart(fig_cm_base, width="stretch")

with col_info_base:
    st.markdown("#### Proč výchozí model ztrácí Precision?")
    st.markdown(r"""
- **8 falešně pozitivních pacientů ($FP = 8$):** Model označil 8 zdravých lidí za abnormální.
- **Důvod:** Neomezený strom se přeučil na specifický šum v trénovacích datech a vytvořil příliš komplikovaná dílčí pravidla.
- **Hloubka stromu:** `max_depth = 10`, počet listů = `28`.
- **Cíl kroku 8:** Omezit růst stromu a dosáhnout vyšší hodnoty Precision ($> 85{,}19\,\%$).
    """)

st.divider()

# =============================================================================
# KROK 8: EXPERIMENTY S HYPERPARAMETRY PRO ZVÝŠENÍ PRECISION
# =============================================================================
st.subheader("8. Experimenty s hyperparametry: Překonání výchozí Precision")
st.markdown(r"""
Podle přednášky v kurzu máme několik možností regulace růstu stromu (Pre-pruning). Otestovali jsme:
1. **`max_depth` (maximální hloubka stromu):** Zastaví růst na zvolené úrovni.
2. **`min_samples_leaf`:** Minimální počet pacientů v koncovém listu.
3. **`min_samples_split`:** Minimální počet pacientů v uzlu nutný pro jeho rozdělení.
""")

# Srovnávací tabulka nejlepších variant
st.markdown("#### Přehledové srovnání otestovaných architektur stromu")
comp_table = pd.DataFrame({
    "Konfigurace modelu": [
        "Výchozí neomezený strom (default)",
        "Rozhodovací pařez (max_depth=1)",
        "Vyvážený strom (max_depth=3)",
        "Listová regularizace (min_samples_leaf=4)"
    ],
    "Hloubka": [base_m["max_depth"], 1, 3, 5],
    "Počet listů": [base_m["n_leaves"], 2, d3_m["n_leaves"], 13],
    "Test Precision": [
        f"{base_m['precision']*100:.2f} %",
        f"{stump_m['precision']*100:.2f} %",
        f"{d3_m['precision']*100:.2f} %",
        "88.24 %"
    ],
    "Test Accuracy": [
        f"{base_m['accuracy']*100:.2f} %",
        f"{stump_m['accuracy']*100:.2f} %",
        f"{d3_m['accuracy']*100:.2f} %",
        "76.92 %"
    ],
    "Zlepšení Precision": [
        "Baseline (0.00 %)",
        f"+{(stump_m['precision'] - base_m['precision'])*100:.2f} % 🏆 (Maximum)",
        f"+{(d3_m['precision'] - base_m['precision'])*100:.2f} % 🎯 (Optimální rovnováha)",
        "+3.05 %"
    ]
})
st.dataframe(comp_table, width="stretch", hide_index=True)

# Interaktivní simulátor max_depth
st.markdown("#### Interaktivní simulátor: Vliv `max_depth` na přesnost a přeučení")
sel_depth = st.slider("Zvolte úroveň `max_depth`:", min_value=1, max_value=15, value=3, step=1)
cur_row = depth_sweep[depth_sweep["depth"] == sel_depth].iloc[0]

c_m1, c_m2, c_m3, c_m4 = st.columns(4)
c_m1.metric("Test Precision", f"{cur_row['test_precision']*100:.2f} %", delta=f"{(cur_row['test_precision'] - base_m['precision'])*100:.2f} % vs baseline")
c_m2.metric("Test Accuracy", f"{cur_row['test_accuracy']*100:.2f} %")
c_m3.metric("Test Recall", f"{cur_row['test_recall']*100:.2f} %")
c_m4.metric("Počet listů", f"{int(cur_row['n_leaves'])}")

# Křivka Precision vs Depth
fig_sweep = go.Figure()
fig_sweep.add_trace(go.Scatter(
    x=depth_sweep["depth"],
    y=depth_sweep["test_precision"] * 100,
    mode="lines+markers",
    name="Test Precision (%)",
    line=dict(color="#1f77b4", width=3)
))
fig_sweep.add_trace(go.Scatter(
    x=depth_sweep["depth"],
    y=depth_sweep["train_precision"] * 100,
    mode="lines+markers",
    name="Train Precision (%)",
    line=dict(color="#2ca02c", width=2, dash="dash")
))
fig_sweep.add_trace(go.Scatter(
    x=depth_sweep["depth"],
    y=depth_sweep["test_accuracy"] * 100,
    mode="lines+markers",
    name="Test Accuracy (%)",
    line=dict(color="#ff7f0e", width=2)
))
fig_sweep.add_hline(y=base_m["precision"] * 100, line_dash="dot", line_color="red", annotation_text=f"Výchozí Precision ({base_m['precision']*100:.1f} %)")
fig_sweep.add_vline(x=sel_depth, line_dash="dash", line_color="orange", annotation_text=f"Aktuální: depth={sel_depth}")

fig_sweep.update_layout(
    title="Vývoj Precision a Accuracy v závislosti na max_depth",
    xaxis_title="max_depth",
    yaxis_title="Metrika v %",
    height=400,
    margin=dict(l=40, r=40, t=40, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_sweep, width="stretch")

st.divider()

# =============================================================================
# VIZUÁLNÍ STRUKTURA VYBRANÝCH MODELŮ
# =============================================================================
st.subheader("9. Medicínská interpretace & Vizuální architektura stromu")
st.markdown(r"""
Který hyperparametr vybrat v praxi?
- **Varianta A: Rozhodovací pařez (`max_depth=1`)**  
  Dosahuje absolutně nejvyšší **Precision = 95,56 %** ($FP = 2$). Jediné pravidlo zní:  
  `degree_spondylolisthesis <= 0.07` $\implies$ **Normal (0)**, jinak **Abnormal (1)**.
- **Varianta B: Vyvážený strom (`max_depth=3`)**  
  Dosahuje vynikající **Precision = 91,67 %** ($FP = 4$) a navíc zachytává kombinace úhlů pánve (`pelvic_tilt`, `lumbar_lordosis_angle`, `sacral_slope`).
""")

tab_d3, tab_stump = st.tabs(["🌳 Vyvážený strom (max_depth=3)", "🪵 Rozhodovací pařez (max_depth=1)"])

with tab_d3:
    if plot_d3_path.exists():
        st.image(str(plot_d3_path), caption="Struktura stromu s max_depth=3 (Scikit-learn plot_tree)", width="stretch")
    else:
        st.info("Obrázek stromu se načítá...")

with tab_stump:
    if plot_stump_path.exists():
        st.image(str(plot_stump_path), caption="Rozhodovací pařez (max_depth=1) - jediné biomechanické pravidlo", width="stretch")
    else:
        st.info("Obrázek pařezu se načítá...")

st.markdown(r"""
> **Klinický závěr:**  
> Omezení maximální hloubky stromu (`max_depth=1` nebo `max_depth=3`) efektivně vyřešilo problém s přeučením a snížilo počet falešně pozitivních diagnóz z **8** na **2 až 4**, čímž se metrika **Precision zvýšila z 85,19 % na 91,67 % až 95,56 %**!
""")
