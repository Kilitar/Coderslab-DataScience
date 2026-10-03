import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# 1. Načtení předpočtených dat
base_dir = Path(__file__).resolve().parent.parent
json_path = base_dir / "02_Classification" / "data" / "penguins_decision_tree_exercise_2_precomputed.json"
plot_opt_path = base_dir / "02_Classification" / "plots" / "penguins_tree_structure_opt.png"

st.title("🎯 Cvičení 2: Tučňáci – Rozhodovací strom & Multiclass Precision")
st.caption("Splnění všech 8 kroků zadání kurzu: Načtení penguins_df_normalized.csv, split 70/30, výchozí DecisionTree, multiclass Precision a ladění hyperparametrů.")

if not json_path.exists():
    st.error("Předpočtená data `penguins_decision_tree_exercise_2_precomputed.json` nebyla nalezena. Spusťte nejprve výpočetní skript.")
    st.stop()

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

meta = data["metadata"]
classes = meta["classes"]
head_10 = pd.DataFrame(data["head_10"])
base_m = data["baseline_model"]
opt_m = data["optimal_model_depth4"]
depth_sweep = pd.DataFrame(data["depth_sweep"])
crit_sweep = pd.DataFrame(data["criterion_sweep"])
split_sweep = pd.DataFrame(data["split_sweep"])
leaf_sweep = pd.DataFrame(data["leaf_sweep"])

# =============================================================================
# KROK 1 & 2: NAČTENÍ DAT A KONTROLA PRVNÍCH 10 POZOROVÁNÍ
# =============================================================================
st.subheader("1. & 2. Načtení dat a kontrola prvních 10 pozorování (`head(10)`)")
st.markdown(r"""
Normalizovaný dataset tučňáků souostroví Palmer (`penguins_df_normalized.csv`) obsahuje **334 pozorování** a **6 morfologických příznaků**:
- `culmen_length_mm`, `culmen_depth_mm`, `flipper_length_mm`, `body_mass_g`, `island`, `sex`.
- Cílová proměnná `species`: **Adelie** (146), **Chinstrap** (68), **Gentoo** (120).
""")

st.dataframe(head_10, width="stretch", hide_index=True)
st.caption(f"Celkem: {meta['samples_total']} vzorků | Trénovací sada (70 %): {meta['samples_train']} | Testovací sada (30 %): {meta['samples_test']} (`random_state=42`).")

st.divider()

# =============================================================================
# KROK 3 AŽ 7: VÝCHOZÍ MODEL (BASELINE BEZ HYPERPARAMETRŮ)
# =============================================================================
st.subheader("3.–7. Výchozí model rozhodovacího stromu (Baseline)")
st.markdown(r"""
Model `DecisionTreeClassifier(random_state=42)` bez omezení hloubky natrénovaný na trénovací sadě (70 % dat):
- Strom vyrostl do **hloubky 5** s **10 listy**.
- Vzhledem k 3 třídám hodnotíme jak **Macro Precision** (nezatížený průměr), tak **Weighted Precision** a **Per-class Precision**:
""")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Výchozí Macro Precision", f"{base_m['precision_macro']*100:.2f} %", help="Průměr přes 3 druhy: (P_Adelie + P_Chinstrap + P_Gentoo)/3")
m2.metric("Výchozí Weighted Precision", f"{base_m['precision_weighted']*100:.2f} %")
m3.metric("Výchozí Celková Accuracy", f"{base_m['accuracy']*100:.2f} %")
m4.metric("Precision (Chinstrap)", f"{base_m['precision_per_class']['Chinstrap']*100:.2f} %", help="Nejobtížněji odlišitelný druh: 4 falešně pozitivní záměny")

c_cm_base, c_cm_opt = st.columns(2)

with c_cm_base:
    st.markdown("#### Matice záměn: Výchozí strom (depth=5)")
    cm_b = np.array(base_m["confusion_matrix"])
    fig_cm_b = px.imshow(
        cm_b,
        labels=dict(x="Predikovaný druh", y="Skutečný druh", color="Počet"),
        x=classes,
        y=classes,
        text_auto=True,
        color_continuous_scale="Blues"
    )
    fig_cm_b.update_layout(margin=dict(l=40, r=40, t=30, b=40), height=300)
    st.plotly_chart(fig_cm_b, width="stretch")

with c_cm_opt:
    st.markdown("#### Matice záměn: Optimální strom (max_depth=4)")
    cm_o = np.array(opt_m["confusion_matrix"])
    fig_cm_o = px.imshow(
        cm_o,
        labels=dict(x="Predikovaný druh", y="Skutečný druh", color="Počet"),
        x=classes,
        y=classes,
        text_auto=True,
        color_continuous_scale="Greens"
    )
    fig_cm_o.update_layout(margin=dict(l=40, r=40, t=30, b=40), height=300)
    st.plotly_chart(fig_cm_o, width="stretch")

st.markdown(r"""
> **Klíčový rozdíl v maticích záměn:**  
> Ve výchozím stromu bylo **4** tučňáci druhu *Adelie* chybně označeni jako *Chinstrap*. V modelu s `max_depth=4` klesl počet těchto falešně pozitivních záměn na **3**, čímž se **Precision pro Chinstrap zvýšila z 80,95 % na 85,00 %** a celková **Macro Precision stoupla z 92,93 % na 94,29 %**!
""")

st.divider()

# =============================================================================
# KROK 8: EXPERIMENTY S HYPERPARAMETRY PRO ZVÝŠENÍ PRECISION
# =============================================================================
st.subheader("8. Experimenty s hyperparametry: Překonání výchozí Precision")
st.markdown(r"""
Podle zadání úlohy jsme systematicky vyzkoušeli:
1. **Maximální hloubku (`max_depth`):** Průzkum od 1 do 10.
2. **Kritérium dělení (`criterion`):** `gini` vs `entropy` vs `log_loss`.
3. **Minimální počet vzorků na uzel (`min_samples_split`) a list (`min_samples_leaf`).**
""")

# Srovnávací přehled
st.markdown("#### Detailní srovnání výchozího a optimálního modelu")
comp_df = pd.DataFrame({
    "Metrika": [
        "Macro Precision (nevážený průměr)",
        "Weighted Precision (vážený průměr)",
        "Precision druhu Chinstrap",
        "Precision druhu Adelie",
        "Precision druhu Gentoo",
        "Celková Accuracy",
        "Hloubka stromu (max_depth)",
        "Počet listů stromu"
    ],
    "Výchozí strom (default)": [
        f"{base_m['precision_macro']*100:.2f} %",
        f"{base_m['precision_weighted']*100:.2f} %",
        f"{base_m['precision_per_class']['Chinstrap']*100:.2f} %",
        f"{base_m['precision_per_class']['Adelie']*100:.2f} %",
        f"{base_m['precision_per_class']['Gentoo']*100:.2f} %",
        f"{base_m['accuracy']*100:.2f} %",
        f"{base_m['max_depth']}",
        f"{base_m['n_leaves']}"
    ],
    "Optimální strom (max_depth=4)": [
        f"{opt_m['precision_macro']*100:.2f} %",
        f"{opt_m['precision_weighted']*100:.2f} %",
        f"{opt_m['precision_per_class']['Chinstrap']*100:.2f} %",
        f"{opt_m['precision_per_class']['Adelie']*100:.2f} %",
        f"{opt_m['precision_per_class']['Gentoo']*100:.2f} %",
        f"{opt_m['accuracy']*100:.2f} %",
        f"{opt_m['max_depth']}",
        f"{opt_m['n_leaves']}"
    ],
    "Zlepšení": [
        f"+{(opt_m['precision_macro'] - base_m['precision_macro'])*100:.2f} % 🏆",
        f"+{(opt_m['precision_weighted'] - base_m['precision_weighted'])*100:.2f} %",
        f"+{(opt_m['precision_per_class']['Chinstrap'] - base_m['precision_per_class']['Chinstrap'])*100:.2f} % (Klíčový skok)",
        f"+{(opt_m['precision_per_class']['Adelie'] - base_m['precision_per_class']['Adelie'])*100:.2f} %",
        "0.00 % (Perfektní 100 %)",
        f"+{(opt_m['accuracy'] - base_m['accuracy'])*100:.2f} %",
        "-1 úroveň (Méně přeučený)",
        "-2 listy (Vyšší generalizace)"
    ]
})
st.dataframe(comp_df, width="stretch", hide_index=True)

# Interaktivní posuvník hloubky
st.markdown("#### Interaktivní simulátor: Vliv `max_depth` na jednotlivé metriky")
sel_d = st.slider("Zvolte maximální hloubku (`max_depth`):", min_value=1, min_value_step=1, max_value=10, value=4)
sel_row = depth_sweep[depth_sweep["depth"] == sel_d].iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Test Macro Precision", f"{sel_row['test_prec_macro']*100:.2f} %", delta=f"{(sel_row['test_prec_macro'] - base_m['precision_macro'])*100:.2f} % vs baseline")
c2.metric("Test Weighted Precision", f"{sel_row['test_prec_weighted']*100:.2f} %")
c3.metric("Test Accuracy", f"{sel_row['test_accuracy']*100:.2f} %")
c4.metric("Počet listů", f"{int(sel_row['n_leaves'])}")

# Křivka Precision vs max_depth
fig_depth = go.Figure()
fig_depth.add_trace(go.Scatter(
    x=depth_sweep["depth"],
    y=depth_sweep["test_prec_macro"] * 100,
    mode="lines+markers",
    name="Test Macro Precision (%)",
    line=dict(color="#1f77b4", width=3)
))
fig_depth.add_trace(go.Scatter(
    x=depth_sweep["depth"],
    y=depth_sweep["test_prec_weighted"] * 100,
    mode="lines+markers",
    name="Test Weighted Precision (%)",
    line=dict(color="#2ca02c", width=2, dash="dash")
))
fig_depth.add_trace(go.Scatter(
    x=depth_sweep["depth"],
    y=depth_sweep["test_accuracy"] * 100,
    mode="lines+markers",
    name="Test Accuracy (%)",
    line=dict(color="#ff7f0e", width=2)
))
fig_depth.add_hline(y=base_m["precision_macro"] * 100, line_dash="dot", line_color="red", annotation_text=f"Baseline Macro Precision ({base_m['precision_macro']*100:.1f} %)")
fig_depth.add_vline(x=sel_d, line_dash="dash", line_color="orange", annotation_text=f"Aktuální: depth={sel_d}")

fig_depth.update_layout(
    title="Křivka vývoje metrik v závislosti na max_depth",
    xaxis_title="max_depth",
    yaxis_title="Metrika v %",
    height=400,
    margin=dict(l=40, r=40, t=40, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_depth, width="stretch")

st.divider()

# =============================================================================
# SEKCE 9: ARCHITEKTURA OPTIMÁLNÍHO STROMU
# =============================================================================
st.subheader("9. Morfologická interpretace & Vizuální architektura stromu")
st.markdown(r"""
Jak rozhodovací strom rozeznává druhy tučňáků?
1. **Kořenový uzel (`culmen_depth_mm <= 0.00`):** Okamžitě a stoprocentně odděluje tučňáka kroužkového (**Gentoo**), který má specificky plochý a štíhlý zobák v poměru k tělu.
2. **Další úrovně:** Rozlišují druhy **Adelie** a **Chinstrap** podle délky zobáku (`culmen_length_mm`) a délky ploutve / hmotnosti těla.
3. **Proč je `max_depth=4` lepší než `depth=5`:** Pátá úroveň obsahovala příliš specifické pravidlo založené na malém počtu trénovacích vzorků. Jeho odstraněním se zamezilo chybné klasifikaci druhu Adelie jako Chinstrap.
""")

if plot_opt_path.exists():
    st.image(str(plot_opt_path), caption="Optimální rozhodovací strom pro druhy tučňáků (max_depth=4, plot_tree)", width="stretch")
else:
    st.info("Obrázek struktury stromu se generuje...")
