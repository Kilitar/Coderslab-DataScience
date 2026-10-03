import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# 1. Načtení předpočtených dat
base_dir = Path(__file__).resolve().parent.parent
json_path = base_dir / "02_Classification" / "data" / "decision_tree_classification_precomputed.json"
tree_plot_path = base_dir / "02_Classification" / "plots" / "decision_tree_sample_plot.png"

st.title("🎯 Rozhodovací strom: Ukázka & Laboratoř")
st.caption("Klasifikační strom ze slajdů kurzu, vizualizace architektury, interaktivní kalkulátor Gini vs. Entropie a analýza přeučení (Overfitting).")

if not json_path.exists():
    st.error("Předpočtená data `decision_tree_classification_precomputed.json` nebyla nalezena. Spusťte nejprve výpočetní skript.")
    st.stop()

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

meta = data["metadata"]
school = data["school_model"]
crit_comp = data["criterion_comparison"]
depth_sweep = data["depth_sweep"]
toy = data["toy_weather_example"]

# =============================================================================
# SEKCE 1: ŠKOLNÍ VÝSLEDKY ZE SLAJDŮ KURZU
# =============================================================================
st.subheader("1. Školní model ze slajdů kurzu (`DecisionTreeClassifier`)")
st.markdown(r"""
V materiálech kurzu (*Decision tree in classification*) je klasifikační strom trénován na syntetickém datasetu:
- **Konfigurace dat:** 600 vzorků, 5 příznaků ($x_0$ až $x_4$), 2 třídy (`random_state=42`)
- **Dělení dat:** 70 % trénovací sada ($420$ vzorků), 30 % testovací sada ($180$ vzorků)
- **Hyperparametr:** `max_depth=5`, výchozí kritérium `criterion="gini"`
""")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy (Přesnost)", f"{school['accuracy']*100:.2f} %")
col2.metric("Precision (Preciznost)", f"{school['precision']*100:.2f} %")
col3.metric("Recall (Úplnost)", f"{school['recall']*100:.2f} %")
col4.metric("F1-score", f"{school['f1_score']*100:.2f} %")

c_left, c_right = st.columns([1, 1])

with c_left:
    st.markdown("#### Matice záměn (Testovací sada)")
    cm = np.array(school["confusion_matrix"])
    cm_labels = ["Třída 0", "Třída 1"]
    fig_cm = px.imshow(
        cm,
        labels=dict(x="Predikovaná třída", y="Skutečná třída", color="Počet"),
        x=cm_labels,
        y=cm_labels,
        text_auto=True,
        color_continuous_scale="Blues"
    )
    fig_cm.update_layout(
        margin=dict(l=40, r=40, t=30, b=40),
        height=320
    )
    st.plotly_chart(fig_cm, width="stretch")

with c_right:
    st.markdown("#### Důležitost příznaků (Feature Importances - MDI)")
    feat_df = pd.DataFrame({
        "Příznak": list(school["feature_importances"].keys()),
        "Významnost": list(school["feature_importances"].values())
    }).sort_values(by="Významnost", ascending=True)

    fig_feat = px.bar(
        feat_df,
        x="Významnost",
        y="Příznak",
        orientation="h",
        text=feat_df["Významnost"].apply(lambda v: f"{v*100:.1f} %"),
        color="Významnost",
        color_continuous_scale="Viridis"
    )
    fig_feat.update_layout(
        margin=dict(l=40, r=40, t=30, b=40),
        height=320,
        showlegend=False
    )
    st.plotly_chart(fig_feat, width="stretch")

# Srovnání Gini vs Entropy
st.markdown("#### Srovnání kritérií rozdělení (`criterion='gini'` vs. `'entropy'`)")
comp_df = pd.DataFrame({
    "Kritérium": ["Gini Impurity (default)", "Shannon Entropy (log_loss)"],
    "Test Accuracy": [f"{crit_comp['gini']['accuracy']*100:.2f} %", f"{crit_comp['entropy']['accuracy']*100:.2f} %"],
    "Test Precision": [f"{crit_comp['gini']['precision']*100:.2f} %", f"{crit_comp['entropy']['precision']*100:.2f} %"],
    "Vlastnost": [
        "Výpočetně rychlejší (bez logaritmů), mírně preferuje větší třídy.",
        "Teoreticky podložená míra neurčitosti, citlivější na vyvážené pravděpodobnosti."
    ]
})
st.dataframe(comp_df, width="stretch", hide_index=True)

st.divider()

# =============================================================================
# SEKCE 2: VIZUÁLNÍ STRUKTURA STROMU
# =============================================================================
st.subheader("2. Vizualizace architektury stromu (`plot_tree`)")
st.markdown(r"""
Rozhodovací strom je hierarchická struktura pravidel typu **IF-THEN**:
- **Kořenový uzel (Root Node):** První a nejdůležitější test ($x_0 \le -0{,}199$).
- **Vnitřní uzly (Decision Nodes):** Podmínky dělící vzorky do dvou větví (True / False).
- **Listové uzly (Leaves):** Konečné predikce třídy (0 nebo 1) bez dalšího větvení.
""")

if tree_plot_path.exists():
    st.image(str(tree_plot_path), caption="Rozhodovací strom modelu s max_depth=5 vygenerovaný knihovnou Scikit-learn (plot_tree)", width="stretch")
else:
    st.info("Obrázek struktury stromu se generuje při spuštění skriptu.")

with st.expander("🔍 Jak číst data v jednotlivých uzlech stromu"):
    st.markdown(r"""
Každý obdélník reprezentuje stav vzorků v daném uzlu:
1. **$x[j] \le c$:** Prahová hodnota příznaku pro binární rozdělení (např. $x[0] \le -0{,}199$). Pokud podmínka platí, vzorek jde doleva (**True**), jinak doprava (**False**).
2. **`gini` / `entropy`:** Čistota uzlu. Čím menší číslo, tím homogennější třídy. V čistém listu je $\text{gini} = 0{,}0$.
3. **`samples`:** Počet trénovacích vzorků procházejících tímto uzlem.
4. **`value = [n_0, n_1]`:** Zastoupení jednotlivých tříd v uzlu (např. $[213, 207]$ znamená 213 vzorků třídy 0 a 207 třídy 1).
5. **`class`:** Majoritní třída, která by byla predikována, pokud by se strom v tomto bodě zastavil.
    """)

st.divider()

# =============================================================================
# SEKCE 3: INTERAKTIVNÍ KALKULÁTOR ROZDĚLOVACÍCH KRITÉRIÍ
# =============================================================================
st.subheader("3. Laboratoř: Výpočet Gini vs. Shannonovy entropie")
st.markdown(r"""
Vyzkoušejte si, jak se mění míra nečistoty v uzlu se dvěma třídami podle počtu pozorování:
""")

calc_c1, calc_c2 = st.columns(2)
with calc_c1:
    n_a = st.slider("Počet vzorků Třídy A", min_value=0, max_value=100, value=30, step=1)
with calc_c2:
    n_b = st.slider("Počet vzorků Třídy B", min_value=0, max_value=100, value=30, step=1)

total_n = n_a + n_b
if total_n == 0:
    st.warning("Zadejte alespoň 1 vzorek.")
else:
    p_a = n_a / total_n
    p_b = n_b / total_n

    # Výpočty
    gini_val = 1.0 - (p_a**2 + p_b**2)
    entropy_val = 0.0
    if p_a > 0:
        entropy_val -= p_a * np.log2(p_a)
    if p_b > 0:
        entropy_val -= p_b * np.log2(p_b)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Podíl Třídy A ($p_A$)", f"{p_a*100:.1f} %")
    m2.metric("Podíl Třídy B ($p_B$)", f"{p_b*100:.1f} %")
    m3.metric("Gini Impurity", f"{gini_val:.4f}", help="Maximum je 0.5 pro rovnoměrné rozdělení (50/50)")
    m4.metric("Shannonova Entropie", f"{entropy_val:.4f}", help="Maximum je 1.0 bit pro rovnoměrné rozdělení (50/50)")

    # Graf závislosti nečistoty na pravděpodobnosti
    p_range = np.linspace(0.0001, 0.9999, 200)
    gini_curve = 1.0 - (p_range**2 + (1.0 - p_range)**2)
    entropy_curve = - (p_range * np.log2(p_range) + (1.0 - p_range) * np.log2(1.0 - p_range))

    fig_crit = go.Figure()
    fig_crit.add_trace(go.Scatter(x=p_range, y=entropy_curve, mode="lines", name="Shannonova Entropie H(p)", line=dict(color="#d62728", width=2)))
    fig_crit.add_trace(go.Scatter(x=p_range, y=gini_curve, mode="lines", name="Gini Impurity (CART)", line=dict(color="#1f77b4", width=2)))
    fig_crit.add_trace(go.Scatter(x=p_range, y=gini_curve * 2, mode="lines", name="2 × Gini (škálovaný)", line=dict(color="#1f77b4", dash="dash")))

    # Aktuální bod
    fig_crit.add_trace(go.Scatter(
        x=[p_a], y=[entropy_val],
        mode="markers", marker=dict(color="#d62728", size=12, symbol="cross"),
        name=f"Aktuální Entropie ({entropy_val:.3f})"
    ))
    fig_crit.add_trace(go.Scatter(
        x=[p_a], y=[gini_val],
        mode="markers", marker=dict(color="#1f77b4", size=12, symbol="circle"),
        name=f"Aktuální Gini ({gini_val:.3f})"
    ))

    fig_crit.update_layout(
        title="Průběh nečistoty v závislosti na pravděpodobnosti třídy p",
        xaxis_title="Pravděpodobnost třídy p",
        yaxis_title="Míra nečistoty (Impurity)",
        height=380,
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_crit, width="stretch")

st.markdown(r"""
> **Příklad ze slajdů kurzu (Počasí a 6 vzorků):**
> - **Před rozdělením (Root):** 3 Sunny, 3 Rainy $\implies p = 0{,}5 \implies H = 1{,}0$ bit, $\text{Gini} = 0{,}50$.
> - **Rozdělení podle Teploty:**
>   - *Větev Nízká:* 2 Rainy, 1 Sunny $\implies H = 0{,}918$, $\text{Gini} = 0{,}375$ ($3$ vzorky).
>   - *Větev Vysoká:* 0 Rainy, 3 Sunny $\implies H = 0{,}000$, $\text{Gini} = 0{,}000$ ($3$ vzorky).
> - **Vážená entropie po rozdělení:** $\frac{3}{6} \cdot 0{,}918 + \frac{3}{6} \cdot 0 = 0{,}459$.
> - **Informační zisk (Information Gain):** $\text{IG} = 1{,}0 - 0{,}459 = 0{,}541$ bitů!
""")

st.divider()

# =============================================================================
# SEKCE 4: ANALÝZA PŘEUČENÍ (OVERFITTING LAB)
# =============================================================================
st.subheader("4. Analýza přeučení (Overfitting) podle hloubky stromu")
st.markdown(r"""
Rozhodovací stromy jsou náchylné k **přeučení (overfittingu)**. Pokud strom necháme růst bez omezení, vytvoří pravidlo pro každý šum v datech:
- Trénovací přesnost dosáhne $100\,\%$, ale testovací přesnost začne klesat!
""")

sweep_df = pd.DataFrame(depth_sweep)

# Interaktivní výběr hloubky
selected_depth = st.slider("Zvolte maximální hloubku stromu (`max_depth`) pro detailní pohled:", min_value=1, max_value=15, value=5, step=1)
sel_row = sweep_df[sweep_df["depth"] == selected_depth].iloc[0]

sc1, sc2, sc3, sc4 = st.columns(4)
sc1.metric("Trénovací přesnost", f"{sel_row['train_accuracy']*100:.2f} %")
sc2.metric("Testovací přesnost", f"{sel_row['test_accuracy']*100:.2f} %", delta=f"{(sel_row['test_accuracy'] - sel_row['train_accuracy'])*100:.2f} %")
sc3.metric("Testovací preciznost", f"{sel_row['test_precision']*100:.2f} %")
sc4.metric("Počet listů stromu", f"{int(sel_row['n_leaves'])}")

# Křivka učení (Train vs Test Accuracy)
fig_overfit = go.Figure()
fig_overfit.add_trace(go.Scatter(
    x=sweep_df["depth"],
    y=sweep_df["train_accuracy"] * 100,
    mode="lines+markers",
    name="Trénovací přesnost (Train)",
    line=dict(color="#2ca02c", width=2.5)
))
fig_overfit.add_trace(go.Scatter(
    x=sweep_df["depth"],
    y=sweep_df["test_accuracy"] * 100,
    mode="lines+markers",
    name="Testovací přesnost (Test / Generalizace)",
    line=dict(color="#1f77b4", width=2.5)
))

# Zvýraznění optimální zóny a vybrané hloubky
fig_overfit.add_vline(x=selected_depth, line_dash="dash", line_color="orange", annotation_text=f"Zvoleno: d={selected_depth}")
fig_overfit.add_vrect(x0=4, x1=6, fillcolor="rgba(0, 200, 0, 0.1)", line_width=0, annotation_text="Optimální zóna (d=4 až 6)", annotation_position="top left")

fig_overfit.update_layout(
    title="Vývoj Train vs. Test přesnosti v závislosti na max_depth",
    xaxis_title="Maximální hloubka stromu (max_depth)",
    yaxis_title="Přesnost (Accuracy v %)",
    height=420,
    margin=dict(l=40, r=40, t=40, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_overfit, width="stretch")

st.markdown(r"""
#### Jak čelit přeučení u rozhodovacích stromů?
1. **Pre-pruning (Omezení růstu během tréninku):**
   - `max_depth`: Zastaví větvení po dosažení určité úrovně (např. 4–6).
   - `min_samples_split`: Minimální počet vzorků nutný pro další rozdělení uzlu (např. 10–20).
   - `min_samples_leaf`: Minimální počet vzorků, který musí zůstat v každém listu (např. 5).
2. **Post-pruning (Zpětné prořezávání pomocí komplexity nákladů):**
   - Využívá hyperparametr `ccp_alpha` (Cost-Complexity Pruning). Minimalizuje chybovost penalizovanou počtem listů: $R_\alpha(T) = R(T) + \alpha |T|$.
3. **Ensemble metody (Předzvěst dalších lekcí):**
   - Místo jednoho hlubokého stromu kombinujeme desítky až stovky stromů: **Random Forest** (bagging) nebo **Gradient Boosting / XGBoost / LightGBM**.
""")
