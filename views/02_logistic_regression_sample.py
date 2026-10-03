import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# 1. Načtení předpočtených dat
base_dir = Path(__file__).resolve().parent.parent
json_path = base_dir / "02_Classification" / "data" / "logistic_regression_precomputed.json"

st.title("🎯 Logistická regrese: Ukázka & Laboratoř")
st.caption("Interaktivní rozbor modelu ze slajdů kurzu, geometrické srovnání OLS vs. Sigmoida, onkologický simulátor a ladění rozhodovacího prahu.")

if not json_path.exists():
    st.error("Předpočtená data `logistic_regression_precomputed.json` nebyla nalezena. Spusťte nejprve výpočetní skript.")
    st.stop()

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

syn_data = data["synthetic_model"]
comp_1d = data["comparison_1d"]
tumor_data = data["clinical_tumor_example"]
lumbar_data = data.get("lumbar_analysis")

# =============================================================================
# SEKCE 1: ŠKOLNÍ VÝSLEDKY ZE SLAJDŮ KURZU (Slajdy 18-21)
# =============================================================================
st.subheader("1. Školní model ze slajdů kurzu (`make_classification`)")
st.markdown(r"""
V materiálech kurzu (*slajdy 18–21*) je logistická regrese demonstrována na syntetickém datasetu:
- **Rozsah:** 600 pozorování, 5 nezávislých proměnných, 2 třídy (`random_state=42`)
- **Dělení dat:** 70 % trénovací sada ($420$ vzorků), 30 % testovací sada ($180$ vzorků)
- **Model:** `LogisticRegression(random_state=42)`
""")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{syn_data['accuracy']*100:.2f} %")
col2.metric("Precision", f"{syn_data['precision']*100:.2f} %")
col3.metric("Recall", f"{syn_data['recall']*100:.2f} %")
col4.metric("F1-score", f"{syn_data['f1_score']*100:.2f} %")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("#### Matice záměn (Testovací sada, $\\theta = 0{,}5$)")
    cm = np.array(syn_data["confusion_matrix"])
    cm_labels = ["Negativní (0)", "Pozitivní (1)"]
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

with col_right:
    st.markdown("#### Odhadnuté váhy příznaků $\\beta_j$")
    coef_df = pd.DataFrame({
        "Příznak": [f"Feature {i}" for i in range(len(syn_data["coefficients"]))],
        "Koeficient (Beta)": syn_data["coefficients"],
        "Odds Ratio (exp(Beta))": [round(float(np.exp(c)), 4) for c in syn_data["coefficients"]]
    })
    st.dataframe(coef_df, width="stretch", hide_index=True)
    st.caption(f"Intercept (absolutní posun $b$): `{syn_data['intercept']:.4f}` | Log Loss: `{syn_data['log_loss']:.4f}` | ROC-AUC: `{syn_data['roc_auc']:.4f}`")

st.divider()

# =============================================================================
# SEKCE 2: PROČ OLS SELHÁVÁ PŘI KLASIFIKACI (Slajdy 4-6)
# =============================================================================
st.subheader("2. Geometrické srovnání: OLS versus Logistická regrese (Slajdy 4–6)")
st.markdown(r"""
Proč nelze pro klasifikaci použít běžnou lineární regresi (OLS)?
1. **Překročení intervalu $[0, 1]$:** Přímka OLS $\hat{y} = ax + b$ může pro krajní hodnoty dávat záporná čísla nebo hodnoty $> 1$.
2. **Nelineární zploštění (Sigmoida):** Logistická regrese provádí stlačení pomocí $\sigma(z) = \frac{1}{1 + e^{-z}}$, což zaručuje platné aposteriorní pravděpodobnosti.
""")

x_grid = comp_1d["x_grid"]
y_lin = comp_1d["y_linear"]
y_log = comp_1d["y_logistic"]
pts = comp_1d["sample_points"]

fig_comp = go.Figure()

# Skutečné datové body
fig_comp.add_trace(go.Scatter(
    x=[p["x"] for p in pts],
    y=[p["y"] for p in pts],
    mode="markers",
    marker=dict(size=8, color="#1f77b4", opacity=0.6),
    name="Pozorování (Ground Truth)"
))

# Křivka OLS
fig_comp.add_trace(go.Scatter(
    x=x_grid,
    y=y_lin,
    mode="lines",
    line=dict(color="#d62728", width=2, dash="dash"),
    name="Lineární regrese OLS (padá mimo [0, 1])"
))

# Křivka Logistické regrese
fig_comp.add_trace(go.Scatter(
    x=x_grid,
    y=y_log,
    mode="lines",
    line=dict(color="#2ca02c", width=3),
    name="Logistická regrese (Sigmoida)"
))

# Referenční asymptoty 0 a 1
fig_comp.add_hline(y=0, line_dash="dot", line_color="gray", annotation_text="Asymptota 0.0")
fig_comp.add_hline(y=1, line_dash="dot", line_color="gray", annotation_text="Asymptota 1.0")
fig_comp.add_hline(y=0.5, line_dash="dot", line_color="orange", annotation_text="Práh theta = 0.5")

fig_comp.update_layout(
    title=f"Srovnání OLS vs. Sigmoida na dominantním příznaku ({comp_1d['feature_name']})",
    xaxis_title=f"Hodnota {comp_1d['feature_name']}",
    yaxis_title="Predikce / Pravděpodobnost",
    yaxis=dict(range=[-0.25, 1.25]),
    height=420,
    margin=dict(l=40, r=40, t=40, b=40)
)
st.plotly_chart(fig_comp, width="stretch")

st.divider()

# =============================================================================
# SEKCE 3: KLINICKÝ ONKOLOGICKÝ SIMULÁTOR (Slajd 10)
# =============================================================================
st.subheader("3. Klinický simulátor: Diagnostika malignity tumoru (Slajd 10)")
st.markdown(r"""
Ve slajdu 10 je uveden klinický příklad predikce malignity (zhoubnosti) nádoru na základě věku a průměru v milimetrech:
$$p(\mathbf{x}) = \frac{1}{1 + e^{-(a_1 x_1 + a_2 x_2 + b)}} = \frac{1}{1 + e^{-(0{,}005 \cdot \text{věk} + 0{,}015 \cdot \text{průměr} + 0{,}1)}}$$
Vyzkoušejte interaktivně zadat parametry pacienta a sledujte výslednou pravděpodobnost:
""")

sim_col1, sim_col2 = st.columns([1, 1])

with sim_col1:
    in_age = st.slider("Věk pacienta (roky):", min_value=20, max_value=85, value=50, step=1)
    in_diam = st.slider("Průměr tumoru (mm):", min_value=5, max_value=65, value=35, step=1)
    in_threshold = st.slider("Rozhodovací práh pro označení za maligní (theta):", min_value=0.10, max_value=0.90, value=0.50, step=0.05)

    calc_z = 0.005 * in_age + 0.015 * in_diam + 0.1
    calc_p = 1.0 / (1.0 + np.exp(-calc_z))
    is_malignant = calc_p >= in_threshold

    st.markdown("---")
    st.markdown(f"**Lineární index $z$:** `{calc_z:.4f}`")
    st.markdown(fr"**Pravděpodobnost malignity $p(\mathbf{{x}})$:** `{calc_p*100:.2f} %`")
    
    if is_malignant:
        st.error(f"🚨 **Diagnóza:** MALIGNÍ (Zhubný) – pravděpodobnost {calc_p*100:.1f} % překročila práh {in_threshold*100:.0f} %")
    else:
        st.success(f"✅ **Diagnóza:** BENIGNÍ (Nezhubný) – pravděpodobnost {calc_p*100:.1f} % je pod prahem {in_threshold*100:.0f} %")

with sim_col2:
    grid_ages = tumor_data["grid_ages"]
    grid_diams = tumor_data["grid_diameters"]
    mesh_p = np.array(tumor_data["mesh_probabilities"])

    fig_contour = go.Figure(data=go.Contour(
        z=mesh_p,
        x=grid_ages,
        y=grid_diams,
        colorscale="Reds",
        colorbar=dict(title="Pravděpodobnost"),
        contours=dict(
            start=0.1,
            end=0.9,
            size=0.1,
            showlabels=True
        )
    ))

    # Aktuální pozice vybraného pacienta
    fig_contour.add_trace(go.Scatter(
        x=[in_age],
        y=[in_diam],
        mode="markers",
        marker=dict(size=14, color="blue", symbol="diamond", line=dict(color="white", width=2)),
        name=f"Váš pacient ({calc_p*100:.1f} %)"
    ))

    # Původní pacient ze slajdu
    fig_contour.add_trace(go.Scatter(
        x=[tumor_data["slide_age"]],
        y=[tumor_data["slide_diameter"]],
        mode="markers",
        marker=dict(size=10, color="black", symbol="circle"),
        name="Pacient ze slajdu (50 let, 35 mm: 70.6 %)"
    ))

    fig_contour.update_layout(
        title="Vrstevnicová mapa pravděpodobnosti malignity",
        xaxis_title="Věk pacienta (roky)",
        yaxis_title="Průměr tumoru (mm)",
        height=380,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig_contour, width="stretch")

st.divider()

# =============================================================================
# SEKCE 4: ANALÝZA ROZHODOVACÍHO PRAHU (Slajd 11)
# =============================================================================
st.subheader("4. Analýza rozhodovacího prahu: Precision versus Recall Trade-off (Slajd 11)")
st.markdown(r"""
Výchozí práh $\theta = 0{,}5$ nemusí vyhovovat každé úloze:
- **Medicína & Odhalování fraudů:** Snižujeme $\theta$ (např. na $0{,}20$), abychom maximalizovali **Recall** a nepřehlédli nemocné pacienty.
- **Detekce SPAMu & Automatické sankce:** Zvyšujeme $\theta$ (např. na $0{,}80$), abychom maximalizovali **Precision** a neomezili legitimní uživatele.
""")

th_sweep = syn_data["threshold_sweep"]
th_df = pd.DataFrame(th_sweep)

sel_th = st.slider("Zvolte rozhodovací práh (theta) pro testovací sadu:", min_value=0.05, max_value=0.95, value=0.50, step=0.025)

# Nalezení nejbližšího záznamu v předpočtených datech
closest_idx = int(np.argmin(np.abs(th_df["threshold"] - sel_th)))
sel_row = th_df.iloc[closest_idx]

m_col1, m_col2, m_col3, m_col4 = st.columns(4)
m_col1.metric("Vybraný práh (theta)", f"{sel_row['threshold']:.2f}")
m_col2.metric("Precision (Přesnost)", f"{sel_row['precision']*100:.2f} %")
m_col3.metric("Recall (Senzitivita)", f"{sel_row['recall']*100:.2f} %")
m_col4.metric("F1-score", f"{sel_row['f1']*100:.2f} %")

fig_th = go.Figure()
fig_th.add_trace(go.Scatter(x=th_df["threshold"], y=th_df["precision"], mode="lines", name="Precision", line=dict(color="#1f77b4", width=2.5)))
fig_th.add_trace(go.Scatter(x=th_df["threshold"], y=th_df["recall"], mode="lines", name="Recall", line=dict(color="#ff7f0e", width=2.5)))
fig_th.add_trace(go.Scatter(x=th_df["threshold"], y=th_df["f1"], mode="lines", name="F1-score", line=dict(color="#2ca02c", width=2.5)))
fig_th.add_trace(go.Scatter(x=th_df["threshold"], y=th_df["accuracy"], mode="lines", name="Accuracy", line=dict(color="gray", dash="dash")))

fig_th.add_vline(x=sel_row["threshold"], line_color="red", line_dash="dot", annotation_text=f"Vybráno: theta = {sel_row['threshold']:.2f}")

fig_th.update_layout(
    title="Křivky klasifikačních metrik v závislosti na rozhodovacím prahu theta",
    xaxis_title="Rozhodovací práh theta",
    yaxis_title="Hodnota metriky",
    yaxis=dict(range=[0, 1.05]),
    height=400,
    margin=dict(l=40, r=40, t=40, b=40)
)
st.plotly_chart(fig_th, width="stretch")

st.info(fr"""
Při zvoleném prahu **$\theta = {sel_row['threshold']:.2f}$**:
- **Zachyceno nemocných (TP):** `{sel_row['tp']}` z celkových `{sel_row['tp'] + sel_row['fn']}` pozitivních případů.
- **Přehlédnuto (FN – fatální chyba):** `{sel_row['fn']}` případů.
- **Falešných poplachů (FP):** `{sel_row['fp']}` případů.
""")

st.divider()

# =============================================================================
# SEKCE 5: SROVNÁNÍ NA DATECH BEDERNÍ PÁTEŘE (LUMBAR DATA)
# =============================================================================
if lumbar_data:
    st.subheader("5. Aplikace na data kurzu: Bederní páteř (`lumbar_data.csv`)")
    st.markdown(r"""
    Aplikovali jsme logistickou regresi na reálný medicínský dataset bederní páteře (310 pacientů, 6 anatomických úhlů) a porovnali její výkon s modelem **k-NN** z předchozích cvičení:
    """)

    l_col1, l_col2, l_col3 = st.columns(3)
    l_col1.metric("Logistická regrese (Recall)", f"{lumbar_data['recall']*100:.2f} %", delta=f"{lumbar_data['recall']*100 - 80.70:.2f} % vs k-NN (k=5)")
    l_col2.metric("Logistická regrese (Accuracy)", f"{lumbar_data['accuracy']*100:.2f} %")
    l_col3.metric("Logistická regrese (ROC-AUC)", f"{lumbar_data['roc_auc']:.4f}", delta=f"{lumbar_data['roc_auc'] - 0.8208:+.4f} vs k-NN (k=5)")

    st.markdown("#### Anatomické koeficienty a poměry šancí (Odds Ratios):")
    lumb_coefs = lumbar_data["coefficients"]
    lumb_odds = lumbar_data["odds_ratios"]
    lumb_df = pd.DataFrame({
        "Anatomický parametr": list(lumb_coefs.keys()),
        "Koeficient Beta": list(lumb_coefs.values()),
        "Odds Ratio (exp(Beta))": list(lumb_odds.values()),
        "Klinická interpretace": [
            "Zvyšuje šanci na patologii" if b > 0 else "Snižuje šanci na patologii"
            for b in lumb_coefs.values()
        ]
    }).sort_values("Odds Ratio (exp(Beta))", ascending=False)

    st.dataframe(lumb_df, width="stretch", hide_index=True)
    st.caption("Poznámka: Výpočet proběhl na normalizovaných datech se zapnutou L2 regularizací (C=1.0).")
