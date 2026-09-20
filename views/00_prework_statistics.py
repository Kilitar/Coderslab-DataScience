from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy import stats

base_dir = Path(__file__).resolve().parent.parent
theory_path = base_dir / "00_Prework" / "theory" / "02_statistics_python_tools.md"

st.title("📊 Prework 02: Statistické minimum & Python ekosystém")
st.caption("Míry polohy, rozptylu a závislosti. Interaktivní laboratoř vlivu outlierů na průměr vs. medián a porovnání Pearsonovy vs. Spearmanovy korelace.")

tab1, tab2, tab3 = st.tabs([
    "📖 Teoretický rozbor",
    "🧪 Simulátor: Průměr, Medián a Outliery",
    "🔍 Simulátor: Pearson vs. Spearman korelace"
])

# -----------------------------------------------------------------------------
# TAB 1: TEORETICKÝ TEXT
# -----------------------------------------------------------------------------
with tab1:
    if theory_path.exists():
        with open(theory_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.warning("Soubor 02_statistics_python_tools.md nebyl nalezen.")

# -----------------------------------------------------------------------------
# TAB 2: INTERAKTIVNÍ SIMULÁTOR ROZDĚLENÍ & OUTLIERŮ
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("### 🧪 Laboratoř: Jak odlehlé hodnoty ničí průměr, zatímco medián drží")
    st.caption("Vyberte typ rozdělení a přidejte extrémního 'ředitele s platem 1,5 milionu', abyste viděli reakci metrik.")

    col_ctrl1, col_ctrl2 = st.columns([1, 1])

    with col_ctrl1:
        dist_type = st.selectbox(
            "Zvolte základní rozdělení dat:",
            options=[
                "Symetrické normální rozdělení (např. výška dospělých)",
                "Kladně šikmé rozdělení (např. mzdy, ceny nemovitostí)",
                "Bimodální rozdělení (dva odlišné shluky)"
            ]
        )

    with col_ctrl2:
        add_outlier = st.slider(
            "Přidat extrémní odlehlou hodnotu (Outlier):",
            min_value=0,
            max_value=1000,
            value=0,
            step=50,
            help="Při hodnotě 0 není přítomen žádný outlier. Zkuste posunout na 500 nebo 1000!"
        )

    np.random.seed(42)
    n_samples = 400

    if "normální" in dist_type:
        samples = np.random.normal(loc=100, scale=15, size=n_samples)
    elif "šikmé" in dist_type:
        samples = np.random.lognormal(mean=4.2, sigma=0.6, size=n_samples) + 20
    else:
        s1 = np.random.normal(loc=60, scale=10, size=n_samples // 2)
        s2 = np.random.normal(loc=140, scale=12, size=n_samples // 2)
        samples = np.concatenate([s1, s2])

    if add_outlier > 0:
        samples = np.append(samples, [add_outlier])

    # Výpočet statistických měr
    mean_val = float(np.mean(samples))
    median_val = float(np.median(samples))
    std_val = float(np.std(samples, ddof=1))
    q25 = float(np.percentile(samples, 25))
    q75 = float(np.percentile(samples, 75))
    iqr_val = q75 - q25
    skew_val = float(stats.skew(samples))

    # Zobrazení KPI metrik
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.metric("Aritmetický průměr", f"{mean_val:.1f}")
    with m_col2:
        st.metric("Medián (50 %)", f"{median_val:.1f}")
    with m_col3:
        st.metric("Směrodatná odchylka", f"{std_val:.1f}")
    with m_col4:
        st.metric("IQR (Q3 - Q1)", f"{iqr_val:.1f}")
    with m_col5:
        st.metric("Šikmost (Skewness)", f"{skew_val:+.2f}")

    # Sdružený graf: Horní panel Boxplot (30 % výšky) + Dolní panel Histogram (70 % výšky) se sdílenou osou X
    fig_combined = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.30, 0.70],
        vertical_spacing=0.04,
        subplot_titles=("Krabicový graf (Boxplot – mezichvartilové rozpětí IQR)", "Histogram četností s polohou Průměru a Mediánu")
    )

    # 1. Boxplot (Horní panel)
    fig_combined.add_trace(
        go.Box(
            x=samples,
            name="Boxplot",
            boxpoints="outliers",
            jitter=0.25,
            pointpos=-1.6,
            marker=dict(color="#EF4444", size=9, symbol="diamond"),
            fillcolor="rgba(56, 189, 248, 0.45)",
            line=dict(color="#38BDF8", width=2.5),
            orientation="h"
        ),
        row=1, col=1
    )

    # 2. Histogram (Dolní panel)
    fig_combined.add_trace(
        go.Histogram(
            x=samples,
            nbinsx=50,
            name="Distribuce dat",
            marker=dict(
                color="rgba(56, 189, 248, 0.65)",
                line=dict(color="#0284C7", width=1)
            )
        ),
        row=2, col=1
    )

    # Vertikální linka pro průměr (červená)
    fig_combined.add_vline(
        x=mean_val,
        line_width=3,
        line_dash="dash",
        line_color="#E63946",
        annotation_text=f"Průměr: {mean_val:.1f}",
        annotation_position="top right"
    )

    # Vertikální linka pro medián (zelená)
    fig_combined.add_vline(
        x=median_val,
        line_width=3,
        line_dash="solid",
        line_color="#2A9D8F",
        annotation_text=f"Medián: {median_val:.1f}",
        annotation_position="top left"
    )

    fig_combined.update_layout(
        height=580,
        showlegend=False,
        xaxis2_title="Hodnota veličiny",
        yaxis2_title="Počet pozorování",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    st.plotly_chart(fig_combined, width="stretch")

    if add_outlier >= 300:
        st.warning(
            f"⚠️ **Pozorujte efekt outlieru:** Přidání jediné hodnoty ({add_outlier}) posunulo průměr "
            f"zhruba o {abs(mean_val - median_val):.1f} jednotek směrem nahoru, zatímco medián "
            f"zůstal téměř netečný ({median_val:.1f}). Přesně proto v bankovnictví a nemovitostech "
            f"vždy preferujeme mediánové ceny a robustní škálování!"
        )

# -----------------------------------------------------------------------------
# TAB 3: INTERAKTIVNÍ SIMULÁTOR KORELACÍ
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("### 🔍 Laboratoř: Pearsonův vs. Spearmanův korelační koeficient")
    st.markdown(r"""
    - **Pearsonovo $r$:** Měří výhradně **lineární vztah** po přímce.
    - **Spearmanovo $\rho$:** Počítá korelaci z **pořadí hodnot**, proto zachytí libovolný monotónní vztah i nelineární křivku.
    """)

    rel_choice = st.radio(
        "Vyberte typ vztahu mezi veličinami X a Y:",
        options=[
            "1. Silný lineární vztah se šumem",
            "2. Exponenciální nelineární vztah (y = exp(X))",
            "3. U-tvar / Parabola (y = X^2, symetrická)",
            "4. Lineární vztah s jedním zničujícím outlierem"
        ],
        horizontal=True
    )

    np.random.seed(123)
    n_pts = 100

    if "1. Silný" in rel_choice:
        x_val = np.linspace(1, 10, n_pts)
        y_val = 2.5 * x_val + np.random.normal(0, 2, n_pts)
        note = "Oba koeficienty se shodují na silném kladném vztahu."
    elif "2. Exponenciální" in rel_choice:
        x_val = np.linspace(1, 5, n_pts)
        y_val = np.exp(x_val) + np.random.normal(0, 5, n_pts)
        note = "Spearman zachovává dokonalou korelaci (~1.0), protože pořadí je zachováno. Pearson je nižší kvůli silnému zakřivení."
    elif "3. U-tvar" in rel_choice:
        x_val = np.linspace(-5, 5, n_pts)
        y_val = x_val**2 + np.random.normal(0, 1.5, n_pts)
        note = "Pearsonovo r je téměř 0.00! To neznamená, že proměnné nesouvisejí – souvisejí naprosto deterministicky, ale vztah není lineární!"
    else:
        x_val = np.linspace(1, 10, n_pts)
        y_val = 2.0 * x_val + np.random.normal(0, 1.5, n_pts)
        # Přidáme destruktivní outlier
        x_val = np.append(x_val, [10.0])
        y_val = np.append(y_val, [-45.0])
        note = "Jediný odlehlý bod dramaticky srazil Pearsonovo r, zatímco Spearmanovo rho zůstává stabilní."

    # Výpočet koeficientů
    p_r, _ = stats.pearsonr(x_val, y_val)
    s_rho, _ = stats.spearmanr(x_val, y_val)

    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.metric("Pearsonovo r (Lineární)", f"{p_r:+.3f}")
    with kpi2:
        st.metric("Spearmanovo ρ (Pořadové)", f"{s_rho:+.3f}")
    with kpi3:
        st.metric("Rozdíl |r - ρ|", f"{abs(p_r - s_rho):.3f}")

    fig_corr = go.Figure()
    fig_corr.add_trace(go.Scatter(
        x=x_val,
        y=y_val,
        mode="markers",
        name="Pozorování (X, Y)",
        marker=dict(size=9, color="#38BDF8", line=dict(color="#0284C7", width=1))
    ))

    # OLS regresní přímka přes NumPy bez externí závislosti na statsmodels
    m_fit, b_fit = np.polyfit(x_val, y_val, 1)
    line_x = np.linspace(float(np.min(x_val)), float(np.max(x_val)), 100)
    line_y = m_fit * line_x + b_fit
    fig_corr.add_trace(go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        name=f"Lineární fit: y = {m_fit:.2f}x + {b_fit:.2f}",
        line=dict(color="#E63946", width=2, dash="dash")
    ))

    fig_corr.update_layout(
        title=f"Vztah X vs. Y (Pearson r = {p_r:+.2f}, Spearman ρ = {s_rho:+.2f})",
        xaxis_title="Prediktor X",
        yaxis_title="Odezva Y",
        height=420
    )
    st.plotly_chart(fig_corr, width="stretch")

    st.info(f"💡 **Analýza chování:** {note}")
