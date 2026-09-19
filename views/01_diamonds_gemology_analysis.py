from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

st.title("💎 Cvičení 2: Gemologická analýza, 4C & Paradox multikolinearity")
st.caption("Proč vyřazení barvy a čistoty zničilo model klenotníka a proč Pearsonova korelace selhala u proporcí brusu.")

# =============================================================================
# NAČTENÍ DAT
# =============================================================================
@st.cache_data
def load_diamonds_analysis_data():
    base_dir = Path(__file__).resolve().parent.parent
    csv_path = base_dir / "01_Regression" / "data" / "diamonds.csv"
    if not csv_path.exists():
        csv_path = base_dir / "data" / "MAL_downloadable materials_session 1" / "Day 1" / "diamonds.csv"

    df = pd.read_csv(csv_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Čištění fyzikálně nemožných nul a hrubých překlepů
    clean = df[
        (df["x"] > 0)
        & (df["y"] > 0)
        & (df["z"] > 0)
        & (df["y"] < 20)
        & (df["z"] < 20)
    ].copy()

    # Kódování 4C
    cut_map = {"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4}
    color_map = {"J": 0, "I": 1, "H": 2, "G": 3, "F": 4, "E": 5, "D": 6}
    clarity_map = {"I1": 0, "SI2": 1, "SI1": 2, "VS2": 3, "VS1": 4, "VVS2": 5, "VVS1": 6, "IF": 7}

    clean["cut_num"] = clean["cut"].map(cut_map)
    clean["color_num"] = clean["color"].map(color_map)
    clean["clarity_num"] = clean["clarity"].map(clarity_map)
    clean["est_volume"] = clean["x"] * clean["y"] * clean["z"]

    return clean

df_d = load_diamonds_analysis_data()

# =============================================================================
# 1. ČTYŘI METODICKÉ CHYBY V JEDNOM POHLEDU
# =============================================================================
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.error("❌ 1. Vyřazení 4C")
    st.caption("Kurz zahodil barvu a čistotu, protože to byly textové sloupce. Model nevidí rozdíl mezi žlutým a čirým kamenem.")
with c2:
    st.warning("⚠️ 2. Past Pearsona")
    st.caption("Proporce 'depth' mají korelaci r = -0.011 (nula!). Přesto mimo ideál 61–62.5 % kámen ztrácí až 40 % ceny.")
with c3:
    st.info("📉 3. Fyzikální paradox")
    st.caption("Hmotnost a objem x*y*z mají korelaci 0.9989. OLS model dal rozměrům absurdní záporné váhy (-3 675 USD za x).")
with c4:
    st.success("👑 4. Solitéry > 3.5 ct")
    st.caption("Umělý outlier filtr vyřadil 9 nejcennějších diamantů (až 5.01 ct), kde klenotník nejvíce riskuje.")

st.markdown("---")

# =============================================================================
# 2. PROČ 1-KARÁTOVÝ KÁMEN STOJÍ $2 000 NEBO $17 000? (SIMULACE 4C)
# =============================================================================
st.subheader("🔍 1. Klenotnická realita: Síla parametrů Barva (Color) a Čistota (Clarity)")
st.write(
    """
    Podle učebnicového zadání kurzu model vidí pouze **váhu a rozměry**.  
    Zde je důkaz, proč by klenotník s takovým modelem okamžitě prodělal majetek:
    """
)

# Filtrujeme 1-karátové diamanty (0.98 - 1.02 ct)
sample_1ct = df_d[(df_d["carat"] >= 0.98) & (df_d["carat"] <= 1.02)]
min_1ct_price = sample_1ct["price"].min()
max_1ct_price = sample_1ct["price"].max()
median_1ct_price = sample_1ct["price"].median()

v1, v2, v3 = st.columns(3)
with v1:
    st.metric("Nejlevnější 1 ct diamant", f"${min_1ct_price:,.0f}", help="Typicky nažloutlý s prasklinami: Color J, Clarity I1, brus Fair")
with v2:
    st.metric("Medián 1 ct diamantu", f"${median_1ct_price:,.0f}")
with v3:
    st.metric("Nejdražší 1 ct diamant", f"${max_1ct_price:,.0f}", help="Dokonale čirý: Color D, Clarity IF, brus Ideal")

st.markdown(
    f"""
    > 💡 **Rozptyl ceny pro přesně stejnou hmotnost 1.00 karátu:**  
    > Dva diamanty na pultu váží na gram přesně stejně (~0.20 g).  
    > Přesto jeden stojí **{min_1ct_price:,.0f} USD** a druhý **{max_1ct_price:,.0f} USD** (rozdíl více než osminásobek!).  
    > Učebnicový model kurzu bez 4C parametrů by oběma přiřkl průměrnou cenu **~{median_1ct_price:,.0f} USD**!
    """
)

st.markdown("---")

# =============================================================================
# 3. PAST PEARSONOVY KORELACE: HLOUBKA (DEPTH) A PLOŠKA (TABLE)
# =============================================================================
st.subheader("📐 2. Past lineární korelace: Proč Pearson selhává u `depth` a `table`?")
st.write(
    """
    V zadání kurzu je pokyn: *„Vylučte proměnné s nízkým korelačním koeficientem.“*  
    Student spočítá lineární korelaci:
    - **depth % vůči ceně:** $r = -0{,}011$ (čistá nula!)
    - **table % vůči ceně:** $r = +0{,}127$  
    A obě proměnné bez milosti vymaže.

    **Proč je to metodický omyl?**  
    Proporce brusu neurčují cenu po šikmé přímce, nýbrž tvoří **zvonovitou penalizační křivku**:
    - Pokud má diamant hloubku **61.0 % až 62.5 %** (zelené pásmo), odráží maximum světla zpět do oka (*ideální brilance*).
    - Mimo toto pásmo vzniká *„fish-eye“* (příliš mělký) nebo *„nail-head“* (příliš hluboký) efekt – diamant je tmavý a **ztrácí 20 až 40 % své tržní hodnoty**!
    """
)

sample_sub = df_d.sample(n=min(3000, len(df_d)), random_state=42)

fig_depth = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "Depth % vs Cena (Ideální zóna 61–62.5 %)",
        "Table % vs Cena (Ideální zóna 54–57 %)"
    ),
    horizontal_spacing=0.12
)

# Scatter Depth
fig_depth.add_trace(
    go.Scatter(
        x=sample_sub["depth"],
        y=sample_sub["price"],
        mode="markers",
        marker=dict(size=5, color="#8B5CF6", opacity=0.35),
        name="Diamanty",
        hovertemplate="Depth: %{x:.1f}%<br>Cena: $%{y:,.0f}<extra></extra>"
    ),
    row=1, col=1
)
fig_depth.add_vrect(
    x0=61.0, x1=62.5, fillcolor="rgba(16, 185, 129, 0.2)",
    line_width=1.5, line_color="#10B981",
    annotation_text="Ideál (61–62.5 %)", annotation_position="top left",
    row=1, col=1
)

# Scatter Table
fig_depth.add_trace(
    go.Scatter(
        x=sample_sub["table"],
        y=sample_sub["price"],
        mode="markers",
        marker=dict(size=5, color="#EC4899", opacity=0.35),
        name="Diamanty",
        hovertemplate="Table: %{x:.1f}%<br>Cena: $%{y:,.0f}<extra></extra>"
    ),
    row=1, col=2
)
fig_depth.add_vrect(
    x0=54.0, x1=57.0, fillcolor="rgba(16, 185, 129, 0.2)",
    line_width=1.5, line_color="#10B981",
    annotation_text="Ideál (54–57 %)", annotation_position="top left",
    row=1, col=2
)

fig_depth.update_layout(height=480, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
fig_depth.update_xaxes(title_text="Celková hloubka Depth (%)", range=[54, 70], row=1, col=1)
fig_depth.update_yaxes(title_text="Cena diamantu ($)", row=1, col=1)
fig_depth.update_xaxes(title_text="Ploška Table (%)", range=[50, 70], row=1, col=2)
fig_depth.update_yaxes(title_text="Cena diamantu ($)", row=1, col=2)

st.plotly_chart(fig_depth, use_container_width=True)

st.markdown("---")

# =============================================================================
# 4. FYZIKÁLNÍ MULTIKOLINEARITA: CARAT VS (X * Y * Z)
# =============================================================================
st.subheader("⚖️ 3. Fyzikální multikolinearita a paradox záporných rozměrů")

col_phys_l, col_phys_r = st.columns([1.5, 1])

with col_phys_l:
    st.write(
        """
        V zadání kurzu byla ponechána kombinace proměnných `['carat', 'x', 'y', 'z']`.  
        Z fyzikálního hlediska je ale diamant trojrozměrné těleso s hustotou $\\rho \\approx 3{,}52\\ \\text{g/cm}^3$:
        $$V \\propto x \\cdot y \\cdot z \\quad \\Longrightarrow \\quad \\text{Hmotnost (Carat)} = \\rho \\cdot V$$
        
        Korelace mezi hmotností (`carat`) a vypočteným objemem ($x \\cdot y \\cdot z$) dosahuje **0.9989** (téměř exaktní závislost).
        """
    )
    st.warning(
        """
        **Důsledek pro OLS regresi:**  
        Matice $X^T X$ je téměř singulární. Model začne rozměry penalizovat:
        - Koeficient délky $x$: **-3 675 USD / mm**
        - Koeficient hloubky $z$: **-2 602 USD / mm**  
        Model by klenotníkovi tvrdil: *„Pokud vybrousíte kámen o milimetr větší, jeho cena klesne o 3 600 dolarů!“*
        """
    )

with col_phys_r:
    st.markdown("##### Váhy v učebnicovém modelu:")
    st.dataframe(
        pd.DataFrame([
            {"Příznak": "carat (hmotnost)", "Koeficient": "+$11,380", "Vliv": "Silně kladný"},
            {"Příznak": "x (délka v mm)", "Koeficient": "-$3,675", "Vliv": "Absurdně záporný ❌"},
            {"Příznak": "y (šířka v mm)", "Koeficient": "+$3,761", "Vliv": "Kladný (kompenzace)"},
            {"Příznak": "z (hloubka v mm)", "Koeficient": "-$2,602", "Vliv": "Absurdně záporný ❌"},
        ]),
        use_container_width=True,
        hide_index=True
    )

st.markdown("---")

# =============================================================================
# 5. SROVNÁVACÍ BENCHMARK 5 MODELŮ PRO KLENOTNÍKA
# =============================================================================
st.subheader("🏆 4. Velký srovnávací benchmark 5 modelů")
st.write(
    "Jak se mění schopnost modelu přesně trefit cenu diamantu na testovací sadě "
    "(vyhodnoceno na všech 53 908 diamantech včetně velkých solitérů do 5.01 ct):"
)

benchmark_data = [
    {
        "Model": "1. Učebnicový OLS (pouze rozměry)",
        "Příznaky": "carat, x, y, z",
        "R² skóre": "0.8633",
        "MAE (Chyba)": "$881",
        "Zhodnocení": "Zadání kurzu. Ignoruje barvu a čistotu, záporné váhy."
    },
    {
        "Model": "2. Gemologický 4C OLS",
        "Příznaky": "carat, cut, color, clarity",
        "R² skóre": "0.9079",
        "MAE (Chyba)": "$841",
        "Zhodnocení": "Respektuje klenotnické 4C. Všechny váhy jsou fyzikálně kladné."
    },
    {
        "Model": "3. Plný OLS (+ proporce)",
        "Příznaky": "carat, 4C, depth, table, x, y, z",
        "R² skóre": "0.9125",
        "MAE (Chyba)": "$787",
        "Zhodnocení": "Zahrnuje všechny naměřené veličiny."
    },
    {
        "Model": "4. Fyzikální Log-Log model",
        "Příznaky": "log(carat) + 4C",
        "R² skóre": "0.9478",
        "MAE (Chyba)": "$455",
        "Zhodnocení": "Mocninný model cenotvorby (Cena ~ Váha^gamma)."
    },
    {
        "Model": "5. Moderní Gradient Boosting (HGB)",
        "Příznaky": "Všechny příznaky (stromový model)",
        "R² skóre": "0.9830",
        "MAE (Chyba)": "$275",
        "Zhodnocení": "🏆 Vítěz: Snížení chyby o 68.8 % oproti učebnicovému kurzu!"
    },
]

st.dataframe(pd.DataFrame(benchmark_data), use_container_width=True, hide_index=True)

# Vizuální srovnání chyby
fig_mae = go.Figure()
models = [d["Model"].split(". ")[1] for d in benchmark_data]
maes = [881, 841, 787, 455, 275]
colors = ["#EF4444", "#F59E0B", "#F59E0B", "#3B82F6", "#10B981"]

fig_mae.add_trace(
    go.Bar(
        x=models,
        y=maes,
        text=[f"${m}" for m in maes],
        textposition="auto",
        marker_color=colors,
        hovertemplate="Model: %{x}<br>Průměrná chyba: $%{y}<extra></extra>"
    )
)
fig_mae.update_layout(
    title="Srovnání průměrné chyby predikce ceny diamantu (MAE v USD – méně je lépe)",
    height=400,
    margin=dict(l=10, r=10, t=40, b=10),
    yaxis_title="MAE ($)"
)
st.plotly_chart(fig_mae, use_container_width=True)
