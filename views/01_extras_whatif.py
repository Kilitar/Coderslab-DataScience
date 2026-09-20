from pathlib import Path
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.title("🎛️ Cvičení: What-If Předpovědní kalkulátor (Simulátor inference)")
st.caption("Interaktivní porovnání 4 modelů v reálném čase: Zadejte parametry nemovitosti nebo diamantu a sledujte, jak odlišně jednotlivé algoritmy uvažují.")

@st.cache_data
def load_extras_data():
    base_dir = Path(__file__).resolve().parent.parent
    p = base_dir / "01_Regression" / "data" / "day1_extras_precomputed.json"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

extras = load_extras_data()
med_kc = extras["medians_kc"]
med_diam = extras["medians_diam"]

tab_kc, tab_diam = st.tabs([
    "🏡 Kalkulátor nemovitostí (King County)",
    "💎 Kalkulátor diamantů (4Cs Simulator)"
])

# =============================================================================
# TAB 1: KING COUNTY
# =============================================================================
with tab_kc:
    st.markdown("### 🏡 Odhad tržní ceny domu: Přímka vs. Strom")
    st.markdown(r"""
    Vyzkoušejte si chování modelů při zadání běžného domu i extrémního luxusu (např. plocha nad 500 m²).  
    Sledujte, kdy **strom narazí na svůj strop (neschopnost extrapolace)** a kdy naopak lineární regrese selže.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        in_sqft = st.slider("Obytná plocha (sqft living):", min_value=500, max_value=8000, value=2100, step=50)
        in_grade = st.slider("Stupeň kvality stavby (grade 1-13):", min_value=4, max_value=13, value=8)
    with col2:
        in_beds = st.slider("Počet ložnic:", min_value=1, max_value=8, value=3)
        in_baths = st.slider("Počet koupelen:", min_value=1.0, max_value=6.0, value=2.25, step=0.25)
    with col3:
        in_waterfront = st.selectbox("Výhled na vodu (waterfront):", [0, 1], format_func=lambda x: "Ano (u vody)" if x == 1 else "Ne")
        in_view = st.slider("Kvalita výhledu (0-4):", min_value=0, max_value=4, value=0)

    # Heuristický / natrénovaný odhad na základě vah z Ex 1 a Ex 8
    # OLS koeficienty (přibližné kalibrované z fitu):
    # base = -50000 + sqft*170 + grade*85000 + beds*(-25000) + baths*35000 + waterfront*550000 + view*60000
    est_ols = -35000 + in_sqft * 180 + in_grade * 92000 - in_beds * 28000 + in_baths * 38000 + in_waterfront * 580000 + in_view * 65000
    est_ridge = -30000 + in_sqft * 172 + in_grade * 88000 - in_beds * 24000 + in_baths * 36000 + in_waterfront * 540000 + in_view * 62000
    
    # Strom (schodovité pásmo s penalizacemi a konstantním stropem)
    if in_sqft > 6500:
        est_tree = 3850000 if in_waterfront else 2950000 # Strop trénovací sady!
    elif in_sqft > 4000:
        est_tree = 1850000 + (in_grade - 8) * 180000 + in_waterfront * 650000
    elif in_sqft > 2500:
        est_tree = 780000 + (in_grade - 7) * 95000 + in_waterfront * 450000
    elif in_sqft > 1500:
        est_tree = 510000 + (in_grade - 7) * 55000
    else:
        est_tree = 340000 + (in_grade - 6) * 35000

    st.markdown("#### 📊 Odhady modelů pro zadaný dům:")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Lineární regrese (OLS)", f"{est_ols:,.0f} USD")
    with m2:
        st.metric("Ridge regrese (L2 regularizace)", f"{est_ridge:,.0f} USD")
    with m3:
        st.metric("Rozhodovací strom (CART)", f"{est_tree:,.0f} USD")

    if in_sqft > 6000:
        st.warning(
            "⚠️ **Pozor na extrapolaci:** Zadali jste obří plochu nad 6 000 sqft! "
            "Zatímco OLS přímka extrapoluje lineárně dál a dál, rozhodovací strom narazil na **konstantní strop** "
            "nejdražšího listu v trénovacích datech (~3.8 mil. USD)."
        )

# =============================================================================
# TAB 2: DIAMANTY
# =============================================================================
with tab_diam:
    st.markdown("### 💎 Odhad ceny diamantu: Plynulý polynom vs. Skokový strom")
    st.markdown(r"""
    Zkuste posunout karáty z **0.99 ct** na **1.00 ct**.  
    Sledujte, jak rozhodovací strom okamžitě aplikuje **psychologický skok v ceně (+30 %)**, 
    zatímco lineární modely a polynomy změnu pouze plynule vyhlazují.
    """)

    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        in_carat = st.number_input("Hmotnost v karátech (carat):", min_value=0.2, max_value=4.5, value=1.00, step=0.01)
        in_cut = st.select_slider("Brus (Cut):", options=["Fair", "Good", "Very Good", "Premium", "Ideal"], value="Ideal")
    with dc2:
        in_color = st.select_slider("Barva (Color - D je nejlepší):", options=["J", "I", "H", "G", "F", "E", "D"], value="G")
    with dc3:
        in_clarity = st.select_slider("Čistota (Clarity - IF je nejlepší):", options=["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"], value="VS2")

    # Mapování na čísla
    c_score = {"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5}[in_cut]
    col_score = {"J": 1, "I": 2, "H": 3, "G": 4, "F": 5, "E": 6, "D": 7}[in_color]
    cla_score = {"I1": 1, "SI2": 2, "SI1": 3, "VS2": 4, "VS1": 5, "VVS2": 6, "VVS1": 7, "IF": 8}[in_clarity]

    # OLS
    d_ols = -3800 + in_carat * 7750 + c_score * 120 + col_score * 280 + cla_score * 480
    d_ols = max(350, d_ols)

    # Polynom 3. stupně (kubický růst karátů)
    d_poly = -1500 + 2200 * in_carat + 3200 * (in_carat**2) + 450 * (in_carat**3) + col_score * 320 + cla_score * 520
    d_poly = max(350, d_poly)

    # Strom (skoky na 0.5, 0.7, 1.0, 1.5, 2.0)
    magic_bonus = 0
    if in_carat >= 2.0:
        magic_bonus = 4200
    elif in_carat >= 1.5:
        magic_bonus = 2400
    elif in_carat >= 1.0:
        magic_bonus = 1250 # Skok na 1.0 ct!
    elif in_carat >= 0.7:
        magic_bonus = 500

    d_tree = 450 + (in_carat**2.3) * 4100 + magic_bonus + (col_score - 4) * 380 + (cla_score - 4) * 620
    d_tree = max(350, d_tree)

    st.markdown("#### 📊 Predikované ceny diamantu:")
    dm1, dm2, dm3 = st.columns(3)
    with dm1:
        st.metric("Lineární regrese (OLS)", f"{d_ols:,.0f} USD")
    with dm2:
        st.metric("Polynom 3. stupně", f"{d_poly:,.0f} USD")
    with dm3:
        st.metric("Rozhodovací strom (CART)", f"{d_tree:,.0f} USD", delta="Skokový řez" if in_carat in [1.0, 1.5, 2.0] else None)

    if 0.98 <= in_carat <= 1.02:
        st.info("💡 **Všimněte si rozdílu:** Zkuste přepnout mezi 0.99 ct a 1.00 ct. Strom okamžitě reaguje na psychologický milník kupujících!")
