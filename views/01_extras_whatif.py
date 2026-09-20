from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

st.title("🎛️ Cvičení: What-If Předpovědní kalkulátor (Simulátor inference)")
st.caption("Interaktivní porovnání reálně natrénovaných modelů Scikit-learn v reálném čase: Zadejte parametry a sledujte, jak odlišně jednotlivé algoritmy uvažují.")

@st.cache_resource(show_spinner="Načítám a trénuji modely pro King County...")
def get_kc_models():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "01_Regression" / "data" / "kc_house_data.csv"
    df = pd.read_csv(data_path)
    features = ["sqft_living", "grade", "bedrooms", "bathrooms", "waterfront", "view"]
    X = df[features]
    y = df["price"]

    ols = LinearRegression().fit(X, y)
    ridge = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=100.0))]).fit(X, y)
    tree = DecisionTreeRegressor(max_depth=7, min_samples_leaf=15, random_state=42).fit(X, y)
    return ols, ridge, tree, features

@st.cache_resource(show_spinner="Načítám a trénuji modely pro Diamanty...")
def get_diamond_models():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "01_Regression" / "data" / "diamonds.csv"
    df = pd.read_csv(data_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    clean_df = df[(df["x"] > 0) & (df["y"] > 0) & (df["z"] > 0)].copy()

    cut_order = {"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5}
    color_order = {"J": 1, "I": 2, "H": 3, "G": 4, "F": 5, "E": 6, "D": 7}
    clarity_order = {"I1": 1, "SI2": 2, "SI1": 3, "VS2": 4, "VS1": 5, "VVS2": 6, "VVS1": 7, "IF": 8}

    clean_df["cut"] = clean_df["cut"].map(cut_order)
    clean_df["color"] = clean_df["color"].map(color_order)
    clean_df["clarity"] = clean_df["clarity"].map(clarity_order)

    features = ["carat", "cut", "color", "clarity"]
    X = clean_df[features]
    y = clean_df["price"]

    ols = LinearRegression().fit(X, y)
    poly = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=100.0))
    ]).fit(X, y)
    tree = DecisionTreeRegressor(max_depth=8, min_samples_leaf=20, random_state=42).fit(X, y)
    return ols, poly, tree, features, cut_order, color_order, clarity_order

kc_ols, kc_ridge, kc_tree, kc_features = get_kc_models()
d_ols, d_poly, d_tree, d_features, cut_order, color_order, clarity_order = get_diamond_models()

tab_kc, tab_diam = st.tabs([
    "🏡 Kalkulátor nemovitostí (King County)",
    "💎 Kalkulátor diamantů (4Cs Simulator)"
])

# =============================================================================
# TAB 1: KING COUNTY
# =============================================================================
with tab_kc:
    st.markdown("### 🏡 Odhad tržní ceny domu: Skutečné modely Scikit-learn")
    st.markdown(r"""
    Tento simulátor volá **skutečné natrénované Scikit-learn modely** (`predict()`) na reálné databázi 21 613 nemovitostí z King County.  
    Vyzkoušejte si chování modelů při zadání běžného domu i extrémního luxusu (např. plocha nad 6 000 sqft).  
    Sledujte, kdy **strom narazí na svůj strop (neschopnost extrapolace)** a kdy naopak lineární regrese extrapoluje do obřích výšek.
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

    # Vytvoření vstupního DataFrame pro modely
    input_kc = pd.DataFrame([{
        "sqft_living": in_sqft,
        "grade": in_grade,
        "bedrooms": in_beds,
        "bathrooms": in_baths,
        "waterfront": in_waterfront,
        "view": in_view
    }])[kc_features]

    # Skutečné predikce
    est_ols = float(kc_ols.predict(input_kc)[0])
    est_ridge = float(kc_ridge.predict(input_kc)[0])
    est_tree = float(kc_tree.predict(input_kc)[0])

    st.markdown("#### 📊 Reálné predikce modelů pro zadanou nemovitost:")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Lineární regrese (OLS)", f"{est_ols:,.0f} USD")
    with m2:
        st.metric("Ridge regrese (L2 regularizace)", f"{est_ridge:,.0f} USD")
    with m3:
        st.metric("Rozhodovací strom (CART)", f"{est_tree:,.0f} USD")

    if in_sqft >= 5500:
        st.warning(
            "⚠️ **Pozor na extrapolaci:** Zadali jste plochu nad 5 500 sqft! "
            "Zatímco OLS přímka extrapoluje lineárně dál a dál, rozhodovací strom narazil na **konstantní strop** "
            "nejdražšího listu v trénovacích datech."
        )

# =============================================================================
# TAB 2: DIAMANTY
# =============================================================================
with tab_diam:
    st.markdown("### 💎 Odhad ceny diamantu: Plynulý polynom vs. Skokový strom")
    st.markdown(r"""
    Tento simulátor volá **skutečně natrénované Scikit-learn modely** na 53 940 diamantech.  
    Zkuste posunout karáty z **0.99 ct** na **1.00 ct**.  
    Sledujte, jak rozhodovací strom okamžitě reaguje na **psychologický skok v ceně**, 
    zatímco hladký polynom 2. stupně změnu pouze plynule vyhlazuje.
    """)

    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        in_carat = st.number_input("Hmotnost v karátech (carat):", min_value=0.2, max_value=4.5, value=1.00, step=0.01)
        in_cut = st.select_slider("Brus (Cut):", options=["Fair", "Good", "Very Good", "Premium", "Ideal"], value="Ideal")
    with dc2:
        in_color = st.select_slider("Barva (Color - D je nejlepší):", options=["J", "I", "H", "G", "F", "E", "D"], value="G")
    with dc3:
        in_clarity = st.select_slider("Čistota (Clarity - IF je nejlepší):", options=["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"], value="VS2")

    input_diam = pd.DataFrame([{
        "carat": in_carat,
        "cut": cut_order[in_cut],
        "color": color_order[in_color],
        "clarity": clarity_order[in_clarity]
    }])[d_features]

    pred_d_ols = float(d_ols.predict(input_diam)[0])
    pred_d_poly = float(d_poly.predict(input_diam)[0])
    pred_d_tree = float(d_tree.predict(input_diam)[0])

    # Ošetření teoretického záporu u lineárních modelů při extrémně nízkých karátech
    pred_d_ols_clamped = max(100.0, pred_d_ols)
    pred_d_poly_clamped = max(100.0, pred_d_poly)

    st.markdown("#### 📊 Reálné predikce modelů pro zadaný diamant:")
    dm1, dm2, dm3 = st.columns(3)
    with dm1:
        st.metric(
            "Lineární regrese (OLS)",
            f"{pred_d_ols_clamped:,.0f} USD",
            delta=f"Surový odhad: {pred_d_ols:,.0f} USD" if pred_d_ols < 100 else None
        )
    with dm2:
        st.metric("Polynom 2. stupně + Ridge", f"{pred_d_poly_clamped:,.0f} USD")
    with dm3:
        st.metric("Rozhodovací strom (CART)", f"{pred_d_tree:,.0f} USD")

    if 0.98 <= in_carat <= 1.02:
        st.info("💡 **Všimněte si rozdílu:** Zkuste přepnout mezi 0.99 ct a 1.00 ct. Všimněte si, jak strom i polynom reagují na změnu.")
