from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

st.set_page_config(
    page_title="Lineární regrese | Coderslab ML",
    page_icon="📈",
    layout="wide",
)

base_dir = Path(__file__).resolve().parent.parent
reg_dir = base_dir / "01_Regression"
data_dir = reg_dir / "data"
plots_dir = reg_dir / "plots"

# -----------------------------------------------------------------------------
# Caching: Načtení dat a trénování modelů
# -----------------------------------------------------------------------------
@st.cache_data
def load_kc_data():
    csv_path = data_dir / "kc_house_data_preprocessed.csv"
    if not csv_path.exists():
        csv_path = data_dir / "kc_house_data.csv"
    return pd.read_csv(csv_path)

@st.cache_resource
def train_kc_models(df):
    X = df.drop(columns=["price"])
    y = df["price"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. Standard OLS
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    ols = LinearRegression().fit(X_tr_s, y_tr)

    # 2. Log OLS
    log_model = TransformedTargetRegressor(
        regressor=Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())]),
        func=np.log1p,
        inverse_func=np.expm1,
    ).fit(X_tr, y_tr)

    # 3. Gradient Boosting
    hgb = HistGradientBoostingRegressor(random_state=42).fit(X_tr, y_tr)

    scores = {
        "OLS": {"R2": ols.score(X_te_s, y_te), "MAE": mean_absolute_error(y_te, ols.predict(X_te_s))},
        "Log-OLS": {"R2": r2_score(y_te, log_model.predict(X_te)), "MAE": mean_absolute_error(y_te, log_model.predict(X_te))},
        "HGB": {"R2": r2_score(y_te, hgb.predict(X_te)), "MAE": mean_absolute_error(y_te, hgb.predict(X_te))},
    }
    return ols, scaler, log_model, hgb, scores, list(X.columns)

@st.cache_data
def load_diamonds_data():
    csv_path = data_dir / "diamonds_preprocessed.csv"
    return pd.read_csv(csv_path)

@st.cache_resource
def train_diamonds_models(df):
    y = df["price"]
    X_all = df.drop(columns=["price"])
    X_tr, X_te, y_tr, y_te = train_test_split(X_all, y, test_size=0.2, random_state=42)

    # Baseline: pouze rozměry a karáty
    simple_cols = ["carat", "x", "y", "z"]
    m_simple = LinearRegression().fit(X_tr[simple_cols], y_tr)

    # 4C model
    m_4c = Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())]).fit(X_tr, y_tr)

    # HGB model
    m_hgb = HistGradientBoostingRegressor(random_state=42).fit(X_tr, y_tr)

    scores = {
        "Simple_OLS": {
            "R2": m_simple.score(X_te[simple_cols], y_te),
            "MAE": mean_absolute_error(y_te, m_simple.predict(X_te[simple_cols])),
        },
        "4C_OLS": {
            "R2": m_4c.score(X_te, y_te),
            "MAE": mean_absolute_error(y_te, m_4c.predict(X_te)),
        },
        "HGB": {
            "R2": r2_score(y_te, m_hgb.predict(X_te)),
            "MAE": mean_absolute_error(y_te, m_hgb.predict(X_te)),
        },
    }
    return m_simple, m_4c, m_hgb, scores, list(X_all.columns)

# -----------------------------------------------------------------------------
# Hlavička stránky
# -----------------------------------------------------------------------------
st.title("📈 Blok 1: Lineární regrese")
st.caption("Teoretické základy, praktická implementace v Scikit-learn a interaktivní predikční simulátory.")

tab_kc, tab_diamonds, tab_theory = st.tabs([
    "🏡 Cvičení 1: Ceny domů (King County)",
    "💎 Cvičení 2: Klenotník a diamanty",
    "📚 Teorie & Moderní ML (09/2026)",
])

# =============================================================================
# TAB 1: CVIČENÍ 1 – KING COUNTY HOUSING
# =============================================================================
with tab_kc:
    st.subheader("🏡 Interaktivní oceňování nemovitostí v King County (USA)")
    st.write(
        "V tomto cvičení jsme vyčistili historickou databázi prodejů nemovitostí, odstranili anomálie "
        "(např. dům s 33 ložnicemi o ploše 1620 sqft) a natrénovali lineární regresní modely."
    )

    kc_df = load_kc_data()
    ols_kc, scaler_kc, log_kc, hgb_kc, kc_scores, kc_features = train_kc_models(kc_df)

    # Horní metriky modelů
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            label="Základní OLS Regrese",
            value=f"R² = {kc_scores['OLS']['R2']:.3f}",
            delta=f"MAE: ${kc_scores['OLS']['MAE']:,.0f}",
        )
    with c2:
        st.metric(
            label="Log-Transformed OLS",
            value=f"R² = {kc_scores['Log-OLS']['R2']:.3f}",
            delta=f"MAE: ${kc_scores['Log-OLS']['MAE']:,.0f}",
        )
    with c3:
        st.metric(
            label="Gradient Boosting (2026 Benchmark)",
            value=f"R² = {kc_scores['HGB']['R2']:.3f}",
            delta=f"MAE: ${kc_scores['HGB']['MAE']:,.0f}",
        )

    st.markdown("---")
    st.markdown("### 🎛️ Živý odhadce tržní ceny nemovitosti")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        sqft_living = st.slider("Obytná plocha domu (sqft):", min_value=500, max_value=8000, value=2200, step=50)
        grade = st.slider("Kvalita stavby a materiálů (Grade 1–13):", min_value=3, max_value=13, value=8, step=1)
        bathrooms = st.slider("Počet koupelen:", min_value=1.0, max_value=6.0, value=2.5, step=0.25)
        bedrooms = st.slider("Počet ložnic:", min_value=1, max_value=8, value=3, step=1)

    with col_right:
        floors = st.selectbox("Počet pater:", [1.0, 1.5, 2.0, 2.5, 3.0], index=2)
        yr_built = st.slider("Rok výstavby:", min_value=1900, max_value=2015, value=1995, step=1)
        waterfront = st.radio("Výhled na vodu (Waterfront):", ["Ne (0)", "Ano (1)"], index=0)
        lat = st.slider("Zeměpisná šířka (lokalita Sever–Jih):", min_value=47.15, max_value=47.78, value=47.60, step=0.01)

    # Sestavení vzorku
    median_row = kc_df.drop(columns=["price"]).median().to_dict()
    median_row["sqft_living"] = sqft_living
    median_row["grade"] = grade
    median_row["bathrooms"] = bathrooms
    median_row["bedrooms"] = bedrooms
    median_row["floors"] = floors
    median_row["yr_built"] = yr_built
    median_row["waterfront"] = 1 if "Ano" in waterfront else 0
    median_row["lat"] = lat
    median_row["sqft_above"] = sqft_living * 0.85
    median_row["sqft_living15"] = sqft_living

    input_df = pd.DataFrame([median_row])[kc_features]
    input_scaled = scaler_kc.transform(input_df)

    pred_ols = ols_kc.predict(input_scaled)[0]
    pred_log = log_kc.predict(input_df)[0]
    pred_hgb = hgb_kc.predict(input_df)[0]

    st.markdown("#### 💰 Výsledný odhad ceny:")
    r1, r2, r3 = st.columns(3)
    with r1:
        st.success(f"**Lineární regrese (OLS):**\n### ${max(0, pred_ols):,.0f}")
    with r2:
        st.info(f"**Log-Normal OLS:**\n### ${max(0, pred_log):,.0f}")
    with r3:
        st.warning(f"**Gradient Boosting (HGB):**\n### ${max(0, pred_hgb):,.0f}")

    with st.expander("📊 Zobrazit diagnostické grafy z analýzy King County"):
        g1, g2 = st.columns(2)
        with g1:
            p1 = plots_dir / "01_correlation_matrix.png"
            if p1.exists():
                st.image(str(p1), caption="Korelační matice příznaků s cenou")
        with g2:
            p2 = plots_dir / "04_actual_vs_predicted_residuals.png"
            if p2.exists():
                st.image(str(p2), caption="Diagnostika reziduí (heteroskedasticita)")

# =============================================================================
# TAB 2: CVIČENÍ 2 – KLENOTNÍK A DIAMANTY
# =============================================================================
with tab_diamonds:
    st.subheader("💎 Klenotnická oceňovací kalkulačka diamantů")
    st.write(
        "Klenotník má v prodejně nával zákazníků nabízejících diamanty k výkupu. "
        "Tento model automaticky odhadne férovou tržní cenu a doporučené výkupní pásmo."
    )

    diamonds_df = load_diamonds_data()
    m_simple_d, m_4c_d, m_hgb_d, d_scores, d_features = train_diamonds_models(diamonds_df)

    d1, d2, d3 = st.columns(3)
    with d1:
        st.metric(
            label="Základní OLS (jen rozměry)",
            value=f"R² = {d_scores['Simple_OLS']['R2']:.3f}",
            delta=f"MAE: ${d_scores['Simple_OLS']['MAE']:,.0f}",
        )
    with d2:
        st.metric(
            label="OLS s 4C parametry",
            value=f"R² = {d_scores['4C_OLS']['R2']:.3f}",
            delta=f"MAE: ${d_scores['4C_OLS']['MAE']:,.0f}",
        )
    with d3:
        st.metric(
            label="Moderní Gradient Boosting",
            value=f"R² = {d_scores['HGB']['R2']:.3f}",
            delta=f"MAE: ${d_scores['HGB']['MAE']:,.0f} (Chyba jen $277!)",
        )

    st.markdown("---")
    st.markdown("### 💍 Zadejte parametry diamantu (Gemologické 4C):")

    c_left, c_right = st.columns(2)

    with c_left:
        carat = st.slider("Hmotnost v karátech (Carat):", min_value=0.20, max_value=3.50, value=1.00, step=0.01)
        cut_label = st.select_slider(
            "Kvalita brusu (Cut):",
            options=["Fair", "Good", "Very Good", "Premium", "Ideal"],
            value="Ideal",
        )
        color_label = st.select_slider(
            "Barva diamantu (Color):",
            options=["J (nejhorší)", "I", "H", "G", "F", "E", "D (nejlepší/bezbarvá)"],
            value="G",
        )
        clarity_label = st.select_slider(
            "Čistota diamantu (Clarity):",
            options=["I1 (nejhorší)", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF (nejlepší/bez vad)"],
            value="VS1",
        )

    with c_right:
        st.caption("Rozměry diamantu v milimetrech (automaticky odvozeno z karátů):")
        # Fyzikální odhad rozměrů pro kulatý briliant
        base_x = float(np.round(6.5 * (carat ** (1 / 3)), 2))
        base_y = base_x
        base_z = float(np.round(base_x * 0.615, 2))

        x_dim = st.number_input("Délka x (mm):", min_value=3.0, max_value=12.0, value=base_x, step=0.05)
        y_dim = st.number_input("Šířka y (mm):", min_value=3.0, max_value=12.0, value=base_y, step=0.05)
        z_dim = st.number_input("Hloubka z (mm):", min_value=2.0, max_value=8.0, value=base_z, step=0.05)
        depth_val = st.number_input("Celková hloubka (%):", min_value=50.0, max_value=75.0, value=61.8, step=0.1)
        table_val = st.number_input("Šířka tabulky (%):", min_value=45.0, max_value=75.0, value=57.0, step=0.5)

    cut_map = {"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4}
    color_map = {
        "J (nejhorší)": 0, "I": 1, "H": 2, "G": 3, "F": 4, "E": 5, "D (nejlepší/bezbarvá)": 6,
    }
    clarity_map = {
        "I1 (nejhorší)": 0, "SI2": 1, "SI1": 2, "VS2": 3, "VS1": 4, "VVS2": 5, "VVS1": 6, "IF (nejlepší/bez vad)": 7,
    }

    sample_diamond = pd.DataFrame([{
        "carat": carat,
        "cut": cut_map[cut_label],
        "color": color_map[color_label],
        "clarity": clarity_map[clarity_label],
        "depth": depth_val,
        "table": table_val,
        "x": x_dim,
        "y": y_dim,
        "z": z_dim,
    }])[d_features]

    pred_4c = float(m_4c_d.predict(sample_diamond)[0])
    pred_hgb_d = float(m_hgb_d.predict(sample_diamond)[0])

    st.markdown("#### 💵 Doporučené ocenění pro klenotníka:")
    v1, v2 = st.columns([1, 1])
    with v1:
        st.success(
            f"""
            **Férová tržní hodnota (AI Gradient Boosting):**  
            # ${max(300, pred_hgb_d):,.0f} USD
            """
        )
    with v2:
        buy_min = max(200, pred_hgb_d * 0.82)
        sell_rec = max(350, pred_hgb_d * 1.15)
        st.info(
            f"""
            **Doporučené výkupní a prodejní pásmo (marže klenotníka):**  
            - **Výkupní cena:** ${buy_min:,.0f} USD  
            - **Doporučená prodejní cena:** ${sell_rec:,.0f} USD
            """
        )

    with st.expander("🔍 Gemologický paradox multikolinearity v lineární regresi"):
        st.warning(
            """
            **Proč v základní lineární regresi vyšly záporné váhy pro délku x (-$3,675) a hloubku z (-$2,602)?**  
            Protože hmotnost v karátech je fyzikálně svázána s objemem kamene ($V \\approx x \\cdot y \\cdot z$). 
            Korelace mezi `carat` a odhadnutým objemem je **0.9989**. OLS se v přítomnosti takto extrémní 
            multikolinearity snaží váhy odečítat, což vede k fyzikálnímu nesmyslu.  
            *Moderní řešení:* Ponechat pouze `carat` a parametry 4C, nebo použít Gradient Boosting.
            """
        )
        p_out = plots_dir / "08_diamonds_outliers_detection.png"
        if p_out.exists():
            st.image(str(p_out), caption="Detekce odlehlých hodnot a anomálií u diamantů")

# =============================================================================
# TAB 3: TEORIE & MODERNÍ ML (09/2026)
# =============================================================================
with tab_theory:
    st.subheader("📚 Teoretický rozbor a stav ML světa k 09/2026")

    theory_md_path = reg_dir / "theory" / "01_linear_regression_theory.md"
    if theory_md_path.exists():
        with open(theory_md_path, "r", encoding="utf-8") as f:
            content = f.read()
        st.markdown(content)
    else:
        st.info("Teoretický dokument se připravuje...")
