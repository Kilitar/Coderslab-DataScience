from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

base_dir = Path(__file__).resolve().parent.parent
reg_dir = base_dir / "01_Regression"
data_dir = reg_dir / "data"
plots_dir = reg_dir / "plots"

@st.cache_data
def load_diamonds_data():
    csv_path = data_dir / "diamonds_preprocessed.csv"
    return pd.read_csv(csv_path)

@st.cache_resource
def train_diamonds_models(df):
    y = df["price"]
    X_all = df.drop(columns=["price"])
    X_tr, X_te, y_tr, y_te = train_test_split(X_all, y, test_size=0.2, random_state=42)

    simple_cols = ["carat", "x", "y", "z"]
    m_simple = LinearRegression().fit(X_tr[simple_cols], y_tr)

    m_4c = Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())]).fit(X_tr, y_tr)
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

st.title("💎 Cvičení 2: Klenotník a oceňování diamantů")
st.caption("Automatizace odhadu hodnoty drahých kamenů pro klenotníka na základě gemologických parametrů 4C.")

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
st.subheader("💍 Zadejte parametry diamantu (Gemologické 4C):")

c_left, c_right = st.columns(2)

with c_left:
    carat = st.slider("Hmotnost v karátech (Carat):", min_value=0.20, max_value=5.01, value=1.00, step=0.01)
    if carat > 3.5:
        st.warning("⚠️ **Pozor na rozsah trénovacích dat:** Kameny nad 3.5 ct byly z předzpracovaného trénovacího datasetu odstraněny jako extrémní odlehlé hodnoty. Predikce pro > 3.5 ct představuje extrapolaci mimo trénovací doménu!")

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

pred_hgb_d = float(m_hgb_d.predict(sample_diamond)[0])

st.markdown("#### 💵 Doporučené ocenění pro klenotníka:")
v1, v2 = st.columns([1, 1])
with v1:
    st.success(
        f"""
        **Výukový odhad tržní hodnoty (HistGradientBoosting):**  
        # {max(300, pred_hgb_d):,.0f} USD  
        *(Model natrénován na historickém vzorku 53 908 diamantů)*
        """
    )
with v2:
    buy_min = max(200, pred_hgb_d * 0.82)
    sell_rec = max(350, pred_hgb_d * 1.15)
    st.info(
        f"""
        **Ilustrativní obchodní pásmo (marže klenotníka):**  
        - **Výkupní cena (-18 %):** {buy_min:,.0f} USD  
        - **Doporučená prodejní cena (+15 %):** {sell_rec:,.0f} USD  
        *(Ilustrativní modelová heuristika, nikoli statistický interval spolehlivosti)*
        """
    )

st.markdown("---")
st.subheader("🔍 Gemologický paradox multikolinearity a analýza dat")
st.warning(
    """
    **Proč v základní lineární regresi vyšly záporné váhy pro délku x (\u20133 675 USD/mm) a hloubku z (\u20132 602 USD/mm)?**  
    Protože hmotnost v karátech je fyzikálně svázána s objemem kamene ($V \\approx x \\cdot y \\cdot z$). 
    Korelace mezi `carat` a odhadnutým objemem je **0.9989**. OLS se v přítomnosti takto extrémní 
    multikolinearity snaží váhy odečítat, což vede k fyzikálnímu nesmyslu.  
    *Moderní řešení:* Ponechat pouze `carat` a parametry 4C, nebo použít Gradient Boosting.
    """
)

# Interaktivní grafy diamantů
st.markdown("### 📊 Interaktivní diagnostika a anomálie diamantů")
st.caption("Grafy jsou uspořádány samostatně na plnou šířku pro maximální přehlednost.")

# 1. Carat vs Price interaktivní scatter
st.markdown("#### 1. Nelineární mocninný vztah: Hmotnost v karátech vs. Cena diamantu")
sample_d = diamonds_df.sample(n=min(2500, len(diamonds_df)), random_state=42)
fig_carat = px.scatter(
    sample_d,
    x="carat",
    y="price",
    color="clarity",
    title="Carat vs. Cena (zřetelný exponenciální růst)",
    opacity=0.5,
    hover_data=["cut", "color", "x", "y", "z"],
    color_continuous_scale="Viridis",
)
fig_carat.update_layout(height=500, margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig_carat, width="stretch")

# 2. Původní statický graf anomálií
with st.expander("🖼️ Zobrazit detekci odlehlých hodnot a anomálií (nulové rozměry a překlepy)"):
    p_out = plots_dir / "08_diamonds_outliers_detection.png"
    if p_out.exists():
        st.image(str(p_out), caption="Detekce odlehlých hodnot (x, y, z anomálie)", width="stretch")
