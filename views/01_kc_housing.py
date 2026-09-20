from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from sklearn.compose import TransformedTargetRegressor
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

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    ols = LinearRegression().fit(X_tr_s, y_tr)

    log_model = TransformedTargetRegressor(
        regressor=Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())]),
        func=np.log1p,
        inverse_func=np.expm1,
    ).fit(X_tr, y_tr)

    hgb = HistGradientBoostingRegressor(random_state=42).fit(X_tr, y_tr)

    scores = {
        "OLS": {"R2": ols.score(X_te_s, y_te), "MAE": mean_absolute_error(y_te, ols.predict(X_te_s))},
        "Log-OLS": {"R2": r2_score(y_te, log_model.predict(X_te)), "MAE": mean_absolute_error(y_te, log_model.predict(X_te))},
        "HGB": {"R2": r2_score(y_te, hgb.predict(X_te)), "MAE": mean_absolute_error(y_te, hgb.predict(X_te))},
    }
    return ols, scaler, log_model, hgb, scores, list(X.columns), X_te_s, y_te

st.title("🏡 Cvičení 1: Odhad cen nemovitostí (King County, USA)")
st.caption("Predikce tržních cen domů na základě plochy, kvality stavby, lokality a počtu místností.")

kc_df = load_kc_data()
ols_kc, scaler_kc, log_kc, hgb_kc, kc_scores, kc_features, X_te_s_kc, y_te_kc = train_kc_models(kc_df)

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
        label="Moderní Gradient Boosting (2026)",
        value=f"R² = {kc_scores['HGB']['R2']:.3f}",
        delta=f"MAE: ${kc_scores['HGB']['MAE']:,.0f}",
    )

st.markdown("---")
st.subheader("🎛️ Interaktivní kalkulátor ceny nemovitosti")

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

st.markdown("#### 💰 Výsledný odhad tržní ceny:")
r1, r2, r3 = st.columns(3)
with r1:
    st.success(f"**Lineární regrese (OLS):**\n### {max(0, pred_ols):,.0f} USD")
with r2:
    st.info(f"**Log-Normal OLS:**\n### {max(0, pred_log):,.0f} USD")
with r3:
    st.warning(f"**Gradient Boosting (HGB):**\n### {max(0, pred_hgb):,.0f} USD")

st.markdown("---")

# =============================================================================
# INTERAKTIVNÍ DASHBOARD POD SEBOU (FULL WIDTH)
# =============================================================================
st.subheader("📊 Diagnostický dashboard a analýza dat")
st.write(
    "Jednotlivé grafy jsou umístěny samostatně na plnou šířku. Jsou plně **interaktivní (Plotly)** – "
    "můžete v nich zoomovat, najíždět myší na konkrétní body a v pravém horním rohu každého grafu "
    "kliknout na ikonu **fullscreen** pro zvětšení na celou obrazovku."
)

# 1. GRAF: KORELAČNÍ MATICE (NA CELOU ŠÍŘKU)
st.markdown("### 1. Interaktivní korelační matice příznaků")
st.caption("Přejeďte myší přes buňky pro zobrazení přesného Pearsonova korelačního koeficientu mezi veličinami.")

corr = kc_df.corr()
fig_corr = px.imshow(
    corr,
    text_auto=".2f",
    aspect="auto",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
)
fig_corr.update_layout(
    height=650,
    margin=dict(l=10, r=10, t=30, b=10),
    font=dict(size=11),
)
st.plotly_chart(fig_corr, width="stretch")

with st.expander("🖼️ Zobrazit původní statický graf (PNG) v tiskové kvalitě"):
    p1 = plots_dir / "01_correlation_matrix.png"
    if p1.exists():
        st.image(str(p1), caption="Původní statický Seaborn heatmap", width="stretch")

st.markdown("---")

# 2. GRAF: DIAGNOSTIKA REZIDUÍ A HOMOSKEDASTICITY (NA CELOU ŠÍŘKU)
st.markdown("### 2. Analýza chyb a reziduí (Homoskedasticita & Normalita)")
st.caption(
    "Vlevo: Závislost rezidua (chyby) na predikované ceně. Zřetelný trychtýřovitý rozptyl dokazuje heteroskedasticitu "
    "(u dražších domů dělá lineární model výrazně větší chyby). Vpravo: Histogram rozdělení chyb."
)

y_pred_kc = ols_kc.predict(X_te_s_kc)
residuals_kc = y_te_kc - y_pred_kc

# Vzorek 2000 bodů pro svižné vykreslení
plot_sample_idx = np.random.RandomState(42).choice(len(y_pred_kc), size=min(2000, len(y_pred_kc)), replace=False)
sample_preds = y_pred_kc[plot_sample_idx]
sample_res = residuals_kc.iloc[plot_sample_idx]

fig_res = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Rezidua vs Predikovaná cena (Ověření rozptylu)", "Rozdělení chyb predikce (Normalita)"),
    horizontal_spacing=0.1
)

# Scatter plot
fig_res.add_trace(
    go.Scatter(
        x=sample_preds,
        y=sample_res,
        mode="markers",
        marker=dict(size=5, color="#8B5CF6", opacity=0.4),
        name="Domy (vzorek)",
        hovertemplate="Predikce: $%{x:,.0f}<br>Chyba: $%{y:,.0f}<extra></extra>"
    ),
    row=1, col=1
)
fig_res.add_hline(y=0, line_dash="dash", line_color="red", line_width=1.5, row=1, col=1)

# Histogram
fig_res.add_trace(
    go.Histogram(
        x=residuals_kc,
        nbinsx=50,
        marker_color="#3B82F6",
        name="Chyby",
        opacity=0.75,
        hovertemplate="Chyba: $%{x:,.0f}<br>Počet: %{y}<extra></extra>"
    ),
    row=1, col=2
)

fig_res.update_layout(
    height=500,
    showlegend=False,
    margin=dict(l=10, r=10, t=40, b=10),
)
fig_res.update_xaxes(title_text="Predikovaná cena ($)", row=1, col=1)
fig_res.update_yaxes(title_text="Reziduum (Skutečnost - Predikce)", row=1, col=1)
fig_res.update_xaxes(title_text="Chyba ($)", row=1, col=2)
fig_res.update_yaxes(title_text="Počet domů", row=1, col=2)

st.plotly_chart(fig_res, width="stretch")

with st.expander("🖼️ Zobrazit původní statický diagnostický graf reziduí"):
    p2 = plots_dir / "04_actual_vs_predicted_residuals.png"
    if p2.exists():
        st.image(str(p2), caption="Původní statický graf reziduí", width="stretch")
