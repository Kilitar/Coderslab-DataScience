from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, median_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score

st.title("📐 Cvičení 3: Expertní analýza metrik, Cross-Validation & MLOps")
st.caption("Kritické zhodnocení metodiky kurzu: Bug v signatuře funkcí, past jediného splitu, MedAE vs RMSE a byznysové metriky (WAPE).")

# =============================================================================
# NAČTENÍ DAT A VÝPOČTY
# =============================================================================
@st.cache_data
def run_extended_metrics_analysis():
    base_dir = Path(__file__).resolve().parent.parent
    csv_path = base_dir / "01_Regression" / "data" / "kc_house_data_preprocessed.csv"
    if not csv_path.exists():
        csv_path = base_dir / "data" / "MAL_downloadable materials_session 1" / "Day 1" / "kc_house_data_preprocessed.csv"

    df = pd.read_csv(csv_path)
    X = df.drop(columns=["price"])
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Základní model OLS
    ols = LinearRegression().fit(X_train, y_train)
    y_pred = ols.predict(X_test)
    residuals = y_test - y_pred
    abs_errors = np.abs(residuals)

    # Robustní a byznysové metriky
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    medae = median_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100.0
    wape = np.sum(abs_errors) / np.sum(y_test) * 100.0

    # 5-Fold Cross-Validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2_scores = cross_val_score(LinearRegression(), X, y, cv=cv, scoring="r2")
    cv_mae_scores = -cross_val_score(LinearRegression(), X, y, cv=cv, scoring="neg_mean_absolute_error")

    # Multi-Model srovnání
    log_ols = TransformedTargetRegressor(
        regressor=Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())]),
        func=np.log1p, inverse_func=np.expm1
    ).fit(X_train, y_train)
    y_pred_log = log_ols.predict(X_test)

    hgb = HistGradientBoostingRegressor(random_state=42).fit(X_train, y_train)
    y_pred_hgb = hgb.predict(X_test)

    return {
        "r2": r2, "mae": mae, "rmse": rmse, "medae": medae, "mape": mape, "wape": wape,
        "cv_r2": cv_r2_scores, "cv_mae": cv_mae_scores,
        "y_test": y_test, "y_pred": y_pred, "abs_errors": abs_errors,
        "y_pred_log": y_pred_log, "y_pred_hgb": y_pred_hgb
    }

res = run_extended_metrics_analysis()

# =============================================================================
# 1. ČTYŘI METODICKÉ VÝHRADY KE KURZU
# =============================================================================
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.error("🐛 1. Bug v argumentech")
    st.caption("Slajdy kurzu uvádějí chybné pořadí `(y_pred, y_true)`. Scikit-learn vyžaduje striktně `(y_true, y_pred)`.")
with m2:
    st.warning("🎲 2. Past jednoho splitu")
    st.caption("Náhodný split dává iluzi jistoty ($R^2=0.7118$). 5-Fold CV ukazuje rozptyl od 0.6926 do 0.7132.")
with m3:
    st.info("📉 3. MAE vs MedAE")
    st.caption("MAE je 117k USD, ale MedAE je jen 84.5k USD! U 50 % domů je odchylka mnohem menší než průměr.")
with m4:
    st.success("💼 4. Byznysová metrika")
    st.caption("Říkat managementu MSE (30 mld USD²) nemá smysl. WAPE = 22.2 % dává jasný finanční kontext.")

st.markdown("---")

# =============================================================================
# 2. DETAIL: CHYBA V PREZENTACI KURZU (ARGUMENT ORDER BUG)
# =============================================================================
st.subheader("🐛 1. Kritická oprava: Chybná signatura v prezentaci kurzu")
st.write(
    """
    V oficiální prezentaci ke cvičení (*Metrics of regression models...*) autoři uvádějí kód:
    """
)

col_err, col_fix = st.columns(2)
with col_err:
    st.markdown("**❌ Chybné pořadí v prezentaci kurzu:**")
    st.code(
        """# CHYBA ZE SLAJDŮ KURZU:
r2 = r2_score(y_pred, y_true)
mse = mean_squared_error(y_pred, y_true)
mae = mean_absolute_error(y_pred, y_true)""",
        language="python"
    )
    st.caption("U symetrických metrik náhodou projde, ale u asymetrických vede k fatálním chybám.")

with col_fix:
    st.markdown("**✅ Správná signatura Scikit-Learn (API Standard):**")
    st.code(
        """# SPRÁVNÝ STANDARD SCIKIT-LEARN:
r2 = r2_score(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)""",
        language="python"
    )
    st.caption("Vždy nejprve skutečnost (Ground Truth), poté odhad (Prediction).")

st.markdown("---")

# =============================================================================
# 3. K-FOLD CROSS-VALIDATION (5 FOLDŮ)
# =============================================================================
st.subheader("🎲 2. K-Fold Cross-Validation: Rozptyl metriky v realitě")
st.write(
    """
    Zadání kurzu hodnotí model na jediném náhodném rozdělení 80/20, kde vyšlo $R^2 = 0{,}7118$.  
    Zde je důkaz, proč je jediný split ošidný – **výsledky 5-násobné křížové validace (5-Fold CV)**:
    """
)

cv_df = pd.DataFrame({
    "Fold": [f"Fold {i+1}" for i in range(5)],
    "R² skóre": [f"{s:.4f}" for s in res["cv_r2"]],
    "MAE ($)": [f"${s:,.0f}" for s in res["cv_mae"]],
})

c_tbl, c_chart = st.columns([1, 1.6])
with c_tbl:
    st.dataframe(cv_df, use_container_width=True, hide_index=True)
    st.metric("Průměrné CV R²", f"{res['cv_r2'].mean():.4f} ± {res['cv_r2'].std():.4f}")
    st.metric("Průměrné CV MAE", f"${res['cv_mae'].mean():,.0f} ± ${res['cv_mae'].std():,.0f}")

with c_chart:
    fig_cv = go.Figure()
    fig_cv.add_trace(go.Bar(
        x=[f"Fold {i+1}" for i in range(5)],
        y=res["cv_r2"],
        marker_color=["#3B82F6", "#3B82F6", "#10B981", "#EF4444", "#3B82F6"],
        text=[f"{s:.4f}" for s in res["cv_r2"]],
        textposition="auto"
    ))
    fig_cv.add_hline(
        y=res["cv_r2"].mean(), line_dash="dash", line_color="orange",
        annotation_text=f"Průměr CV: {res['cv_r2'].mean():.4f}"
    )
    fig_cv.update_layout(
        title="Stabilita R² napříč 5 foldy (Min: 0.6926 vs Max: 0.7132)",
        height=320, margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(range=[0.65, 0.75])
    )
    st.plotly_chart(fig_cv, use_container_width=True)

st.markdown("---")

# =============================================================================
# 4. MEDIAN ABSOLUTE ERROR (MedAE) VS MAE VS RMSE
# =============================================================================
st.subheader("📊 3. Skrytá realita chyb: MedAE (84.5k USD) vs. MAE (117k USD) vs. RMSE (175k USD)")
st.write(
    """
    Proč je mezi jednotlivými metrikami tak dramatický rozdíl?  
    Graf níže ukazuje **distribuci absolutních chyb** pro všechny domy v testovací sadě.
    """
)

fig_dist = go.Figure()
fig_dist.add_trace(go.Histogram(
    x=res["abs_errors"],
    nbinsx=80,
    marker_color="#8B5CF6",
    opacity=0.7,
    name="Absolutní chyba",
    hovertemplate="Chyba: $%{x:,.0f}<br>Počet domů: %{y}<extra></extra>"
))

fig_dist.add_vline(x=res["medae"], line_color="#10B981", line_width=2.5, line_dash="solid",
                  annotation_text=f"MedAE: ${res['medae']:,.0f} (50 % domů)", annotation_position="top left")
fig_dist.add_vline(x=res["mae"], line_color="#F59E0B", line_width=2.5, line_dash="dash",
                  annotation_text=f"MAE: ${res['mae']:,.0f} (Průměr L1)", annotation_position="top right")
fig_dist.add_vline(x=res["rmse"], line_color="#EF4444", line_width=2.5, line_dash="dot",
                  annotation_text=f"RMSE: ${res['rmse']:,.0f} (Kvadratická L2)", annotation_position="top right")

fig_dist.update_layout(
    title="Distribuce chyb modelu a poloha metrik MedAE, MAE a RMSE",
    height=420, margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(title="Velikost chyby (USD)", range=[0, 500000]),
    yaxis=dict(title="Počet domů")
)
st.plotly_chart(fig_dist, use_container_width=True)

st.info(
    f"""
    💡 **Zjištění pro management:**  
    - **MedAE = {res['medae']:,.0f} USD:** Celá **polovina všech domů** na trhu je oceněna s chybou **menší než 84 500 USD**!  
    - **MAE = {res['mae']:,.0f} USD:** Průměr je tažen nahoru několika luxusními vilami.  
    - **RMSE = {res['rmse']:,.0f} USD:** Kvadratické umocnění dává chybě u vily za 2 miliony dolarů obrovskou váhu.
    """
)

st.markdown("---")

# =============================================================================
# 5. BYZNYSOVÉ METRIKY: WAPE & MAPE
# =============================================================================
st.subheader("💼 4. Finanční a byznysové metriky: WAPE & MAPE")

b1, b2 = st.columns(2)
with b1:
    st.metric(
        "WAPE (Weighted Absolute Percentage Error)",
        f"{res['wape']:.2f} %",
        help="Celková vážená chyba vůči celkovému obratu trhu."
    )
    st.write(
        "Vedení realitní kanceláře nerozumí číslům v dolarech na druhou. "
        "WAPE říká jasnou řeč: **Model se v průměru odchyluje o 22.2 % z celkového finančního objemu portfolia.**"
    )

with b2:
    st.metric(
        "MAPE (Mean Absolute Percentage Error)",
        f"{res['mape']:.2f} %",
        help="Průměrná procentuální chyba jednotlivých domů."
    )
    st.write(
        "MAPE měří relativní chybu per nemovitost. U levných domů je procentuální chyba přirozeně vyšší, "
        "u drahých vil nižší."
    )
