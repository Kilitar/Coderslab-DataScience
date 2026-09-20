import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

st.title("🎛️ Cvičení 5: Regularizace v lineární regresi (Lasso L1 & Ridge L2)")
st.caption("Analýza vlivu hyperparametru síly regularizace α, eliminace multikolineárních příznaků a srovnání s OLS.")

# =============================================================================
# VÝPOČET A CACHOVÁNÍ VÝSLEDKŮ (OKAMŽITÉ NAČTENÍ Z PŘEDPOČÍTANÝCH DAT)
# =============================================================================
@st.cache_data
def load_and_run_regularization():
    base_dir = Path(__file__).resolve().parent.parent
    precomputed_path = base_dir / "01_Regression" / "data" / "kc_regularization_precomputed.json"

    if precomputed_path.exists():
        with open(precomputed_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        lasso_models = {
            float(k): {
                "coefs": np.array(v["coefs"]),
                "zeroed": v["zeroed"],
                "metrics": v["metrics"]
            }
            for k, v in data["lasso_models"].items()
        }
        ridge_models = {
            float(k): {
                "coefs": np.array(v["coefs"]),
                "zeroed": v["zeroed"],
                "metrics": v["metrics"]
            }
            for k, v in data["ridge_models"].items()
        }

        return (
            data["feature_names"],
            data["ols_metrics"],
            lasso_models,
            pd.DataFrame(data["lasso_metrics_list"]),
            ridge_models,
            pd.DataFrame(data["ridge_metrics_list"]),
            data["n_tr"],
            data["n_te"],
            data["k_feat"],
        )

    csv_path = base_dir / "01_Regression" / "data" / "kc_house_data_preprocessed.csv"
    if not csv_path.exists():
        csv_path = base_dir / "data" / "MAL_downloadable materials_session 1" / "Day 1" / "kc_house_data_preprocessed.csv"


    df = pd.read_csv(csv_path)
    X = df.drop(columns=["price"])
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    n_tr, k_feat = X_train.shape
    n_te = X_test.shape[0]
    feature_names = list(X.columns)

    # Standardizace příznaků pro korektní regularizaci
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # 1. Baseline OLS
    ols = LinearRegression().fit(X_train, y_train)
    pred_ols = ols.predict(X_test)
    r2_ols = r2_score(y_test, pred_ols)
    adj_r2_ols = 1.0 - ((1.0 - r2_ols) * (n_te - 1) / (n_te - k_feat - 1))
    mae_ols = mean_absolute_error(y_test, pred_ols)
    mse_ols = mean_squared_error(y_test, pred_ols)
    rmse_ols = np.sqrt(mse_ols)

    ols_metrics = {
        "Model": "OLS (Bez regularizace)",
        "Alpha": 0.0,
        "R2": r2_ols,
        "Adj_R2": adj_r2_ols,
        "MAE": mae_ols,
        "MSE": mse_ols,
        "RMSE": rmse_ols,
    }

    # 2. Lasso Grid
    lasso_alphas = [0.1, 1.0, 10.0, 100.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0]
    lasso_models = {}
    lasso_metrics_list = []

    for a in lasso_alphas:
        l = Lasso(alpha=a, random_state=42, max_iter=2000, tol=0.01).fit(X_train_sc, y_train)
        pred = l.predict(X_test_sc)
        r2 = r2_score(y_test, pred)
        adj_r2 = 1.0 - ((1.0 - r2) * (n_te - 1) / (n_te - k_feat - 1))
        mae = mean_absolute_error(y_test, pred)
        mse = mean_squared_error(y_test, pred)
        rmse = np.sqrt(mse)

        zeroed = [f for f, w in zip(feature_names, l.coef_) if abs(w) < 1e-5]
        active = [f for f, w in zip(feature_names, l.coef_) if abs(w) >= 1e-5]

        m_dict = {
            "Model": f"Lasso (α={a})",
            "Alpha": a,
            "R2": r2,
            "Adj_R2": adj_r2,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "Zeroed_Count": len(zeroed),
            "Zeroed_Features": ", ".join(zeroed) if zeroed else "Žádné",
            "Active_Count": len(active),
        }
        lasso_metrics_list.append(m_dict)
        lasso_models[a] = {"coefs": l.coef_, "zeroed": zeroed, "metrics": m_dict}

    # 3. Ridge Grid
    ridge_alphas = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0]
    ridge_models = {}
    ridge_metrics_list = []

    for a in ridge_alphas:
        r = Ridge(alpha=a, random_state=42).fit(X_train_sc, y_train)
        pred = r.predict(X_test_sc)
        r2 = r2_score(y_test, pred)
        adj_r2 = 1.0 - ((1.0 - r2) * (n_te - 1) / (n_te - k_feat - 1))
        mae = mean_absolute_error(y_test, pred)
        mse = mean_squared_error(y_test, pred)
        rmse = np.sqrt(mse)

        m_dict = {
            "Model": f"Ridge (α={a})",
            "Alpha": a,
            "R2": r2,
            "Adj_R2": adj_r2,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "Zeroed_Count": 0,
            "Zeroed_Features": "Žádné (L2 nenuluje)",
            "Active_Count": k_feat,
        }
        ridge_metrics_list.append(m_dict)
        ridge_models[a] = {"coefs": r.coef_, "zeroed": [], "metrics": m_dict}

    return (
        feature_names,
        ols_metrics,
        lasso_models,
        pd.DataFrame(lasso_metrics_list),
        ridge_models,
        pd.DataFrame(ridge_metrics_list),
        n_tr,
        n_te,
        k_feat,
    )


feature_names, ols_metrics, lasso_models, lasso_df, ridge_models, ridge_df, n_tr, n_te, k_feat = load_and_run_regularization()

# =============================================================================
# INTERAKTIVNÍ OVLÁDACÍ PANEL
# =============================================================================
st.markdown("### 🎛️ Interaktivní konfigurátor regularizace")

col_ctrl1, col_ctrl2 = st.columns([1, 1])

with col_ctrl1:
    model_type = st.radio(
        "Vyberte typ regularizace:",
        ["Lasso (L1 Regularizace – výběr příznaků)", "Ridge (L2 Regularizace – tlumení vah)"],
        index=0,
    )

is_lasso = "Lasso" in model_type
active_models = lasso_models if is_lasso else ridge_models
active_df = lasso_df if is_lasso else ridge_df
available_alphas = list(active_models.keys())

with col_ctrl2:
    selected_alpha = st.select_slider(
        "Zvolte sílu regularizace (hyperparametr α):",
        options=available_alphas,
        value=500.0 if is_lasso else 10.0,
    )

current_model = active_models[selected_alpha]
cur_m = current_model["metrics"]
cur_coefs = current_model["coefs"]
cur_zeroed = current_model["zeroed"]

# =============================================================================
# KPI KARTY S DELTOU OPROTI OLS
# =============================================================================
delta_r2 = cur_m["R2"] - ols_metrics["R2"]
delta_mae = cur_m["MAE"] - ols_metrics["MAE"]
delta_rmse = cur_m["RMSE"] - ols_metrics["RMSE"]

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Koeficient determinace R²",
    f"{cur_m['R2']:.5f}",
    f"{delta_r2:+.5f} vs OLS",
    delta_color="normal",
)
c2.metric(
    "Adjusted R²",
    f"{cur_m['Adj_R2']:.5f}",
    f"{cur_m['Adj_R2'] - ols_metrics['Adj_R2']:+.5f}",
)
c3.metric(
    "MAE (Střední abs. chyba)",
    f"${cur_m['MAE']:,.2f}",
    f"${delta_mae:+,.2f} vs OLS",
    delta_color="inverse",
)
c4.metric(
    "RMSE (Odmocnina MSE)",
    f"${cur_m['RMSE']:,.2f}",
    f"${delta_rmse:+,.2f} vs OLS",
    delta_color="inverse",
)

# Stav nulování příznaků
if is_lasso:
    if len(cur_zeroed) > 0:
        st.warning(
            f"**Lasso nulování vah (Sparsity):** Při $\\alpha = {selected_alpha}$ model úspěšně vynuloval **{len(cur_zeroed)}** z {k_feat} příznaků: `{cur_zeroed}`."
        )
    else:
        st.info(
            f"**Lasso:** Při $\\alpha = {selected_alpha}$ jsou všechny váhy nenulové. Zvyšte $\\alpha$ pro zahájení eliminace proměnných."
        )
else:
    st.info(
        f"**Ridge ($L_2$):** Ridge nikdy koeficienty nenuluje úplně, ale rovnoměrně tlumí jejich čtverce, čímž eliminuje nestabilitu multikolinearity."
    )

# =============================================================================
# GRAF REGRESNÍCH KOEFICIENTŮ
# =============================================================================
st.markdown(f"### 📊 Hodnoty regresních koeficientů ({'Lasso' if is_lasso else 'Ridge'}, $\\alpha = {selected_alpha}$)")

coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": cur_coefs,
    "Abs_Coef": np.abs(cur_coefs),
    "Status": ["Vynulováno (0.00)" if abs(c) < 1e-5 else "Aktivní" for c in cur_coefs]
}).sort_values(by="Abs_Coef", ascending=True)

fig_bar = px.bar(
    coef_df,
    x="Coefficient",
    y="Feature",
    orientation="h",
    color="Status",
    color_discrete_map={"Aktivní": "#1D3557", "Vynulováno (0.00)": "#E63946"},
    title=f"Standardizované regresní koeficienty pro {'Lasso' if is_lasso else 'Ridge'} (α = {selected_alpha})",
    labels={"Coefficient": "Hodnota standardizovaného koeficientu β", "Feature": "Predikční příznak"},
    height=550,
)
fig_bar.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.7)
fig_bar.update_layout(margin=dict(l=20, r=20, t=40, b=20), legend_title="Stav příznaku")
st.plotly_chart(fig_bar, width="stretch")

# =============================================================================
# KOMPLETNÍ SROVNÁVACÍ TABULKA NAPŘÍČ ALPHA
# =============================================================================
st.markdown("### 📋 Souhrnné vyhodnocení všech testovaných hodnot $\\alpha$")

tab1, tab2, tab3 = st.tabs(["Lasso (L1) Grid", "Ridge (L2) Grid", "Benchmark: OLS vs. Nejlepší modely"])

with tab1:
    st.dataframe(
        lasso_df[[
            "Model", "Alpha", "R2", "Adj_R2", "MAE", "RMSE", "Zeroed_Count", "Zeroed_Features"
        ]].style.format({
            "R2": "{:.5f}",
            "Adj_R2": "{:.5f}",
            "MAE": "${:,.2f}",
            "RMSE": "${:,.2f}",
        }),
        width="stretch",
    )

with tab2:
    st.dataframe(
        ridge_df[[
            "Model", "Alpha", "R2", "Adj_R2", "MAE", "RMSE"
        ]].style.format({
            "R2": "{:.5f}",
            "Adj_R2": "{:.5f}",
            "MAE": "${:,.2f}",
            "RMSE": "${:,.2f}",
        }),
        width="stretch",
    )

with tab3:
    best_l = lasso_df.loc[lasso_df["R2"].idxmax()]
    best_r = ridge_df.loc[ridge_df["R2"].idxmax()]

    summary_compare = pd.DataFrame([
        {
            "Model": "OLS (Základní model bez regularizace)",
            "Alpha": 0.0,
            "R2": ols_metrics["R2"],
            "Adj_R2": ols_metrics["Adj_R2"],
            "MAE": ols_metrics["MAE"],
            "RMSE": ols_metrics["RMSE"],
            "Počet zachovaných příznaků": f"{k_feat} / {k_feat} (100 %)",
        },
        {
            "Model": f"Nejlepší Lasso (α = {best_l['Alpha']})",
            "Alpha": best_l["Alpha"],
            "R2": best_l["R2"],
            "Adj_R2": best_l["Adj_R2"],
            "MAE": best_l["MAE"],
            "RMSE": best_l["RMSE"],
            "Počet zachovaných příznaků": f"{k_feat - best_l['Zeroed_Count']} / {k_feat} ({((k_feat - best_l['Zeroed_Count'])/k_feat)*100:.0f} %)",
        },
        {
            "Model": f"Nejlepší Ridge (α = {best_r['Alpha']})",
            "Alpha": best_r["Alpha"],
            "R2": best_r["R2"],
            "Adj_R2": best_r["Adj_R2"],
            "MAE": best_r["MAE"],
            "RMSE": best_r["RMSE"],
            "Počet zachovaných příznaků": f"{k_feat} / {k_feat} (100 %)",
        },
    ])

    st.dataframe(
        summary_compare.style.format({
            "R2": "{:.5f}",
            "Adj_R2": "{:.5f}",
            "MAE": "${:,.2f}",
            "RMSE": "${:,.2f}",
        }),
        width="stretch",
    )

# =============================================================================
# ODPOVĚDI NA OTÁZKY ZE ZADÁNÍ
# =============================================================================
st.markdown("---")
st.markdown("### 💡 Odpovědi na otázky ze zadání cvičení")

st.markdown("""
1. **Které proměnné model Lasso vynuloval?**
   - Při malých hodnotách $\\alpha \\le 100$ jsou zachovány všechny proměnné.
   - Při $\\alpha = 2500$ model jako první eliminuje `sqft_lot` (velikost pozemku) a `sqft_basement` (plocha suterénu).
   - Při $\\alpha = 5000$ nuluje navíc `sqft_lot15` (velikost pozemku sousedů).
   - Při $\\alpha = 10000$ nuluje celkem 6 proměnných: `['bedrooms', 'sqft_lot', 'floors', 'sqft_basement', 'zipcode', 'sqft_lot15']`.
2. **Pro který regularizační koeficient $\\alpha$ je model nejlepší?**
   - **Pro Lasso:** Nejvyšší $R^2 = 0.71188$ a nejnižší RMSE = 175 296 USD dosahuje model při **$\\alpha = 500.0$** (zlepšení MAE o více než 219 USD oproti OLS).  
     *Zajímavost:* Pokud sledujeme čistě MAE, při $\\alpha = 5000.0$ klesá MAE až na **116 306 USD** (úspora 936 USD na každém domě!), a to i přesto, že model vyhodil 3 zbytečné proměnné!
   - **Pro Ridge:** Nejvyšší $R^2 = 0.71180$ dosahuje model při **$\\alpha = 10.0$** (MAE = 117 225 USD).
3. **Jak si vedou regularizované modely v porovnání s OLS bez regularizace?**
   - Na tomto rozsáhlém datasetu (n = 21 553, k = 18) OLS netrpí extrémním přetrénováním, protože počet vzorků mnohonásobně převyšuje počet parametrů.
   - Regularizace přesto mírně zlepšuje generalizační chybu na testovacích datech a Lasso navíc poskytuje **vysokou interpretovatelnost a redukci dimenzionality** (řídký model).
""")
