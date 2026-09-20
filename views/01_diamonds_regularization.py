import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

st.title("💎 Cvičení 6: Regularizace modelu diamantů (Lasso & Ridge)")
st.caption("Aplikace regularizace L1 a L2 pro oceňování diamantů, eliminace prostorové kolinearity (x, y, z) a srovnání s OLS.")

# =============================================================================
# CACHE VÝPOČTŮ (OKAMŽITÉ NAČTENÍ Z PŘEDPOČÍTANÝCH DAT)
# =============================================================================
@st.cache_data
def load_and_run_diamonds_regularization():
    base_dir = Path(__file__).resolve().parent.parent
    precomputed_path = base_dir / "01_Regression" / "data" / "diamonds_regularization_precomputed.json"

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

    csv_path = base_dir / "01_Regression" / "data" / "diamonds_preprocessed.csv"

    df = pd.read_csv(csv_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    X = df.drop(columns=["price"])
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    n_tr, k_feat = X_train.shape
    n_te = X_test.shape[0]
    feature_names = list(X.columns)

    # Standardizace
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # 1. OLS Baseline
    ols = LinearRegression().fit(X_train_sc, y_train)
    pred_ols = ols.predict(X_test_sc)
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
    lasso_alphas = [0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    lasso_models = {}
    lasso_metrics_list = []

    for a in lasso_alphas:
        l = Lasso(alpha=a, random_state=42, max_iter=3000, tol=0.01).fit(X_train_sc, y_train)
        pred = l.predict(X_test_sc)
        r2 = r2_score(y_test, pred)
        adj_r2 = 1.0 - ((1.0 - r2) * (n_te - 1) / (n_te - k_feat - 1))
        mae = mean_absolute_error(y_test, pred)
        mse = mean_squared_error(y_test, pred)
        rmse = np.sqrt(mse)

        zeroed = [f for f, w in zip(feature_names, l.coef_) if abs(w) < 1e-4]
        active = [f for f, w in zip(feature_names, l.coef_) if abs(w) >= 1e-4]

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


(
    feature_names,
    ols_metrics,
    lasso_models,
    lasso_df,
    ridge_models,
    ridge_df,
    n_tr,
    n_te,
    k_feat,
) = load_and_run_diamonds_regularization()

# =============================================================================
# OVLÁDACÍ PANEL
# =============================================================================
st.markdown("### 🎛️ Interaktivní konfigurátor regularizace diamantů")

col1, col2 = st.columns(2)

with col1:
    model_type = st.radio(
        "Vyberte regularizační algoritmus:",
        ["Lasso (L1 – Selekce příznaků a řídkost)", "Ridge (L2 – Tlumení multikolinearity)"],
        index=0,
    )

is_lasso = "Lasso" in model_type
active_models = lasso_models if is_lasso else ridge_models
active_df = lasso_df if is_lasso else ridge_df
available_alphas = list(active_models.keys())

with col2:
    selected_alpha = st.select_slider(
        "Nastavte sílu regularizace (hyperparametr α):",
        options=available_alphas,
        value=50.0 if is_lasso else 10.0,
    )

current_model = active_models[selected_alpha]
cur_m = current_model["metrics"]
cur_coefs = current_model["coefs"]
cur_zeroed = current_model["zeroed"]

# =============================================================================
# KPI KARTY S DELTOU
# =============================================================================
delta_r2 = cur_m["R2"] - ols_metrics["R2"]
delta_mae = cur_m["MAE"] - ols_metrics["MAE"]
delta_rmse = cur_m["RMSE"] - ols_metrics["RMSE"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Koeficient determinace R²", f"{cur_m['R2']:.5f}", f"{delta_r2:+.5f} vs OLS", delta_color="normal")
c2.metric("Adjusted R²", f"{cur_m['Adj_R2']:.5f}", f"{cur_m['Adj_R2'] - ols_metrics['Adj_R2']:+.5f}")
c3.metric("MAE (Střední abs. chyba)", f"${cur_m['MAE']:,.2f}", f"${delta_mae:+,.2f} vs OLS", delta_color="inverse")
c4.metric("RMSE (Odmocnina MSE)", f"${cur_m['RMSE']:,.2f}", f"${delta_rmse:+,.2f} vs OLS", delta_color="inverse")

if is_lasso:
    if len(cur_zeroed) > 0:
        st.warning(
            f"**Lasso selekce příznaků:** Při $\\alpha = {selected_alpha}$ model vynuloval **{len(cur_zeroed)}** z {k_feat} příznaků: `{cur_zeroed}`."
        )
    else:
        st.info(f"Při $\\alpha = {selected_alpha}$ jsou všechny váhy aktivní.")
else:
    st.info(
        f"**Ridge ($L_2$):** Všechny koeficienty zůstávají aktivní, ale dochází ke stabilizaci fyzikální multikolinearity mezi $x, y, z$ a karátem."
    )

# =============================================================================
# BAR CHART KOEFICIENTŮ
# =============================================================================
st.markdown(f"### 📊 Regresní váhy modelu ({'Lasso' if is_lasso else 'Ridge'}, $\\alpha = {selected_alpha}$)")

coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": cur_coefs,
    "Abs_Coef": np.abs(cur_coefs),
    "Status": ["Vynulováno (0.00)" if abs(c) < 1e-4 else "Aktivní" for c in cur_coefs]
}).sort_values(by="Abs_Coef", ascending=True)

fig_bar = px.bar(
    coef_df,
    x="Coefficient",
    y="Feature",
    orientation="h",
    color="Status",
    color_discrete_map={"Aktivní": "#1D3557", "Vynulováno (0.00)": "#E63946"},
    title=f"Standardizované koeficienty pro {'Lasso' if is_lasso else 'Ridge'} (α = {selected_alpha})",
    labels={"Coefficient": "Standardizovaný koeficient β", "Feature": "Vlastnost diamantu"},
    height=450,
)
fig_bar.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.7)
st.plotly_chart(fig_bar, width="stretch")

# =============================================================================
# TABULKY VÝSLEDKŮ
# =============================================================================
st.markdown("### 📋 Výsledky všech testovaných hodnot $\\alpha$")

t1, t2, t3 = st.tabs(["Lasso (L1) Grid", "Ridge (L2) Grid", "Benchmark: OLS vs. Nejlepší modely"])

with t1:
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

with t2:
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

with t3:
    best_l = lasso_df.loc[lasso_df["R2"].idxmax()]
    best_r = ridge_df.loc[ridge_df["R2"].idxmax()]

    benchmark_df = pd.DataFrame([
        {
            "Model": "OLS (Bez regularizace)",
            "Alpha": 0.0,
            "R2": ols_metrics["R2"],
            "Adj_R2": ols_metrics["Adj_R2"],
            "MAE": ols_metrics["MAE"],
            "RMSE": ols_metrics["RMSE"],
            "Aktivní příznaky": f"{k_feat} / {k_feat} (100 %)",
        },
        {
            "Model": f"Nejlepší Lasso (α = {best_l['Alpha']})",
            "Alpha": best_l["Alpha"],
            "R2": best_l["R2"],
            "Adj_R2": best_l["Adj_R2"],
            "MAE": best_l["MAE"],
            "RMSE": best_l["RMSE"],
            "Aktivní příznaky": f"{k_feat - best_l['Zeroed_Count']} / {k_feat} (100 %)",
        },
        {
            "Model": "Lasso s automatickou 4C selekcí (α = 50.0)",
            "Alpha": 50.0,
            "R2": lasso_models[50.0]["metrics"]["R2"],
            "Adj_R2": lasso_models[50.0]["metrics"]["Adj_R2"],
            "MAE": lasso_models[50.0]["metrics"]["MAE"],
            "RMSE": lasso_models[50.0]["metrics"]["RMSE"],
            "Aktivní příznaky": "4 / 9 (Pouze 4C: carat, cut, color, clarity)",
        },
        {
            "Model": f"Nejlepší Ridge (α = {best_r['Alpha']})",
            "Alpha": best_r["Alpha"],
            "R2": best_r["R2"],
            "Adj_R2": best_r["Adj_R2"],
            "MAE": best_r["MAE"],
            "RMSE": best_r["RMSE"],
            "Aktivní příznaky": f"{k_feat} / {k_feat} (100 %)",
        },
    ])

    st.dataframe(
        benchmark_df.style.format({
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
1. **Které koeficienty model Lasso vynuloval?**
   - Při $\\alpha = 10.0$ jako první nuluje `y` (šířka diamantu, která je téměř na 100 % shodná s $x$).
   - Při $\\alpha = 50.0$ dochází k zásadnímu jevu: model nuluje **5 příznaků**: `['depth', 'table', 'x', 'y', 'z']`.  
     Všechny prostorové rozměry a proporce jsou vyřazeny a modelu zůstávají **výhradně gemologické 4C** (`carat`, `cut`, `color`, `clarity`), přičemž $R^2 = 0.90412$ (ztráta vysvětleného rozptylu je minimální, pod 0.5 %!).
   - Při $\\alpha = 200.0$ nuluje navíc `cut` a při $\\alpha = 500.0$ i `color`.
2. **Pro které $\\alpha$ je model nejlepší?**
   - **Lasso:** Nejvyšší $R^2 = 0.90949$ dosahuje při **$\\alpha = 0.1$** a nejnižší MAE = 783.83 USD při **$\\alpha = 1.0$**.
   - **Ridge:** Nejvyšší $R^2 = 0.90958$ dosahuje při **$\\alpha = 10.0$** (tlumení protiběžných extrémů mezi prostorovými rozměry a karátem).
3. **Porovnání s OLS bez regularizace:**
   - OLS dosahuje $R^2 = 0.90947$ a MAE = 784.78 USD.
   - Regularizace přináší stabilizaci modelu a Lasso dokazuje, že měření rozměrů v milimetrech ($x, y, z$) nepřináší klenotníkovi téměř žádnou přidanou hodnotu, pokud zná karát a brus.
""")
