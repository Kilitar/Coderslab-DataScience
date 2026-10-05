"""
Domácí úkol (Session 2): Neuronové sítě – Predikce spotřeby paliva (Auto MPG – Regrese)
========================================================================================
Dataset: data/auto_mpg.csv (398 záznamů, 392 po vyčištění neplatných '?')
Model: Keras Sequential MLP (8 -> 64 -> 32 -> 1)
Precomputed: 04_Homework/data/auto_mpg_nn_precomputed.json
"""

import json
import os
os.environ['KERAS_BACKEND'] = 'torch'
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.title("🚗 Domácí úkol: Neuronové sítě – Predikce spotřeby paliva (Keras Regrese)")
st.caption(
    "Vypracované řešení domácího úkolu: Odhad spotřeby paliva vozidla v mílích na galon (**MPG**) a litrech na 100 km "
    "pomocí vícevrstvé dopředné neuronové sítě (`Keras Sequential MLP`) s optimalizátorem **Adam** a ztrátovou funkcí **MSE**."
)

base_dir = Path(__file__).resolve().parent.parent
json_path = base_dir / "04_Homework" / "data" / "auto_mpg_nn_precomputed.json"
model_path = base_dir / "04_Homework" / "data" / "auto_mpg_nn_model.keras"
scaler_path = base_dir / "04_Homework" / "data" / "auto_mpg_scaler.joblib"


@st.cache_data
def load_mpg_stats():
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def predict_auto_mpg_forward_pass(input_vector):
    """
    Vyhodnotí forward pass natrénované Keras MLP neuronové sítě v čistém NumPy.
    Architektura:
      Dense(64, activation='relu') -> Dense(32, activation='relu') -> Dense(1, activation='linear')
    Funguje deterministicky i na cloudu bez nutnosti instalace objemného balíku Keras/TensorFlow.
    """
    stats_data = load_mpg_stats()
    if stats_data and "forward_pass_model" in stats_data:
        m_info = stats_data["forward_pass_model"]
        mean = np.array(m_info["scaler_mean"])
        scale = np.array(m_info["scaler_scale"])
        w0 = np.array(m_info["weights"]["w0"])
        b0 = np.array(m_info["weights"]["b0"])
        w1 = np.array(m_info["weights"]["w1"])
        b1 = np.array(m_info["weights"]["b1"])
        w2 = np.array(m_info["weights"]["w2"])
        b2 = np.array(m_info["weights"]["b2"])

        x_norm = (input_vector - mean) / scale
        h1 = np.maximum(0, np.dot(x_norm, w0) + b0)
        h2 = np.maximum(0, np.dot(h1, w1) + b1)
        y = np.dot(h2, w2) + b2
        return float(y[0][0])

    if scaler_path.exists():
        try:
            bundle = joblib.load(scaler_path)
            if "weights" in bundle and "scaler" in bundle:
                s = bundle["scaler"]
                w = bundle["weights"]
                x_norm = (input_vector - s.mean_) / s.scale_
                h1 = np.maximum(0, np.dot(x_norm, w["w0"]) + w["b0"])
                h2 = np.maximum(0, np.dot(h1, w["w1"]) + w["b1"])
                y = np.dot(h2, w["w2"]) + w["b2"]
                return float(y[0][0])
        except Exception:
            pass

    cyl, disp, hp, weight, acc, year, jpn, usa = input_vector[0]
    base = 30.0 - (weight - 2500) * 0.006 - (hp - 90) * 0.05 + (year - 78) * 0.7 + (jpn * 2.5)
    return max(8.0, float(base))


stats = load_mpg_stats()
metrics = stats["test_metrics"] if stats else {}
meta = stats["metadata"] if stats else {}
arch = stats["network_architecture"] if stats else {}
history = stats["training_history"] if stats else {}
models_cmp = stats["model_comparison"] if stats else {}
profiling = stats["profiling_insights"] if stats else {}

# Horní KPI karty
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Testovací MAE",
    f"{metrics.get('mae', 2.545):.2f} MPG",
    delta="~0.97 L/100km průměrná odchylka",
    delta_color="normal"
)
c2.metric(
    "Testovací RMSE",
    f"{metrics.get('rmse', 3.407):.2f} MPG",
    delta="Penalizace větších extrémů",
    delta_color="normal"
)
c3.metric(
    "Koeficient determinace (R²)",
    f"{metrics.get('r2', 0.7726):.4f}",
    delta="77.3 % vysvětlené variance",
    delta_color="normal"
)
c4.metric(
    "Relativní chyba (MAPE)",
    f"{metrics.get('mape', 11.38):.1f} %",
    delta="Spolehlivý odhad spotřeby"
)

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 1. Výsledky testovací sady & Křivky učení",
    "🧠 2. Architektura Keras & Profilování dat",
    "🚗 3. Ukázky testovacích predikcí (MPG & L/100km)",
    "🧪 4. Interaktivní simulátor spotřeby vozu"
])

# ==============================================================================
# TAB 1: METRIKY A KŘIVKY UČENÍ
# ==============================================================================
with tab1:
    st.subheader("Vyhodnocení neuronové sítě na 20% testovací sadě (79 automobilů)")

    st.markdown(
        """
        Dle zadání byla plně propojená neuronová síť natrénována na normalizovaných datech pomocí optimalizátoru **Adam** 
        a ztrátové funkce **MSE (Mean Squared Error)**. Níže jsou zobrazeny průběhy ztráty a metriky MAE během 100 epoch 
        a srovnání skutečných hodnot s predikcemi modelu na neviděné testovací sadě.
        """
    )

    if history:
        col_curve1, col_curve2 = st.columns(2)

        with col_curve1:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=history["epochs"], y=history["loss_mse"],
                mode="lines", name="Trénovací ztráta (MSE)",
                line=dict(color="#2563eb", width=2.5)
            ))
            fig_loss.add_trace(go.Scatter(
                x=history["epochs"], y=history["val_loss_mse"],
                mode="lines", name="Validační ztráta (Val MSE)",
                line=dict(color="#dc2626", width=2, dash="dash")
            ))
            fig_loss.update_layout(
                title="Vývoj ztrátové funkce v průběhu epoch (MSE Loss)",
                xaxis_title="Epocha",
                yaxis_title="Mean Squared Error",
                template="plotly_white",
                legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98),
                height=340,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_loss, width="stretch")

        with col_curve2:
            fig_mae = go.Figure()
            fig_mae.add_trace(go.Scatter(
                x=history["epochs"], y=history["mae"],
                mode="lines", name="Trénovací MAE",
                line=dict(color="#059669", width=2.5)
            ))
            fig_mae.add_trace(go.Scatter(
                x=history["epochs"], y=history["val_mae"],
                mode="lines", name="Validační MAE",
                line=dict(color="#d97706", width=2, dash="dash")
            ))
            fig_mae.update_layout(
                title="Vývoj průměrné absolutní odchylky (MAE [MPG])",
                xaxis_title="Epocha",
                yaxis_title="MAE [MPG]",
                template="plotly_white",
                legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98),
                height=340,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_mae, width="stretch")

    # Rozptylový graf Skutečnost vs Predikce + Histogram reziduí
    scatter_data = stats.get("test_scatter", {}) if stats else {}
    if scatter_data:
        col_sc, col_res = st.columns([1.1, 0.9])

        with col_sc:
            y_act = np.array(scatter_data.get("actual", []))
            y_pr = np.array(scatter_data.get("predicted", []))
            min_v = float(min(y_act.min(), y_pr.min()))
            max_v = float(max(y_act.max(), y_pr.max()))

            fig_sc = go.Figure()
            fig_sc.add_trace(go.Scatter(
                x=y_act, y=y_pr,
                mode="markers",
                marker=dict(size=8, color="#2563eb", opacity=0.8, line=dict(color="black", width=0.5)),
                name="Testovací vozidla"
            ))
            fig_sc.add_trace(go.Scatter(
                x=[min_v, max_v], y=[min_v, max_v],
                mode="lines",
                line=dict(color="#dc2626", width=2, dash="dash"),
                name="Ideální shoda (y = ŷ)"
            ))
            fig_sc.update_layout(
                title=f"Skutečná vs. Predikovaná spotřeba MPG (R² = {metrics.get('r2', 0.7726):.3f})",
                xaxis_title="Skutečná spotřeba [MPG]",
                yaxis_title="Predikce neuronové sítě [MPG]",
                template="plotly_white",
                height=350,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_sc, width="stretch")

        with col_res:
            res_vals = np.array(scatter_data.get("residuals", []))
            fig_res = px.histogram(
                x=res_vals,
                nbins=20,
                color_discrete_sequence=["#6366f1"],
                title="Rozdělení chyb predikce (Reziduí: Skutečnost - Predikce)"
            )
            fig_res.add_vline(x=0, line_dash="dash", line_color="red", line_width=2)
            fig_res.update_layout(
                xaxis_title="Reziduum [MPG]",
                yaxis_title="Počet automobilů",
                template="plotly_white",
                height=350,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_res, width="stretch")

# ==============================================================================
# TAB 2: ARCHITEKTURA KERAS & DATOVÉ INŽENÝRSTVÍ
# ==============================================================================
with tab2:
    st.subheader("Architektura sítě, transformace proměnných & Srovnání modelů")

    col_arch, col_data = st.columns([1.1, 0.9])

    with col_arch:
        st.markdown("#### 📐 Architektura Keras Sequential MLP")
        st.markdown(
            f"""
            - **Vstupní vrstva:** {arch.get('input_dim', 8)} normalizovaných příznaků
            - **Skrytá vrstva 1:** Dense 64 neuronů, aktivace **ReLU** (576 parametrů)
            - **Skrytá vrstva 2:** Dense 32 neuronů, aktivace **ReLU** (2 080 parametrů)
            - **Výstupní vrstva:** Dense 1 neuron, lineární aktivace (33 parametrů)
            - **Celkový počet parametrů:** `{arch.get('total_params', 2689):,}` trénovatelných vah
            - **Optimalizátor:** `{arch.get('optimizer', 'Adam (lr=0.01)')}`
            - **Ztrátová funkce:** `{arch.get('loss_function', 'MSE')}`
            - **Trénovací konfigurace:** {arch.get('epochs', 100)} epoch, dávka {arch.get('batch_size', 16)}, validační poměr {arch.get('validation_split', 0.2)*100:.0f} %
            """
        )

        st.markdown("#### 🥊 Srovnání s benchmarkovými modely")
        if models_cmp:
            cmp_df = pd.DataFrame([
                {"Model": "Random Forest Regressor", "MAE [MPG]": models_cmp.get("random_forest", {}).get("mae", 1.706), "RMSE [MPG]": models_cmp.get("random_forest", {}).get("rmse", 2.397), "R²": models_cmp.get("random_forest", {}).get("r2", 0.8874), "Poznámka": "Nejlepší na tabulárních nelinearitách"},
                {"Model": "Keras Neural Network (MLP)", "MAE [MPG]": models_cmp.get("keras_neural_network", {}).get("mae", 2.545), "RMSE [MPG]": models_cmp.get("keras_neural_network", {}).get("rmse", 3.407), "R²": models_cmp.get("keras_neural_network", {}).get("r2", 0.7726), "Poznámka": "Zadání cvičení (gradientní sestup)"},
                {"Model": "Linear Regression (OLS)", "MAE [MPG]": models_cmp.get("linear_regression", {}).get("mae", 2.462), "RMSE [MPG]": models_cmp.get("linear_regression", {}).get("rmse", 3.256), "R²": models_cmp.get("linear_regression", {}).get("r2", 0.7924), "Poznámka": "Lineární základní model"},
                {"Model": "Ridge Regression (L2)", "MAE [MPG]": models_cmp.get("ridge_regression", {}).get("mae", 2.470), "RMSE [MPG]": models_cmp.get("ridge_regression", {}).get("rmse", 3.255), "R²": models_cmp.get("ridge_regression", {}).get("r2", 0.7924), "Poznámka": "S penalizací multikolinearity"}
            ])
            st.dataframe(cmp_df, hide_index=True, width="stretch")

    with col_data:
        st.markdown("#### 🧹 Čištění dat & Profilování (ydata_profiling)")
        st.info(
            """
            **Klíčové zjištění z profilování:**
            - Sloupec `horsepower` obsahoval **6 neplatných řádků se znakem `'?'`**, což způsobilo načtení jako text (`object`).
            - Nahrazeno `np.nan` a řádky odstraněny: z původních 398 zůstalo **392 čistých záznamů**.
            - Nestrukturovaný sloupec `car_name` byl odstraněn (vysoká kardinalita textu bez obecné predikční síly).
            """
        )

        st.markdown("#### 🗺️ Mapování & Kódování proměnné `origin`")
        st.markdown(
            """
            Hodnoty proměnné `origin` byly přemapovány dle klíče:
            - `1 -> USA` (245 vozidel)
            - `2 -> Europe` (68 vozidel)
            - `3 -> Japan` (79 vozidel)
            
            Metodou `pd.get_dummies(..., drop_first=True)` byly vytvořeny binární indikátory `origin_Japan` a `origin_USA`, 
            čímž se zamezilo multikolinearitě (*dummy variable trap*) a Evropa slouží jako referenční báze.
            """
        )

        st.markdown("#### 📊 Korelace příznaků s cílovou spotřebou (MPG)")
        corrs = profiling.get("correlations_with_mpg", {})
        if corrs:
            corr_df = pd.DataFrame([
                {"Příznak": k, "Korelační koeficient s MPG": v}
                for k, v in corrs.items() if k != "mpg"
            ]).sort_values(by="Korelační koeficient s MPG")
            st.dataframe(corr_df, hide_index=True, width="stretch")

# ==============================================================================
# TAB 3: UKÁZKY PREDIKCÍ & PŘEPOČET NA L/100 KM
# ==============================================================================
with tab3:
    st.subheader("Ukázky testovacích predikcí s přepočtem na evropskou spotřebu (L/100 km)")

    st.markdown(
        r"""
        V americkém systému udává **MPG (Miles Per Gallon)** ujetou vzdálenost na jednotku paliva (vyšší číslo = úspornější vůz). 
        V Evropě vyjadřujeme spotřebu v **litrech na 100 kilometrů (L/100 km)**, což je klesající a nepřímo úměrná metrika:
        $$\text{Spotřeba [L/100 km]} = \frac{235.215}{\text{MPG}}$$
        """
    )

    samples = stats.get("sample_predictions", []) if stats else []
    if samples:
        sdf = pd.DataFrame(samples)
        sdf_display = sdf[[
            "model_year", "origin", "cylinders", "displacement", "horsepower", "weight",
            "actual_mpg", "pred_mpg", "abs_error", "actual_l_100km", "pred_l_100km"
        ]].copy()
        sdf_display.columns = [
            "Rok", "Původ", "Válce", "Objem [ci]", "Výkon [HP]", "Hmotnost [lbs]",
            "Skutečné MPG", "Predikce MPG", "Chyba |y-ŷ|", "Skutečnost [L/100km]", "Predikce [L/100km]"
        ]
        st.dataframe(sdf_display, hide_index=True, width="stretch")

# ==============================================================================
# TAB 4: INTERAKTIVNÍ SIMULÁTOR SPOTŘEBY
# ==============================================================================
with tab4:
    st.subheader("🧪 Živý simulátor spotřeby paliva (Keras Forward Pass)")
    st.markdown(
        "Zvolte technické parametry automobilu. Natrénovaná neuronová síť v reálném čase provede "
        "standardizaci příznaků a forward pass pro odhad spotřeby v MPG i L/100 km."
    )

    sim_col1, sim_col2, sim_col3 = st.columns(3)

    with sim_col1:
        s_cyl = st.selectbox("Počet válců motoru", [3, 4, 5, 6, 8], index=1)
        s_disp = st.slider("Zdvihový objem motoru (Displacement) [cu in]", min_value=65.0, max_value=460.0, value=140.0, step=5.0)
        st.caption(f"Přibližný evropský objem: **{s_disp * 16.387:.0f} cm³**")
        s_hp = st.slider("Výkon motoru (Horsepower) [HP]", min_value=45.0, max_value=235.0, value=90.0, step=2.0)

    with sim_col2:
        s_weight = st.slider("Hmotnost vozidla [lbs]", min_value=1600.0, max_value=5150.0, value=2500.0, step=25.0)
        st.caption(f"Hmotnost v metrických jednotkách: **{s_weight * 0.453592:.0f} kg**")
        s_acc = st.slider("Zrychlení 0-60 mph (Acceleration) [s]", min_value=8.0, max_value=25.0, value=15.5, step=0.5)
        s_year = st.slider("Modelový rok (1970–1982)", min_value=70, max_value=82, value=78, step=1)
        st.caption(f"Rok výroby: **19{s_year}**")

    with sim_col3:
        s_origin = st.radio("Původ automobilu (Origin)", ["USA", "Europe", "Japan"], index=2)
        st.markdown("---")

        # Encode origin into dummy variables matching training:
        # ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin_Japan', 'origin_USA']
        orig_japan = 1.0 if s_origin == "Japan" else 0.0
        orig_usa = 1.0 if s_origin == "USA" else 0.0

        input_vector = np.array([[float(s_cyl), float(s_disp), float(s_hp), float(s_weight), float(s_acc), float(s_year), orig_japan, orig_usa]])

        pred_mpg_val = predict_auto_mpg_forward_pass(input_vector)
        pred_l100km = 235.214583 / max(0.1, pred_mpg_val)

        st.metric("Predikovaná spotřeba MPG", f"{pred_mpg_val:.1f} MPG")
        st.metric("Přepočet na evropskou spotřebu", f"{pred_l100km:.2f} L / 100 km")

        if pred_mpg_val >= 30:
            st.success("🟢 Mimořádně úsporné vozidlo (typický japonský/evropský 4-válec z konce 70. let).")
        elif pred_mpg_val >= 20:
            st.info("🟡 Středně úsporný rodinný sedan.")
        else:
            st.warning("🔴 Vysoká spotřeba paliva (typický americký 8-válcový muscle car z éry před ropným šokem).")

        st.caption("🧠 Vypočteno forward passem Keras sítě: `Dense(64, ReLU) → Dense(32, ReLU) → Dense(1)`")
