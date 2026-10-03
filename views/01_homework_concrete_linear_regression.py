"""
Homework: Lineární regrese – Pevnost betonu (Concrete Compressive Strength)
==========================================================================
Tento modul vizualizuje výsledky cvičení 'Linear regression - exercise':
1. Trénování OLS modelu (LinearRegression) na 70 % trénovací sadě.
2. Vyhodnocení metrik (R2, RMSE, MAE, MAPE, MSE) na testovací sadě (30 %).
3. Interaktivní regresní koeficienty (Beta weights) v Plotly.
4. Interaktivní Actual vs. Predicted scatter plot a diagnostika reziduí.
5. Fyzikální interpretace a limity lineárního modelu.
6. Interaktivní simulátor pevnosti betonu (live predikce).
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


@st.cache_data
def load_concrete_preprocessed_data():
    base_dir = Path(__file__).resolve().parent.parent
    csv_path = base_dir / "01_Regression" / "data" / "concrete_data_preprocessed.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        feature_cols = [
            "cement", "slag", "flyash", "water",
            "superplasticizer", "coarseaggregate", "fineaggregate", "age"
        ]
        X = df[feature_cols]
        y = df["csMPa"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred_test = lr.predict(X_test)
        residuals_test = y_test - y_pred_test

        test_df = X_test.copy()
        test_df["actual_csMPa"] = y_test
        test_df["pred_csMPa"] = y_pred_test
        test_df["residual"] = residuals_test
        test_df["abs_error"] = np.abs(residuals_test)
        return df, lr, test_df
    return None, None, None


def load_concrete_lr_precomputed():
    base_dir = Path(__file__).resolve().parent.parent
    cache_path = base_dir / "01_Regression" / "data" / "concrete_linear_regression_precomputed.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def render_concrete_linear_regression_view():
    st.title("📈 DÚ: Lineární regrese – Pevnost betonu")
    st.markdown(
        "**Vypracování cvičení:** Konstrukce základního modelu vícerozměrné lineární regrese (`LinearRegression`), "
        "trénování na 70 % rozdělených škálovaných dat a komplexní vyhodnocení na testovací sadě (30 %)."
    )

    data = load_concrete_lr_precomputed()
    full_df, lr_model, test_results_df = load_concrete_preprocessed_data()

    if not data:
        st.error("Předpočtená data nebyla nalezena. Spusťte skript `01_Regression/11_homework_concrete_linear_regression.py`.")
        return

    meta = data["metadata"]
    params = data["model_parameters"]
    m = data["metrics"]
    test_m = m["test"]
    train_m = m["train"]
    coefs = params["coefficients"]
    intercept = params["intercept"]

    # Rychlé KPI metriky v záhlaví
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Testovací R²", f"{test_m['r2'] * 100:.2f} %", delta=f"{train_m['r2'] * 100:.2f} % Train", delta_color="off")
    with c2:
        st.metric("Testovací RMSE", f"{test_m['rmse']:.2f} MPa", delta=f"MSE: {test_m['mse']:.1f}", delta_color="inverse")
    with c3:
        st.metric("Testovací MAE", f"{test_m['mae']:.2f} MPa", delta=f"MAPE: {test_m['mape']:.1f} %", delta_color="inverse")
    with c4:
        st.metric("Max. absolutní chyba", f"{test_m['max_error']:.2f} MPa", delta="Extrémní reziduum", delta_color="inverse")

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Evaluace & Zadání",
        "⚖️ Regresní koeficienty (Plotly)",
        "📊 Skutečnost vs. Predikce (Plotly)",
        "🔍 Diagnostika reziduí (Plotly)",
        "🧠 Limity lineárního modelu",
        "🎛️ Simulátor pevnosti (Live)"
    ])

    # TAB 1: Evaluace & Zadání
    with tab1:
        st.subheader("1. Vyhodnocení efektivity modelu dle zadání")
        st.markdown(
            "Zadání požadovalo otestovat efektivitu natrénovaného regresoru na testovací sadě (30 % dat, tj. 302 vzorků). "
            "Model byl natrénován pomocí knihovny `scikit-learn` metodou nejmenších čtverců (OLS)."
        )

        col_t1, col_t2 = st.columns([1, 1])
        with col_t1:
            st.markdown("##### 🧪 Metriky na testovací sadě (Test Set, 30 %)")
            test_metrics_table = [
                {"Metrika": "Koeficient determinace (R²)", "Hodnota": f"{test_m['r2'] * 100:.2f} %", "Popis": "Podíl vysvětleného rozptylu cíle csMPa"},
                {"Metrika": "Odmocněná chyba (RMSE)", "Hodnota": f"{test_m['rmse']:.4f} MPa", "Popis": "Směrodatná odchylka chyb odhadu"},
                {"Metrika": "Průměrná absolutní chyba (MAE)", "Hodnota": f"{test_m['mae']:.4f} MPa", "Popis": "Průměrná absolutní odchylka od reality"},
                {"Metrika": "Střední kvadratická chyba (MSE)", "Hodnota": f"{test_m['mse']:.4f} MPa²", "Popis": "Kvadratická penalizace větších odchylek"},
                {"Metrika": "Relativní chyba (MAPE)", "Hodnota": f"{test_m['mape']:.2f} %", "Popis": "Průměrné procento chyby"},
                {"Metrika": "Maximální chyba (Max Error)", "Hodnota": f"{test_m['max_error']:.4f} MPa", "Popis": "Největší zaznamenaná odchylka"}
            ]
            st.dataframe(pd.DataFrame(test_metrics_table), width="stretch", hide_index=True)

        with col_t2:
            st.markdown("##### 🏋️ Trénovací sada (Train Set, 70 %) vs. Generalizace")
            train_metrics_table = [
                {"Metrika": "Koeficient determinace (R²)", "Train (70 %)": f"{train_m['r2'] * 100:.2f} %", "Test (30 %)": f"{test_m['r2'] * 100:.2f} %"},
                {"Metrika": "RMSE (MPa)", "Train (70 %)": f"{train_m['rmse']:.4f}", "Test (30 %)": f"{test_m['rmse']:.4f}"},
                {"Metrika": "MAE (MPa)", "Train (70 %)": f"{train_m['mae']:.4f}", "Test (30 %)": f"{test_m['mae']:.4f}"},
                {"Metrika": "MAPE (%)", "Train (70 %)": f"{train_m['mape']:.2f} %", "Test (30 %)": f"{test_m['mape']:.2f} %"}
            ]
            st.dataframe(pd.DataFrame(train_metrics_table), width="stretch", hide_index=True)

            st.caption(
                "Rozdíl mezi Train $R^2$ (62.2 %) a Test $R^2$ (56.1 %) ukazuje mírný pokles generalizace, "
                "ale nedochází k drastickému přeučení (overfittingu), což je pro jednoduché OLS typické."
            )

        st.code(
            r"""# Implementace zadaného postupu:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

y_pred_test = linear_reg.predict(X_test)
test_r2 = r2_score(y_test, y_pred_test)
test_rmse = root_mean_squared_error(y_test, y_pred_test)
test_mae = mean_absolute_error(y_test, y_pred_test)""",
            language="python"
        )

        st.info(
            f"**Závěr zadání:** Koeficient determinace na testovací sadě dosahuje **$R^2 = {test_m['r2']*100:.2f} \\%$**, což znamená, "
            f"že více než polovina rozptylu pevnosti betonu je vysvětlena jednoduchou lineární kombinací složek. "
            f"Průměrná absolutní chyba činí **{test_m['mae']:.2f} MPa**."
        )

    # TAB 2: Regresní koeficienty (Plně interaktivní Plotly)
    with tab2:
        st.subheader("2. Interaktivní standardizované regresní koeficienty (Beta weights)")
        st.markdown(
            "Jelikož byly všechny prediktory standardizovány (`StandardScaler`), "
            "koeficienty $\\beta_j$ mají společné měřítko a přímo vyjadřují relativní váhu každé složky. "
            "Koeficient udává změnu pevnosti betonu v MPa při zvýšení dané složky o **1 směrodatnou odchylku**."
        )

        coef_df = pd.DataFrame([
            {
                "feature": feat,
                "coef": coef,
                "description": meta["feature_descriptions"][feat],
                "direction": "Pozitivní vliv (+)" if coef > 0 else "Negativní vliv (-)",
                "color": "#2ca02c" if coef > 0 else "#d62728"
            }
            for feat, coef in sorted(coefs.items(), key=lambda x: x[1], reverse=True)
        ])

        fig_coef = px.bar(
            coef_df,
            x="coef",
            y="feature",
            orientation="h",
            color="direction",
            color_discrete_map={"Pozitivní vliv (+)": "#2ca02c", "Negativní vliv (-)": "#d62728"},
            labels={"coef": "Váha koeficientu β (MPa na 1σ)", "feature": "Vstupní složka"},
            title="Standardizované regresní koeficienty modelu LinearRegression (OLS)",
            text_auto="+.2f"
        )
        fig_coef.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_coef, width="stretch")

        col_c1, col_c2 = st.columns([1, 1])
        with col_c1:
            st.markdown("##### 📋 Tabulka parametrů modelu")
            coef_list = [
                {
                    "Proměnná": row["feature"],
                    "Popis složky": row["description"],
                    "Koeficient β (MPa/1σ)": f"{row['coef']:+.4f}",
                    "Směr vlivu": "Pozitivní 🟢" if row['coef'] > 0 else "Negativní 🔴"
                }
                for _, row in coef_df.iterrows()
            ]
            st.dataframe(pd.DataFrame(coef_list), width="stretch", hide_index=True)
            st.caption(f"Absolutní člen (Intercept $\\beta_0$): **{intercept:.4f} MPa** (odpovídá průměrné pevnosti).")

        with col_c2:
            st.markdown("##### 🔬 Fyzikální interpretace vlivu složek")
            st.markdown(
                r"""
                - **Cement ($\beta = +12.79\ \text{MPa}$):** Zdaleka nejvýznamnější složka. Každý nárůst cementu o $+1\sigma$ ($104.3\ \text{kg/m}^3$) zvýší pevnost o téměř $13\ \text{MPa}$.
                - **Struska ($\beta = +9.23\ \text{MPa}$):** Významný sekundární hydratační prvek s vysokou pojivou schopností.
                - **Stáří betonu `age` ($\beta = +7.13\ \text{MPa}$):** Čas nutný k vykrystalizování nosné matrice.
                - **Popílek `flyash` ($\beta = +6.05\ \text{MPa}$):** Pucolánová reakce zvyšující pevnost.
                - **Voda `water` ($\beta = -2.32\ \text{MPa}$):** Záporný koeficient! Každá nadbytečná voda v záměsi oslabuje kompozit (Abramsův zákon).
                """
            )

    # TAB 3: Skutečnost vs. Predikce (Plně interaktivní Plotly)
    with tab3:
        st.subheader("3. Interaktivní srovnání: Skutečnost vs. Predikce (Actual vs. Predicted)")
        st.markdown(
            "Porovnání skutečné naměřené pevnosti betonu z testovací sady proti hodnotě predikované modelem. "
            "Červená přerušovaná čára představuje ideální predikci ($y = \\hat{y}$)."
        )

        if test_results_df is not None:
            fig_pred = px.scatter(
                test_results_df,
                x="actual_csMPa",
                y="pred_csMPa",
                color="abs_error",
                color_continuous_scale="Viridis",
                labels={
                    "actual_csMPa": "Skutečná pevnost v tlaku (MPa)",
                    "pred_csMPa": "Predikovaná pevnost OLS (MPa)",
                    "abs_error": "Chyba (MPa)"
                },
                title="Actual vs. Predicted: OLS model na testovací sadě (302 vzorků)",
                hover_data={"actual_csMPa": ":.2f", "pred_csMPa": ":.2f", "abs_error": ":.2f", "residual": ":.2f"}
            )
            # Přidání ideální diagonály y = x
            min_val = min(test_results_df["actual_csMPa"].min(), test_results_df["pred_csMPa"].min())
            max_val = max(test_results_df["actual_csMPa"].max(), test_results_df["pred_csMPa"].max())
            fig_pred.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="Ideální predikce (y = ŷ)",
                line=dict(color="red", dash="dash", width=2)
            ))
            fig_pred.update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_pred, width="stretch")

        st.markdown("##### 🔍 Ukázka predikcí na konkrétních testovacích vzorcích")
        test_samples_df = pd.DataFrame(data["test_sample_predictions"])
        st.dataframe(test_samples_df, width="stretch", hide_index=True)

    # TAB 4: Diagnostika reziduí (Plně interaktivní Plotly)
    with tab4:
        st.subheader("4. Interaktivní diagnostika reziduí (Chyby predikce)")
        st.markdown(
            "Správný lineární model by měl mít rezidua ($e = y - \\hat{y}$) s nulovým průměrem, "
            "konstantním rozptylem (homoskedasticita) a přibližně normálním rozdělením."
        )

        res_info = data["residuals"]
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Průměr reziduí (Test)", f"{res_info['test_mean']:+.2f} MPa", help="Ideál je 0.0.")
        rc2.metric("Směrodatná odchylka reziduí", f"{res_info['test_std']:.2f} MPa")
        rc3.metric("Šikmost rozdělení chyb", f"{res_info['test_skewness']:+.2f}", help="Blízko 0 = symetrické normální chyby.")

        if test_results_df is not None:
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                fig_res_scatter = px.scatter(
                    test_results_df,
                    x="pred_csMPa",
                    y="residual",
                    title="Scatter reziduí proti predikované hodnotě",
                    labels={"pred_csMPa": "Predikce OLS (MPa)", "residual": "Reziduální chyba (MPa)"},
                    opacity=0.75
                )
                fig_res_scatter.add_hline(y=0, line_dash="dash", line_color="red", line_width=1.5)
                fig_res_scatter.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_res_scatter, width="stretch")

            with col_r2:
                fig_res_hist = px.histogram(
                    test_results_df,
                    x="residual",
                    nbins=30,
                    marginal="box",
                    title="Rozdělení reziduí na testovací sadě",
                    labels={"residual": "Reziduum (MPa)"},
                    color_discrete_sequence=["#636EFA"]
                )
                fig_res_hist.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_res_hist, width="stretch")

    # TAB 5: Limity lineárního modelu
    with tab5:
        st.subheader("5. Proč lineární regrese vysvětluje jen 56 % a kde naráží?")
        st.markdown(
            r"""
            Výsledek $R^2 \approx 56.1\ \%$ a $\text{RMSE} \approx 11.24\ \text{MPa}$ ukazuje, že lineární model 
            je dobrým základním odhadcem (*baseline*), ale pro přesnou betonářskou praxi nestačí.
            
            #### 1. Nelineární zrání v čase (Logaritmický charakter stáří):
            Pevnost betonu roste s časem nelineárně. Během prvních 7 až 28 dnů hydratuje většina slinku 
            a pevnost roste exponenciálně rychle. Po 28 dnech dochází k nasycení (*plató*) a další nárůst mezi 
            90 a 365 dny je minimální. Lineární přímka modeluje nárůst konstantně za den, čímž mladé betony podhodnocuje 
            a staré betony silně nadhodnocuje!
            
            #### 2. Synergie a nelineární poměr $w/c$ (Voda / Cement):
            Pevnost kompozitu není dána prostým součtem $a \cdot \text{cement} + b \cdot \text{voda}$, 
            nýbrž **podílem** $\frac{\text{voda}}{\text{cement}}$. OLS neumí bez manuálního inženýrství příznaků 
            modelovat dělení proměnných ani multiplikativní interakce.
            """
        )

    # TAB 6: Simulátor pevnosti (Live interactive)
    with tab6:
        st.subheader("6. 🎛️ Živý simulátor pevnosti betonu (Interaktivní prediktor)")
        st.markdown(
            "Nastavte složení betonové směsi v původních fyzikálních jednotkách ($\text{kg/m}^3$) "
            "a sledujte živý výpočet odhadu pevnosti v tlaku modelem lineární regrese."
        )

        sim_c1, sim_c2 = st.columns(2)
        with sim_c1:
            sim_cement = st.slider("Cement (kg/m³):", min_value=100.0, max_value=550.0, value=280.0, step=5.0)
            sim_slag = st.slider("Vysokopecní struska (kg/m³):", min_value=0.0, max_value=360.0, value=75.0, step=5.0)
            sim_flyash = st.slider("Popílek (kg/m³):", min_value=0.0, max_value=200.0, value=50.0, step=5.0)
            sim_water = st.slider("Záměsová voda (kg/m³):", min_value=120.0, max_value=250.0, value=180.0, step=2.0)
        with sim_c2:
            sim_sp = st.slider("Superplastifikátor (kg/m³):", min_value=0.0, max_value=35.0, value=6.0, step=0.5)
            sim_coarse = st.slider("Hrubé kamenivo (kg/m³):", min_value=800.0, max_value=1150.0, value=970.0, step=10.0)
            sim_fine = st.slider("Jemné kamenivo / písek (kg/m³):", min_value=590.0, max_value=1000.0, value=770.0, step=10.0)
            sim_age = st.slider("Doba zrání (dny):", min_value=1, max_value=365, value=28, step=1)

        # Standardizace pro simulátor
        means = {"cement": 281.17, "slag": 73.90, "flyash": 54.19, "water": 181.57, "superplasticizer": 6.20, "coarseaggregate": 972.92, "fineaggregate": 773.58, "age": 45.66}
        stds = {"cement": 104.51, "slag": 86.28, "flyash": 63.99, "water": 21.36, "superplasticizer": 5.97, "coarseaggregate": 77.75, "fineaggregate": 80.18, "age": 63.17}

        z_cement = (sim_cement - means["cement"]) / stds["cement"]
        z_slag = (sim_slag - means["slag"]) / stds["slag"]
        z_flyash = (sim_flyash - means["flyash"]) / stds["flyash"]
        z_water = (sim_water - means["water"]) / stds["water"]
        z_sp = (sim_sp - means["superplasticizer"]) / stds["superplasticizer"]
        z_coarse = (sim_coarse - means["coarseaggregate"]) / stds["coarseaggregate"]
        z_fine = (sim_fine - means["fineaggregate"]) / stds["fineaggregate"]
        z_age = (sim_age - means["age"]) / stds["age"]

        pred_sim = (
            intercept +
            coefs["cement"] * z_cement +
            coefs["slag"] * z_slag +
            coefs["flyash"] * z_flyash +
            coefs["water"] * z_water +
            coefs["superplasticizer"] * z_sp +
            coefs["coarseaggregate"] * z_coarse +
            coefs["fineaggregate"] * z_fine +
            coefs["age"] * z_age
        )

        wc_ratio = sim_water / (sim_cement + 1e-5)
        st.markdown("---")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric("Odhadnutá pevnost betonu (csMPa)", f"{pred_sim:.2f} MPa")
            if pred_sim < 20:
                st.warning("Třída pevnosti: C12/15 až C16/20 (Nízkopevnostní podkladní betony)")
            elif pred_sim < 40:
                st.success("Třída pevnosti: C25/30 až C30/37 (Běžný konstrukční beton)")
            else:
                st.info("Třída pevnosti: C40/50+ (Vysokopevnostní beton – HPC)")

        with res_col2:
            st.metric("Vodní součinitel w/c poměr", f"{wc_ratio:.2f}",
                      help="Ideální poměr pro konstrukční beton je 0.40 až 0.55.")


if __name__ == "__main__":
    render_concrete_linear_regression_view()
