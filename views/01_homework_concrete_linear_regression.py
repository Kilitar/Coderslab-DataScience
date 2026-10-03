"""
Homework: Lineární regrese – Pevnost betonu (Concrete Compressive Strength)
===========================================================================
Tento modul vizualizuje výsledky cvičení lineární regrese ze zadání:
1. Rozdělení 70/30 (random_state=42).
2. Třída LinearRegression z Scikit-learn (instance linear_reg).
3. Evaluační metriky na testovací sadě (R2, RMSE, MAE, MAPE).
4. Fyzikální interpretace standardizovaných regresních koeficientů (Beta weights).
5. Diagnostika reziduí a meze použitelnosti lineárního modelu pro betonový kompozit.
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


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
        "**Vypracování cvičení:** Konstrukce a vyhodnocení baseline modelu `LinearRegression` "
        "na normalizovaném datasetu pevných směsí betonu (*Concrete Compressive Strength*)."
    )

    data = load_concrete_lr_precomputed()
    if not data:
        st.error("Předpočtená data nebyla nalezena. Spusťte skript `01_Regression/11_homework_concrete_linear_regression.py`.")
        return

    meta = data["metadata"]
    metrics = data["metrics"]
    train_m = metrics["train"]
    test_m = metrics["test"]
    params = data["model_parameters"]
    intercept = params["intercept"]
    coefs = params["coefficients"]

    # KPI záhlaví
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "Testovací R²",
            f"{test_m['r2'] * 100:.2f} %",
            delta=f"{(test_m['r2'] - train_m['r2']) * 100:.2f} % vs Train",
            delta_color="normal"
        )
    with c2:
        st.metric(
            "Testovací RMSE",
            f"{test_m['rmse']:.2f} MPa",
            delta=f"{test_m['rmse'] - train_m['rmse']:+.2f} MPa",
            delta_color="inverse"
        )
    with c3:
        st.metric(
            "Testovací MAE",
            f"{test_m['mae']:.2f} MPa",
            delta=f"{test_m['mae'] - train_m['mae']:+.2f} MPa",
            delta_color="inverse"
        )
    with c4:
        st.metric(
            "Relativní chyba (MAPE)",
            f"{test_m['mape']:.1f} %",
            help="Průměrná procentuální chyba predikce pevnosti."
        )

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 1. Výsledky & Metriky",
        "⚖️ 2. Regresní koeficienty (Beta)",
        "📉 3. Skutečnost vs. Predikce",
        "🔬 4. Diagnostika reziduí",
        "💡 5. Limity lineárního modelu"
    ])

    # TAB 1: Výsledky & Metriky
    with tab1:
        st.subheader("1. Vyhodnocení efektivity modelu (Test Set 30 %)")
        st.markdown(
            f"Model byl natrénován na **{meta['train_samples']} vzorcích (70 %)** "
            f"a otestován na **{meta['test_samples']} neviděných vzorcích (30 %)** se zafixovaným `random_state=42`."
        )

        col_m1, col_m2 = st.columns([1, 1])
        with col_m1:
            st.markdown("##### 📊 Srovnání metrik Train vs. Test")
            metrics_table = pd.DataFrame([
                {
                    "Metrika": "Koeficient determinace (R²)",
                    "Trénovací sada (Train)": f"{train_m['r2'] * 100:.2f} %",
                    "Testovací sada (Test)": f"{test_m['r2'] * 100:.2f} %",
                    "Rozdíl": f"{(test_m['r2'] - train_m['r2']) * 100:+.2f} %"
                },
                {
                    "Metrika": "Root Mean Squared Error (RMSE)",
                    "Trénovací sada (Train)": f"{train_m['rmse']:.3f} MPa",
                    "Testovací sada (Test)": f"{test_m['rmse']:.3f} MPa",
                    "Rozdíl": f"{test_m['rmse'] - train_m['rmse']:+.3f} MPa"
                },
                {
                    "Metrika": "Mean Absolute Error (MAE)",
                    "Trénovací sada (Train)": f"{train_m['mae']:.3f} MPa",
                    "Testovací sada (Test)": f"{test_m['mae']:.3f} MPa",
                    "Rozdíl": f"{test_m['mae'] - train_m['mae']:+.3f} MPa"
                },
                {
                    "Metrika": "Mean Absolute Percentage Error (MAPE)",
                    "Trénovací sada (Train)": f"{train_m['mape']:.2f} %",
                    "Testovací sada (Test)": f"{test_m['mape']:.2f} %",
                    "Rozdíl": f"{test_m['mape'] - train_m['mape']:+.2f} %"
                },
                {
                    "Metrika": "Maximální chyba (Max Error)",
                    "Trénovací sada (Train)": f"{train_m['max_error']:.2f} MPa",
                    "Testovací sada (Test)": f"{test_m['max_error']:.2f} MPa",
                    "Rozdíl": f"{test_m['max_error'] - train_m['max_error']:+.2f} MPa"
                }
            ])
            st.dataframe(metrics_table, width="stretch", hide_index=True)

        with col_m2:
            st.markdown("##### 📝 Kód řešení dle zadání")
            st.code(
                """from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import pandas as pd

# 1. Načtení předzpracovaných dat
df = pd.read_csv('concrete_data_preprocessed.csv')
X = df.drop(columns=['csMPa'])
y = df['csMPa']

# 2. Rozdělení 70/30
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Instance modelu dle zadání
linear_reg = LinearRegression()

# 4. Trénování a testování
linear_reg.fit(X_train, y_train)
y_pred = linear_reg.predict(X_test)

# 5. Vyhodnocení metrik
r2 = r2_score(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"R2: {r2:.4f}, RMSE: {rmse:.2f} MPa, MAE: {mae:.2f} MPa")""",
                language="python"
            )

        st.info(
            r"**Závěr k metrikám:** Model dosahuje na testovací sadě **$R^2 = 56.08\ \%$**, což znamená, "
            r"že více než polovina rozptylu pevnosti betonu je vysvětlena jednoduchou lineární kombinací složek. "
            r"Průměrná odchylka činí **$8.98\ \text{MPa}$**."
        )

    # TAB 2: Regresní koeficienty
    with tab2:
        st.subheader("2. Standardizované regresní koeficienty (Beta weights)")
        st.markdown(
            "Jelikož byly všechny prediktory standardizovány (`StandardScaler`), "
            "koeficienty $\\beta_j$ mají společné měřítko a přímo vyjadřují relativní váhu každé složky. "
            "Koeficient udává změnu pevnosti betonu v MPa při zvýšení dané složky o **1 směrodatnou odchylku**."
        )

        coef_img_path = Path(__file__).resolve().parent.parent / "01_Regression" / "plots" / "concrete_lr_coefficients.png"
        if coef_img_path.exists():
            st.image(str(coef_img_path), caption="Standardizované regresní koeficienty modelu LinearRegression", width="stretch")

        col_c1, col_c2 = st.columns([1, 1])
        with col_c1:
            st.markdown("##### 📋 Tabulka parametrů modelu")
            coef_list = [
                {
                    "Proměnná": feat,
                    "Popis složky": meta["feature_descriptions"][feat],
                    "Koeficient β (MPa/1σ)": f"{coef:+.4f}",
                    "Směr vlivu": "Pozitivní 🟢" if coef > 0 else "Negativní 🔴"
                }
                for feat, coef in sorted(coefs.items(), key=lambda x: x[1], reverse=True)
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

    # TAB 3: Skutečnost vs. Predikce
    with tab3:
        st.subheader("3. Skutečnost vs. Predikce (Actual vs. Predicted)")
        st.markdown(
            "Porovnání skutečné naměřené pevnosti betonu z testovací sady proti hodnotě predikované modelem. "
            "Červená přerušovaná čára představuje ideální predikci ($y = \\hat{y}$)."
        )

        pred_img_path = Path(__file__).resolve().parent.parent / "01_Regression" / "plots" / "concrete_lr_actual_vs_predicted.png"
        if pred_img_path.exists():
            st.image(str(pred_img_path), caption="Scatter plot skutečné vs. predikované pevnosti betonu", width="stretch")

        st.markdown("##### 🔍 Ukázka predikcí na konkrétních testovacích vzorcích")
        test_samples_df = pd.DataFrame(data["test_sample_predictions"])
        st.dataframe(test_samples_df, width="stretch", hide_index=True)

    # TAB 4: Diagnostika reziduí
    with tab4:
        st.subheader("4. Diagnostika reziduí (Chyby predikce)")
        st.markdown(
            "Správný lineární model by měl mít rezidua ($e = y - \\hat{y}$) s nulovým průměrem, "
            "konstantním rozptylem (homoskedasticita) a přibližně normálním rozdělením."
        )

        res_img_path = Path(__file__).resolve().parent.parent / "01_Regression" / "plots" / "concrete_lr_residuals.png"
        if res_img_path.exists():
            st.image(str(res_img_path), caption="Diagnostika reziduí: Scatter proti predikci a histogram rozdělení chyb", width="stretch")

        res_info = data["residuals"]
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Průměr reziduí (Test)", f"{res_info['test_mean']:+.2f} MPa", help="Ideál je 0.0.")
        rc2.metric("Směrodatná odchylka reziduí", f"{res_info['test_std']:.2f} MPa")
        rc3.metric("Šikmost rozdělení chyb", f"{res_info['test_skewness']:+.2f}", help="Blízko 0 = symetrické normální chyby.")

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
            
            #### 3. Přítomnost superplastifikátoru a strusky:
            Účinek superplastifikátoru se projeví pouze tehdy, když je přítomna voda, kterou ztekucuje. 
            Jde o klasický interakční člen ($\text{voda} \times \text{plastifikátor}$).
            
            > [!TIP]
            > **Doporučení pro navazující cvičení:** V dalších úlohách vyzkoušíme **polynomiální regresi** (která přidá nelinearity a interakční členy), **regularizované modely (Ridge/Lasso)** a **rozhodovací stromy / Random Forest**, které dokáží nelineární chování betonu zachytit mnohem přesněji!
            """
        )


if __name__ == "__main__":
    render_concrete_linear_regression_view()
