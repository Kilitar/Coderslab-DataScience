"""
Den 3: XGBoost – Cvičení 2 (Regrese cen diamantů) – Výsledky zadání
==================================================================
Zpracování oficiálního zadání kurzu:
1. Načtení dat diamonds_preprocessed.csv (37 729 trénovacích, 16 171 testovacích vzorků).
2. Rozdělení v poměru 70:30 (random_state=42).
3. Vytvoření xgb_regressor s random_state=42 a n_jobs=-1.
4. Definice hyperparametrické mřížky params (max_depth, min_samples_leaf, n_estimators).
5. RandomizedSearchCV s optimalizací na metriku mean_absolute_error (neg_mean_absolute_error).
6. Predikce na testovací sadě a výpočet testovacího MAE a R².
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


@st.cache_data
def load_diamonds_xgb_data():
    base_dir = Path(__file__).resolve().parent.parent
    json_path = base_dir / "03_Advanced_ML_Neural_Networks" / "data" / "diamonds_xgb_exercise_2_precomputed.json"
    if not json_path.exists():
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def render_xgboost_diamonds_exercise_2_view():
    st.title("🎯 Cvičení 2: XGBoost Regrese – Predikce cen diamantů")
    st.markdown(
        r"""
        V tomto cvičení stavíme regresní model cen diamantů pomocí knihovny **XGBoost** (`xgb.XGBRegressor`).
        Ladíme hyperparametry pomocí **`RandomizedSearchCV`** s cílem minimalizovat **průměrnou absolutní chybu (MAE)**.
        """
    )

    data = load_diamonds_xgb_data()
    if not data:
        st.error("Předpočtená data nebyla nalezena. Spusťte prosím skript `03_Advanced_ML_Neural_Networks/07_xgboost_diamonds_exercise_2.py`.")
        return

    school = data["school_model"]
    baselines = data["baselines"]
    n_train = data["dataset"]["n_train"]
    n_test = data["dataset"]["n_test"]

    # KPI Metriky
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(
            label="Testovací MAE (Školní model)",
            value=f"{school['test_mae']:.2f} USD",
            delta=f"-{baselines['dt']['mae'] - school['test_mae']:.2f} USD vs. Strom",
            delta_color="normal"
        )
    with k2:
        st.metric(
            label="Koeficient determinace R²",
            value=f"{school['test_r2']*100:.2f} %",
            delta=f"+{(school['test_r2'] - baselines['dt']['r2'])*100:.2f} % vs. Strom"
        )
    with k3:
        st.metric(
            label="Čas trénování (8 iterací, 3-fold)",
            value=f"{school['fit_duration_s']:.2f} s",
            delta="Extrémní CPU paralelizace"
        )
    with k4:
        st.metric(
            label="Trénovací vzorky",
            value=f"{n_train:,}",
            delta=f"Test: {n_test:,} (poměr 70:30)"
        )

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Výsledky & Hyperparametry",
        "⚖️ Srovnání s baselines (DT a RF)",
        "🔍 Rezidua a predikce",
        "💻 Zdrojový kód zadání"
    ])

    # =========================================================================
    # TAB 1: VÝSLEDKY A HYPERPARAMETRY
    # =========================================================================
    with tab1:
        st.subheader("1. Nejlepší nalezené hyperparametry z RandomizedSearchCV")
        st.markdown(
            r"""
            V souladu se zadáním byla definována mřížka hyperparametrů:
            - `max_depth`: `[3, 6, 9]`
            - `min_samples_leaf`: `[1, 2, 4]` *(zadání kurzu)*
            - `n_estimators`: `[50, 100, 150]`
            - Metrika optimalizace: `scoring="neg_mean_absolute_error"`
            """
        )

        col_p1, col_p2 = st.columns([1, 1])
        with col_p1:
            st.info(
                f"""
                #### 🏆 Vítězná konfigurace:
                - **`max_depth`**: `{school['best_params']['max_depth']}`
                - **`n_estimators`**: `{school['best_params']['n_estimators']}`
                - **`min_samples_leaf`**: `{school['best_params']['min_samples_leaf']}`
                - **CV MAE (křížová validace na trénovací sadě)**: **{school['cv_mae']:.2f} USD**
                - **Testovací MAE (na 16 171 diamantech)**: **{school['test_mae']:.2f} USD**
                - **Testovací RMSE**: **{school['test_rmse']:.2f} USD**
                """
            )
        with col_p2:
            st.warning(
                r"""
                #### ⚠️ Pozor na `min_samples_leaf` v XGBoost:
                Všimněte si, že v knihovně `xgboost` parametr `min_samples_leaf` **neexistuje**. 
                XGBoost při trénování vypsal varování:
                `WARNING: Parameters: { "min_samples_leaf" } are not used.`
                
                Zadání kurzu převzalo klíč z předchozího cvičení na Random Forest. 
                V sekci **Expertní analýza** ukazujeme, jak model reaguje na skutečný ekvivalent `min_child_weight`.
                """
            )

        st.markdown("---")
        st.subheader("Důležitost příznaků (Feature Importances)")
        feat_df = pd.DataFrame([
            {"Příznak": k, "Důležitost": v}
            for k, v in school["feature_importances"].items()
        ]).sort_values(by="Důležitost", ascending=True)

        fig_fi = px.bar(
            feat_df,
            x="Důležitost",
            y="Příznak",
            orientation="h",
            title="Důležitost příznaků (Gain / Relativní příspěvek ke snížení chyby)",
            color="Důležitost",
            color_continuous_scale="Viridis"
        )
        fig_fi.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_fi)

    # =========================================================================
    # TAB 2: SROVNÁNÍ MODELŮ
    # =========================================================================
    with tab2:
        st.subheader("2. Přímé srovnání: Rozhodovací strom vs. Random Forest vs. XGBoost")
        st.markdown(
            r"""
            Všechny tři modely byly natrénovány a vyhodnoceny na **zcela identickém 70:30 splitu (seed 42)** 
            datasetu diamantů:
            """
        )

        cmp_df = pd.DataFrame([
            {
                "Model": "🌲 Samostatný strom (DecisionTree)",
                "MAE (USD)": f"{baselines['dt']['mae']:.2f}",
                "RMSE (USD)": f"{baselines['dt']['rmse']:.2f}",
                "R² skóre": f"{baselines['dt']['r2']*100:.2f} %",
                "Princip": "Jeden hluboký strom (náchylný k přeučení)"
            },
            {
                "Model": "🌳 Náhodný les (Random Forest – Cvičení 2)",
                "MAE (USD)": f"{baselines['rf']['mae']:.2f}",
                "RMSE (USD)": f"{baselines['rf']['rmse']:.2f}",
                "R² skóre": f"{baselines['rf']['r2']*100:.2f} %",
                "Princip": "Bagging: 100 paralelních stromů průměrujících chybu"
            },
            {
                "Model": "🚀 XGBoost (Školní zadání)",
                "MAE (USD)": f"{school['test_mae']:.2f}",
                "RMSE (USD)": f"{school['test_rmse']:.2f}",
                "R² skóre": f"{school['test_r2']*100:.2f} %",
                "Princip": "Gradient Boosting: 50 sekvenčních stromů na reziduích"
            }
        ])
        st.table(cmp_df)

        fig_cmp = go.Figure(data=[
            go.Bar(name="MAE (Chyba v USD)", x=["Rozhodovací strom", "Random Forest", "XGBoost"], y=[baselines['dt']['mae'], baselines['rf']['mae'], school['test_mae']], marker_color=["#ef4444", "#3b82f6", "#10b981"])
        ])
        fig_cmp.update_layout(
            title="Srovnání průměrné absolutní chyby (MAE) v USD – menší je lepší",
            yaxis_title="MAE (USD)",
            height=380,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_cmp)

    # =========================================================================
    # TAB 3: REZIDUA A PREDIKCE
    # =========================================================================
    with tab3:
        st.subheader("3. Analýza predikcí a reziduí na testovací sadě")
        res_sample = pd.DataFrame(data["sample_residuals"])

        c_sc1, c_sc2 = st.columns(2)
        with c_sc1:
            fig_pred = px.scatter(
                res_sample,
                x="actual",
                y="pred_school",
                color="carat",
                title="Skutečná vs. Predikovaná cena (XGBoost)",
                labels={"actual": "Skutečná cena (USD)", "pred_school": "Predikovaná cena (USD)", "carat": "Karát"},
                color_continuous_scale="Plasma"
            )
            # Diagonála perfektní predikce
            max_val = max(res_sample["actual"].max(), res_sample["pred_school"].max())
            fig_pred.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines", name="Ideální shoda (y=x)", line=dict(color="gray", dash="dash")))
            fig_pred.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_pred)

        with c_sc2:
            fig_res = px.scatter(
                res_sample,
                x="pred_school",
                y="res_school",
                color="carat",
                title="Rezidua modelu (Skutečnost - Predikce)",
                labels={"pred_school": "Predikovaná cena (USD)", "res_school": "Reziduum (USD)", "carat": "Karát"},
                color_continuous_scale="Plasma"
            )
            fig_res.add_hline(y=0, line_dash="dash", line_color="red")
            fig_res.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_res)

        st.markdown("#### Ukázka testovacích diamantů a predikcí")
        st.dataframe(
            res_sample[["carat", "clarity", "color", "actual", "pred_school", "res_school"]].rename(
                columns={
                    "actual": "Skutečná cena (USD)",
                    "pred_school": "Predikce XGBoost (USD)",
                    "res_school": "Chyba / Reziduum (USD)"
                }
            ).head(10)
        )

    # =========================================================================
    # TAB 4: ZDROJOVÝ KÓD
    # =========================================================================
    with tab4:
        st.subheader("4. Kompletní kód řešení dle zadání")
        st.code(
            r'''
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

# 1. Načtení dat
df = pd.read_csv("data/diamonds_preprocessed.csv")

# 2. Oddělení X a y
X = df.drop("price", axis=1)
y = df["price"]

# 3. Rozdělení v poměru 70:30 (random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# 4. Instance modelu XGBoost regressor
xgb_regressor = xgb.XGBRegressor(random_state=42, n_jobs=-1)

# 5. Definice mřížky hyperparametrů
params = {
    "max_depth": [3, 6, 9],
    "min_samples_leaf": [1, 2, 4],
    "n_estimators": [50, 100, 150]
}

# 6. RandomizedSearchCV s optimalizací na mean_absolute_error
random_search = RandomizedSearchCV(
    estimator=xgb_regressor,
    param_distributions=params,
    scoring="neg_mean_absolute_error",
    n_iter=8,
    cv=3,
    random_state=42,
    n_jobs=-1
)

# 7. Trénování mřížky
random_search.fit(X_train, y_train)

# 8. Predikce na testovacích datech a výpočet MAE
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

test_mae = mean_absolute_error(y_test, y_pred)
print(f"Průměrná absolutní chyba (MAE) na testovací sadě: {test_mae:.2f} USD")
            ''',
            language="python"
        )


if __name__ == "__main__":
    render_xgboost_diamonds_exercise_2_view()
