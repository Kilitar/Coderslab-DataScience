"""
Homework: Rozhodovací strom (Decision Tree Regression) – Pevnost betonu (Concrete)
==================================================================================
Tento modul vizualizuje výsledky cvičení 'Decision tree (regression) - exercise 1':
1. GridSearchCV s 5-násobnou křížovou validací a metrikou MAE (scoring='neg_mean_absolute_error').
2. Nalezené optimální hyperparametry (best_hyperparams).
3. Efektivita modelu: R2 na Train i Test sadě, MSE a MAE na testovací sadě.
4. Architektura natrénovaného stromu (plot_tree).
5. Interaktivní Plotly vizualizace důležitosti příznaků (Feature Importances).
6. Interaktivní Plotly porovnání predikcí (Decision Tree vs. Lineární regrese OLS).
7. Živý simulátor pevnosti betonu (porovnání stromu a OLS v reálném čase).
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression


@st.cache_data
def load_fitted_tree_models():
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

        # OLS baseline
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        # Best tree from Exercise 1: criterion='squared_error', max_depth=None, min_samples_leaf=1
        tree = DecisionTreeRegressor(criterion="squared_error", max_depth=None, min_samples_leaf=1, random_state=42)
        tree.fit(X_train, y_train)

        test_df = X_test.copy()
        test_df["actual"] = y_test
        test_df["pred_tree"] = tree.predict(X_test)
        test_df["pred_ols"] = lr.predict(X_test)
        test_df["tree_error"] = np.abs(test_df["actual"] - test_df["pred_tree"])
        test_df["ols_error"] = np.abs(test_df["actual"] - test_df["pred_ols"])

        return df, tree, lr, test_df
    return None, None, None, None


def load_concrete_tree_precomputed():
    base_dir = Path(__file__).resolve().parent.parent
    cache_path = base_dir / "01_Regression" / "data" / "concrete_decision_tree_precomputed.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def render_concrete_decision_tree_view():
    st.title("🌳 DÚ: Rozhodovací strom pro regresi – Pevnost betonu")
    st.markdown(
        "**Vypracování cvičení:** Konstrukce regresního rozhodovacího stromu (`DecisionTreeRegressor`) "
        "s optimalizací hyperparametrů (`max_depth`, `criterion`, `min_samples_leaf`) "
        "pomocí `GridSearchCV` a evaluační metrikou **MAE**."
    )

    data = load_concrete_tree_precomputed()
    df_raw, tree_model, lr_model, test_eval_df = load_fitted_tree_models()

    if not data:
        st.error("Předpočtená data nebyla nalezena. Spusťte skript `01_Regression/13_homework_concrete_decision_tree.py`.")
        return

    meta = data["metadata"]
    best_hp = data["best_hyperparams"]
    m = data["metrics"]
    test_m = m["test"]
    train_m = m["train"]
    ols_comp = m["ols_comparison"]
    fi = data["feature_importances"]

    # KPI záhlaví
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "Testovací R²",
            f"{test_m['r2'] * 100:.2f} %",
            delta=f"+{ols_comp['r2_improvement']:.2f} % vs OLS",
            delta_color="normal"
        )
    with c2:
        st.metric(
            "Testovací MSE",
            f"{test_m['mse']:.2f}",
            delta=f"RMSE: {test_m['rmse']:.2f} MPa",
            delta_color="inverse"
        )
    with c3:
        st.metric(
            "Testovací MAE",
            f"{test_m['mae']:.2f} MPa",
            delta=f"-{ols_comp['mae_reduction']:.2f} MPa vs OLS",
            delta_color="inverse"
        )
    with c4:
        st.metric(
            "5-Fold CV MAE",
            f"{data['cv_results']['best_cv_mae']:.2f} MPa",
            delta=f"Kombinací: {data['cv_results']['total_combinations_tested']}",
            delta_color="off"
        )

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Výsledky & Hyperparametry",
        "🌳 Architektura stromu (plot_tree)",
        "📊 Důležitost příznaků (Plotly)",
        "⚔️ Srovnání s Lineární regresí (Plotly)",
        "🔍 Ukázka predikcí",
        "🎛️ Simulátor pevnosti (Live)"
    ])

    # TAB 1: Výsledky & Hyperparametry
    with tab1:
        st.subheader("1. Nalezené optimální hyperparametry (`best_hyperparams`)")
        st.markdown(
            "Metoda `GridSearchCV` prohledala mřížku **147 kombinací** (s 5-násobnou křížovou validací = 735 fitů) "
            "a jako optimální vybrala tuto sadu parametrů:"
        )

        col_l, col_r = st.columns([1, 1])
        with col_l:
            hp_table = pd.DataFrame([
                {"Hyperparametr": "criterion", "Nalezená hodnota": f"'{best_hp['criterion']}'", "Význam": "Kritérium dělícího pravidla (Squared Error)"},
                {"Hyperparametr": "max_depth", "Nalezená hodnota": str(best_hp['max_depth']), "Význam": "Maximální hloubka stromu (neomezená)"},
                {"Hyperparametr": "min_samples_leaf", "Nalezená hodnota": str(best_hp['min_samples_leaf']), "Význam": "Minimální počet vzorků v koncovém listu"}
            ])
            st.dataframe(hp_table, width="stretch", hide_index=True)

        with col_r:
            st.markdown(
                f"""
                ```python
                best_hyperparams = {best_hp}
                ```
                - **5-Fold CV MAE:** `{data['cv_results']['best_cv_mae']:.4f} MPa`
                - **Evaluační metrika:** `scoring='neg_mean_absolute_error'`
                - **Počet kombinací v mřížce:** `{data['cv_results']['total_combinations_tested']}`
                """
            )

        st.markdown("---")
        st.subheader("Evaluace modelu na trénovací a testovací sadě")

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("##### 🏋️ Trénovací sada (Train, 70 %)")
            tr_df = pd.DataFrame([
                {"Metrika": "Koeficient determinace (R²)", "Hodnota": f"{train_m['r2'] * 100:.2f} %"},
                {"Metrika": "Mean Squared Error (MSE)", "Hodnota": f"{train_m['mse']:.4f}"},
                {"Metrika": "Root Mean Squared Error (RMSE)", "Hodnota": f"{train_m['rmse']:.4f} MPa"},
                {"Metrika": "Mean Absolute Error (MAE)", "Hodnota": f"{train_m['mae']:.4f} MPa"},
                {"Metrika": "Mean Absolute Percentage Error (MAPE)", "Hodnota": f"{train_m['mape'] * 100:.2f} %"}
            ])
            st.dataframe(tr_df, width="stretch", hide_index=True)

        with col_m2:
            st.markdown("##### 🧪 Testovací sada (Test, 30 %)")
            te_df = pd.DataFrame([
                {"Metrika": "Koeficient determinace (R²)", "Hodnota": f"{test_m['r2'] * 100:.2f} %"},
                {"Metrika": "Mean Squared Error (MSE)", "Hodnota": f"{test_m['mse']:.4f}"},
                {"Metrika": "Root Mean Squared Error (RMSE)", "Hodnota": f"{test_m['rmse']:.4f} MPa"},
                {"Metrika": "Mean Absolute Error (MAE)", "Hodnota": f"{test_m['mae']:.4f} MPa"},
                {"Metrika": "Mean Absolute Percentage Error (MAPE)", "Hodnota": f"{test_m['mape'] * 100:.2f} %"}
            ])
            st.dataframe(te_df, width="stretch", hide_index=True)

        st.code(
            r"""# Kód vyhodnocení zadaných metrik:
y_pred_train = tree_reg.predict(X_train)
y_pred_test = tree_reg.predict(X_test)

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
test_mse = mean_squared_error(y_test, y_pred_test)
test_mae = mean_absolute_error(y_test, y_pred_test)""",
            language="python"
        )

        st.success(
            f"**Zhodnocení:** Rozhodovací strom dosahuje na testovací sadě koeficientu determinace **$R^2 = {test_m['r2']*100:.2f} %$**, "
            f"přičemž průměrná chyba predikce klesla na pouhých **{test_m['mae']:.2f} MPa**!"
        )

    # TAB 2: Architektura stromu
    with tab2:
        st.subheader("2. Vizualizace architektury natrénovaného stromu (`plot_tree`)")
        st.markdown(
            "Níže je zobrazena struktura prvních tří pater rozhodovacího stromu. "
            "Každý uzel ukazuje dělící podmínku, hodnotu kritéria (squared_error), počet vzorků a průměrnou hodnotu pevnosti."
        )

        tree_img_path = Path(__file__).resolve().parent.parent / "01_Regression" / "plots" / "concrete_tree_structure.png"
        if tree_img_path.exists():
            st.image(str(tree_img_path), caption="Rozhodovací strom pro odhad pevnosti betonu (vrchní 3 patra)", width="stretch")

        st.markdown("##### 🔬 Analýza prvních rozhodovacích pravidel:")
        st.markdown(
            """
            1. **Kořenový uzel (Root):** První a nejdůležitější rozdělení probíhá na základě **stáří betonu (`age`)** a **obsahu cementu (`cement`)**.
            2. **Větvení pro mladý beton vs. vyzrálý beton:** Strom okamžitě odděluje vzorky mladší než 28 dní od vzorků s dlouhým zráním.
            3. **Interakce cementu s vodou:** V listech stromu se kombinují prahy nízkého obsahu vody s vysokým obsahem cementu, což přirozeně emuluje Abramsův zákon vodního součinitele.
            """
        )

    # TAB 3: Důležitost příznaků (Plně interaktivní Plotly)
    with tab3:
        st.subheader("3. Interaktivní relativní důležitost příznaků (Feature Importances)")
        st.markdown(
            "Důležitost příznaku v regresním stromu měří celkové snížení rozptylu cíle (variance reduction), "
            "které daný prediktor přinesl napříč všemi svými rozštěpeními."
        )

        fi_df = pd.DataFrame([
            {
                "feature": feat,
                "importance": imp,
                "description": meta["feature_descriptions"][feat]
            }
            for feat, imp in sorted(fi.items(), key=lambda x: x[1], reverse=True)
        ])

        fig_fi = px.bar(
            fi_df.sort_values(by="importance", ascending=True),
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Viridis",
            labels={"importance": "Relativní důležitost", "feature": "Vstupní složka"},
            title="Důležitost příznaků v modelu DecisionTreeRegressor",
            text_auto=".2%"
        )
        fig_fi.update_layout(height=450, margin=dict(l=20, r=20, t=30, b=20), coloraxis_showscale=False)
        st.plotly_chart(fig_fi, width="stretch")

        col_f1, col_f2 = st.columns([1, 1])
        with col_f1:
            st.markdown("##### 📋 Přehledová tabulka důležitosti")
            fi_table_df = pd.DataFrame([
                {
                    "Proměnná": row["feature"],
                    "Popis": row["description"],
                    "Důležitost (%)": f"{row['importance'] * 100:.2f} %"
                }
                for _, row in fi_df.iterrows()
            ])
            st.dataframe(fi_table_df, width="stretch", hide_index=True)

        with col_f2:
            st.markdown("##### 💡 Inženýrská interpretace")
            st.info(
                """
                - **Cement (36.09 %) & Stáří (31.69 %):** Společně vysvětlují **téměř 68 %** celkové predikční síly!
                - **Voda (12.07 %) & Struska (10.64 %):** Tvoří druhou klíčovou dvojici – množství záměsové vody a přídavek hydraulické strusky.
                - **Ostatní složky (popílek, kamenivo):** Slouží spíše k jemnému doladění lokálních vlastností směsi.
                """
            )

    # TAB 4: Srovnání s Lineární regresí (Plně interaktivní Plotly Scatter)
    with tab4:
        st.subheader("4. Interaktivní srovnání predikcí: Rozhodovací strom vs. Lineární regrese")
        st.markdown(
            "Níže vidíte interaktivní srovnání skutečné pevnosti proti predikcím obou modelů na 302 testovacích vzorcích. "
            "Body rozhodovacího stromu (zelená) leží podstatně blíže červené diagonále ideální predikce ($y = x$)."
        )

        if test_eval_df is not None:
            fig_comp = go.Figure()
            # OLS body
            fig_comp.add_trace(go.Scatter(
                x=test_eval_df["actual"],
                y=test_eval_df["pred_ols"],
                mode="markers",
                name="Lineární regrese OLS (R² = 56.1 %)",
                marker=dict(color="#1f77b4", size=6, opacity=0.6),
                hovertemplate="Skutečnost: %{x:.2f} MPa<br>Predikce OLS: %{y:.2f} MPa<extra></extra>"
            ))
            # Decision Tree body
            fig_comp.add_trace(go.Scatter(
                x=test_eval_df["actual"],
                y=test_eval_df["pred_tree"],
                mode="markers",
                name="Rozhodovací strom (R² = 84.8 %)",
                marker=dict(color="#2ca02c", size=7, opacity=0.85),
                hovertemplate="Skutečnost: %{x:.2f} MPa<br>Predikce Strom: %{y:.2f} MPa<extra></extra>"
            ))
            # Diagonála
            min_v = test_eval_df["actual"].min()
            max_v = test_eval_df["actual"].max()
            fig_comp.add_trace(go.Scatter(
                x=[min_v, max_v],
                y=[min_v, max_v],
                mode="lines",
                name="Ideální shoda (y = x)",
                line=dict(color="red", dash="dash", width=2)
            ))
            fig_comp.update_layout(
                title="Actual vs. Predicted: Rozhodovací strom (zelená) vs. OLS (modrá)",
                xaxis_title="Skutečná pevnost v tlaku (MPa)",
                yaxis_title="Predikovaná pevnost (MPa)",
                height=520,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig_comp, width="stretch")

        st.markdown("##### 📊 Srovnávací tabulka modelů na testovací sadě")
        comp_df = pd.DataFrame([
            {
                "Model": "Lineární regrese OLS (Cvičení 1)",
                "Test R² (%)": f"{ols_comp['r2'] * 100:.2f} %",
                "RMSE (MPa)": f"{ols_comp['rmse']:.4f}",
                "MAE (MPa)": f"{ols_comp['mae']:.4f}",
                "Typ modelu": "Lineární, parametrický"
            },
            {
                "Model": "Rozhodovací strom (Cvičení 2)",
                "Test R² (%)": f"{test_m['r2'] * 100:.2f} %",
                "RMSE (MPa)": f"{test_m['rmse']:.4f}",
                "MAE (MPa)": f"{test_m['mae']:.4f}",
                "Typ modelu": "Nelineární, neparametrický"
            },
            {
                "Model": "Rozdíl / Zlepšení",
                "Test R² (%)": f"+{ols_comp['r2_improvement']:.2f} %",
                "RMSE (MPa)": f"-{ols_comp['rmse_reduction']:.4f} (-{(1 - test_m['rmse']/ols_comp['rmse'])*100:.1f} %)",
                "MAE (MPa)": f"-{ols_comp['mae_reduction']:.4f} (-{(1 - test_m['mae']/ols_comp['mae'])*100:.1f} %)",
                "Typ modelu": "Dramatický pokles chyb!"
            }
        ])
        st.dataframe(comp_df, width="stretch", hide_index=True)

        st.markdown(
            r"""
            #### 🧠 Proč strom tak drtivě porazil lineární regresi?
            1. **Logaritmická křivka zrání:** Růst pevnosti v čase není lineární. Během prvních 28 dní roste exponenciálně, poté se křivka láme do roviny. Strom tuto křivku přirozeně aproximuje schodovitou funkcí ($age \le 7, 14, 28, 90$).
            2. **Synergické interakce ($w/c$ poměr):** Pevnost fyzikálně závisí na podílu $\frac{\text{voda}}{\text{cement}}$. Strom dokáže rozdělit vzorky nejprve podle cementu a v další větvi podle vody, čímž nelineární podíl přesně zachytí.
            """
        )

    # TAB 5: Ukázka predikcí
    with tab5:
        st.subheader("5. Porovnání predikcí na konkrétních testovacích vzorcích")
        st.markdown("Prohlédněte si srovnání chyb u náhodně vybraných zkušebních těles betonu z testovací sady.")

        preds_table = pd.DataFrame(data["sample_predictions"])
        st.dataframe(preds_table, width="stretch", hide_index=True)

    # TAB 6: Simulátor pevnosti
    with tab6:
        st.subheader("6. 🎛️ Živý simulátor pevnosti betonu (Strom vs. OLS)")
        st.markdown(
            "Nastavte parametry betonové směsi v reálných jednotkách a porovnejte predikci "
            "rozhodovacího stromu a lineárního modelu v reálném čase."
        )

        sim_c1, sim_c2 = st.columns(2)
        with sim_c1:
            s_cement = st.slider("Cement (kg/m³):", 100.0, 550.0, 320.0, 5.0, key="dt_s_cement")
            s_slag = st.slider("Vysokopecní struska (kg/m³):", 0.0, 360.0, 50.0, 5.0, key="dt_s_slag")
            s_flyash = st.slider("Popílek (kg/m³):", 0.0, 200.0, 0.0, 5.0, key="dt_s_flyash")
            s_water = st.slider("Záměsová voda (kg/m³):", 120.0, 250.0, 175.0, 2.0, key="dt_s_water")
        with sim_c2:
            s_sp = st.slider("Superplastifikátor (kg/m³):", 0.0, 35.0, 8.0, 0.5, key="dt_s_sp")
            s_coarse = st.slider("Hrubé kamenivo (kg/m³):", 800.0, 1150.0, 950.0, 10.0, key="dt_s_coarse")
            s_fine = st.slider("Jemné kamenivo / písek (kg/m³):", 590.0, 1000.0, 780.0, 10.0, key="dt_s_fine")
            s_age = st.slider("Doba zrání (dny):", 1, 365, 28, 1, key="dt_s_age")

        means = {"cement": 281.17, "slag": 73.90, "flyash": 54.19, "water": 181.57, "superplasticizer": 6.20, "coarseaggregate": 972.92, "fineaggregate": 773.58, "age": 45.66}
        stds = {"cement": 104.51, "slag": 86.28, "flyash": 63.99, "water": 21.36, "superplasticizer": 5.97, "coarseaggregate": 77.75, "fineaggregate": 80.18, "age": 63.17}

        input_z = np.array([[
            (s_cement - means["cement"]) / stds["cement"],
            (s_slag - means["slag"]) / stds["slag"],
            (s_flyash - means["flyash"]) / stds["flyash"],
            (s_water - means["water"]) / stds["water"],
            (s_sp - means["superplasticizer"]) / stds["superplasticizer"],
            (s_coarse - means["coarseaggregate"]) / stds["coarseaggregate"],
            (s_fine - means["fineaggregate"]) / stds["fineaggregate"],
            (s_age - means["age"]) / stds["age"]
        ]])

        if tree_model is not None and lr_model is not None:
            pred_t = tree_model.predict(input_z)[0]
            pred_l = lr_model.predict(input_z)[0]

            st.markdown("---")
            res1, res2, res3 = st.columns(3)
            res1.metric("Predikce: Rozhodovací strom", f"{pred_t:.2f} MPa", delta=f"{pred_t - pred_l:+.2f} MPa vs OLS")
            res2.metric("Predikce: Lineární regrese", f"{pred_l:.2f} MPa")
            res3.metric("Vodní součinitel w/c", f"{s_water / s_cement:.2f}", help="Hmotnostní poměr vody a cementu")


if __name__ == "__main__":
    render_concrete_decision_tree_view()
