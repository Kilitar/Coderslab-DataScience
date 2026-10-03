"""
Homework: Rozhodovací strom (Decision Tree Regression) – Pevnost betonu (Concrete)
==================================================================================
Tento modul vizualizuje výsledky cvičení 'Decision tree (regression) - exercise 1':
1. GridSearchCV s 5-násobnou křížovou validací a metrikou MAE (scoring='neg_mean_absolute_error').
2. Nalezené optimální hyperparametry (best_hyperparams).
3. Efektivita modelu: R2 na Train i Test sadě, MSE a MAE na testovací sadě.
4. Vizualizace struktury natrénovaného stromu (plot_tree).
5. Důležitost příznaků (Feature Importances) a dramatické srovnání s lineární regresí (skok z 56 % na 84.8 %).
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


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
            "Validační CV MAE",
            f"{data['best_cv_mae']:.2f} MPa",
            help="Průměrná absolutní chyba v 5-násobné křížové validaci."
        )

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 1. Výsledky & Metriky",
        "🌳 2. Architektura stromu",
        "📊 3. Důležitost příznaků",
        "📈 4. Srovnání s Lineární regresí",
        "🔍 5. Ukázka predikcí"
    ])

    # TAB 1: Výsledky & Metriky
    with tab1:
        st.subheader("1. Vyhodnocení efektivity modelu dle zadání")
        st.markdown(
            f"Model byl vyladěn pomocí **`GridSearchCV` s 5-násobnou křížovou validací** na {meta['train_samples']} trénovacích vzorcích "
            f"a vyhodnocen na {meta['test_samples']} testovacích vzorcích."
        )

        col_m1, col_m2 = st.columns([1, 1])
        with col_m1:
            st.markdown("##### 🏆 Nalezené optimální hyperparametry (`best_hyperparams`)")
            st.code(
                f"""best_hyperparams = {{
    'criterion': '{best_hp['criterion']}',
    'max_depth': {best_hp['max_depth']},
    'min_samples_leaf': {best_hp['min_samples_leaf']}
}}
# Nejlepší validační skóre (CV MAE): {data['best_cv_mae']:.4f} MPa""",
                language="python"
            )

            st.markdown("##### 📋 Požadované evaluační metriky zadání")
            metrics_table = pd.DataFrame([
                {
                    "Metrika": "Koeficient determinace (R²) – Trénovací sada",
                    "Hodnota": f"{train_m['r2'] * 100:.2f} %",
                    "Význam": "Schopnost vysvětlit variabilitu na trénovacích datech."
                },
                {
                    "Metrika": "Koeficient determinace (R²) – Testovací sada",
                    "Hodnota": f"{test_m['r2'] * 100:.2f} %",
                    "Význam": "Skutečná generalizační schopnost na neviděných datech."
                },
                {
                    "Metrika": "Mean Squared Error (MSE) – Test",
                    "Hodnota": f"{test_m['mse']:.4f}",
                    "Význam": "Střední kvadratická chyba predikcí."
                },
                {
                    "Metrika": "Root Mean Squared Error (RMSE) – Test",
                    "Hodnota": f"{test_m['rmse']:.4f} MPa",
                    "Význam": "Typická odmocninová odchylka v MPa."
                },
                {
                    "Metrika": "Mean Absolute Error (MAE) – Test",
                    "Hodnota": f"{test_m['mae']:.4f} MPa",
                    "Význam": "Průměrná absolutní chyba odhadu v MPa."
                },
                {
                    "Metrika": "Mean Absolute Percentage Error (MAPE)",
                    "Hodnota": f"{test_m['mape']:.2f} %",
                    "Význam": "Průměrná relativní chyba odhadu."
                }
            ])
            st.dataframe(metrics_table, width="stretch", hide_index=True)

        with col_m2:
            st.markdown("##### 📝 Kód vypracovaného řešení")
            st.code(
                """from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Definice mřížky hyperparametrů dle zadání
param_grid = {
    'max_depth': [3, 5, 7, 9, 11, 13, 15, None],
    'criterion': ['squared_error', 'absolute_error'],
    'min_samples_leaf': [1, 2, 4, 6, 8, 12]
}

# 2. GridSearchCV s metrikou MAE
grid_search = GridSearchCV(
    estimator=DecisionTreeRegressor(random_state=42),
    param_grid=param_grid,
    scoring='neg_mean_absolute_error',
    cv=5,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

# 3. Uložení nalezených parametrů
best_hyperparams = grid_search.best_params_

# 4. Natrénování finálního modelu
tree_reg = DecisionTreeRegressor(**best_hyperparams, random_state=42)
tree_reg.fit(X_train, y_train)

# 5. Vyhodnocení metrik
y_pred_train = tree_reg.predict(X_train)
y_pred_test = tree_reg.predict(X_test)

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
test_mse = mean_squared_error(y_test, y_pred_test)
test_mae = mean_absolute_error(y_test, y_pred_test)""",
                language="python"
            )

        st.success(
            f"**Zhodnocení:** Rozhodovací strom dosahuje na testovací sadě koeficientu determinace **$R^2 = {test_m['r2']*100:.2f} \\%$**, "
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

    # TAB 3: Důležitost příznaků
    with tab3:
        st.subheader("3. Relativní důležitost příznaků (Feature Importances)")
        st.markdown(
            "Důležitost příznaku v regresním stromu měří celkové snížení rozptylu cíle (variance reduction), "
            "které daný prediktor přinesl napříč všemi svými rozštěpeními."
        )

        fi_img_path = Path(__file__).resolve().parent.parent / "01_Regression" / "plots" / "concrete_tree_feature_importance.png"
        if fi_img_path.exists():
            st.image(str(fi_img_path), caption="Důležitost složek směsi v modelu DecisionTreeRegressor", width="stretch")

        col_f1, col_f2 = st.columns([1, 1])
        with col_f1:
            st.markdown("##### 📋 Přehledová tabulka důležitosti")
            fi_df = pd.DataFrame([
                {
                    "Proměnná": feat,
                    "Popis": meta["feature_descriptions"][feat],
                    "Důležitost (%)": f"{imp * 100:.2f} %"
                }
                for feat, imp in fi.items()
            ])
            st.dataframe(fi_df, width="stretch", hide_index=True)

        with col_f2:
            st.markdown("##### 💡 Inženýrská interpretace")
            st.info(
                """
                - **Cement (36.09 %) & Stáří (31.69 %):** Společně vysvětlují **téměř 68 %** celkové predikční síly!
                - **Voda (12.07 %) & Struska (10.64 %):** Tvoří druhou klíčovou dvojici – množství záměsové vody a přídavek hydraulické strusky.
                - **Ostatní složky (popílek, kamenivo):** Slouží spíše k jemnému doladění lokálních vlastností směsi.
                """
            )

    # TAB 4: Srovnání s Lineární regresí
    with tab4:
        st.subheader("4. Srovnání modelů: Rozhodovací strom vs. Lineární regrese")
        st.markdown(
            "Porovnání predikcí ukazuje zásadní rozdíl mezi rigidním lineárním modelem a flexibilním rozhodovacím stromem."
        )

        pred_img_path = Path(__file__).resolve().parent.parent / "01_Regression" / "plots" / "concrete_tree_actual_vs_predicted.png"
        if pred_img_path.exists():
            st.image(str(pred_img_path), caption="Skutečnost vs. Predikce: Decision Tree (zelená) vs. Lineární regrese (modrá)", width="stretch")

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
                "Model": "⚡ Zlepšení / Rozdíl",
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


if __name__ == "__main__":
    render_concrete_decision_tree_view()
