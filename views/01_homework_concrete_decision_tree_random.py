"""
Homework: Rozhodovací strom (RandomizedSearchCV) – Pevnost betonu (Concrete)
============================================================================
Tento modul vizualizuje výsledky cvičení 'Decision tree (regression) - exercise 2':
1. RandomizedSearchCV s 5-násobnou křížovou validací a metrikou MAE (scoring='neg_mean_absolute_error').
2. Nalezené optimální hyperparametry (best_hyperparams: max_depth, criterion, min_samples_leaf, max_features).
3. Efektivita modelu: R2 na Train i Test sadě, MSE a MAE na testovací sadě.
4. Architektura natrénovaného stromu (plot_tree).
5. Interaktivní Plotly vizualizace důležitosti příznaků.
6. Interaktivní analýza průběhu RandomizedSearch (Plotly scatter).
7. Interaktivní srovnání modelů (Grid vs. Random vs. OLS).
8. Živý interaktivní simulátor pevnosti betonu.
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


@st.cache_data
def load_fitted_random_tree():
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

        # Best tree from Exercise 2: criterion='poisson', max_depth=25, max_features=6, min_samples_leaf=1
        tree = DecisionTreeRegressor(
            criterion="poisson",
            max_depth=25,
            max_features=6,
            min_samples_leaf=1,
            random_state=42
        )
        tree.fit(X_train, y_train)
        return df, tree
    return None, None


def load_concrete_tree_random_precomputed():
    base_dir = Path(__file__).resolve().parent.parent
    cache_path = base_dir / "01_Regression" / "data" / "concrete_decision_tree_random_precomputed.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def render_concrete_decision_tree_random_view():
    st.title("🎲 DÚ: Rozhodovací strom – RandomizedSearch")
    st.markdown(
        "**Vypracování cvičení 2:** Konstrukce regresního rozhodovacího stromu (`DecisionTreeRegressor`) "
        "s optimalizací hyperparametrů (`max_depth`, `criterion`, `min_samples_leaf`, `max_features`) "
        "pomocí náhodného vzorkování `RandomizedSearchCV` a evaluační metrikou **MAE**."
    )

    data = load_concrete_tree_random_precomputed()
    df_raw, tree_model = load_fitted_random_tree()

    if not data:
        st.error("Předpočtená data nebyla nalezena. Spusťte skript `01_Regression/14_homework_concrete_decision_tree_random.py`.")
        return

    dataset_info = data["dataset_info"]
    search_space = data["search_space"]
    best_hp = data["best_hyperparams"]
    m = data["metrics"]
    test_m = m["test"]
    train_m = m["train"]
    comp_models = data.get("comparison_models", {})
    fi = data["feature_importances"]

    # Rozdíl oproti OLS
    ols_test = comp_models.get("ols", {"r2": 0.5608, "rmse": 11.2354, "mae": 8.9818})
    r2_diff_ols = (test_m["r2"] - ols_test["r2"]) * 100
    mae_diff_ols = ols_test["mae"] - test_m["mae"]

    # KPI záhlaví
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "Testovací R²",
            f"{test_m['r2'] * 100:.2f} %",
            delta=f"+{r2_diff_ols:.2f} % vs OLS",
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
            delta=f"-{mae_diff_ols:.2f} MPa vs OLS",
            delta_color="inverse"
        )
    with c4:
        st.metric(
            "5-Fold CV MAE",
            f"{data['cv_score_mae']:.2f} MPa",
            delta=f"{search_space['n_iter']} pokusů",
            delta_color="off"
        )

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Výsledky & Hyperparametry",
        "🌳 Architektura stromu (plot_tree)",
        "📊 Důležitost příznaků (Plotly)",
        "🎲 Průběh RandomizedSearch (Plotly)",
        "⚔️ Srovnání modelů (Plotly)",
        "🎛️ Simulátor pevnosti (Live)"
    ])

    # TAB 1: Výsledky & Hyperparametry
    with tab1:
        st.subheader("1. Nalezené optimální hyperparametry (`best_hyperparams`)")
        st.markdown(
            "Metoda `RandomizedSearchCV` prohledala prostor **5 148 kombinací** a v 60 náhodně vybraných iteracích "
            "identifikovala tuto optimální konfiguraci minimalizující MAE:"
        )

        col_l, col_r = st.columns([1, 1])
        with col_l:
            hp_df = pd.DataFrame([
                {"Hyperparametr": "criterion", "Nalezená hodnota": str(best_hp.get("criterion")), "Význam": "Kritérium kvality rozdělení (Poissonova regrese)"},
                {"Hyperparametr": "max_depth", "Nalezená hodnota": str(best_hp.get("max_depth")), "Význam": "Maximální hloubka větvění stromu"},
                {"Hyperparametr": "min_samples_leaf", "Nalezená hodnota": str(best_hp.get("min_samples_leaf")), "Význam": "Minimální počet vzorků v koncovém listu"},
                {"Hyperparametr": "max_features", "Nalezená hodnota": str(best_hp.get("max_features")), "Význam": "Počet náhodně vybíraných příznaků pro každý split"}
            ])
            st.dataframe(hp_df, width="stretch", hide_index=True)

        with col_r:
            st.markdown(
                f"""
                ```python
                best_hyperparams = {best_hp}
                ```
                - **5-Fold CV MAE:** `{data['cv_score_mae']:.4f} MPa`
                - **Trénovací sada (70 %):** `{dataset_info['n_train']}` vzorků
                - **Testovací sada (30 %):** `{dataset_info['n_test']}` vzorků
                - **Celkem prohledáno:** `{search_space['n_iter']}` kandidátů (300 fitů)
                """
            )

        st.markdown("---")
        st.subheader("Evaluace finálního modelu na trénovací a testovací sadě")

        met_col1, met_col2 = st.columns(2)
        with met_col1:
            st.markdown("##### 🏋️ Trénovací sada (Train)")
            tr_df = pd.DataFrame([
                {"Metrika": "Koeficient determinace (R²)", "Hodnota": f"{train_m['r2'] * 100:.2f} %"},
                {"Metrika": "Mean Squared Error (MSE)", "Hodnota": f"{train_m['mse']:.4f}"},
                {"Metrika": "Root Mean Squared Error (RMSE)", "Hodnota": f"{train_m['rmse']:.4f} MPa"},
                {"Metrika": "Mean Absolute Error (MAE)", "Hodnota": f"{train_m['mae']:.4f} MPa"},
                {"Metrika": "Mean Absolute Percentage Error (MAPE)", "Hodnota": f"{train_m['mape'] * 100:.2f} %"}
            ])
            st.dataframe(tr_df, width="stretch", hide_index=True)

        with met_col2:
            st.markdown("##### 🧪 Testovací sada (Test)")
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
            f"**Zhodnocení:** Model dosahuje na testovací sadě koeficientu determinace **$R^2 = {test_m['r2']*100:.2f} %$**, "
            f"a průměrné absolutní chyby **{test_m['mae']:.2f} MPa** (o polovinu menší než u lineární regrese)!"
        )

    # TAB 2: Architektura stromu
    with tab2:
        st.subheader("2. Vizualizace architektury natrénovaného stromu (`plot_tree`)")
        st.markdown(
            "Níže je zobrazena struktura prvních 3 pater rozhodovacího stromu natrénovaného s optimálními parametry "
            f"(`max_depth={best_hp['max_depth']}`, `criterion='{best_hp['criterion']}'`, `max_features={best_hp['max_features']}`)."
        )

        tree_img_path = Path(__file__).resolve().parent.parent / "01_Regression" / "plots" / "concrete_tree_random_structure.png"
        if tree_img_path.exists():
            st.image(str(tree_img_path), caption="Rozhodovací strom (RandomizedSearch) – vrchní 3 patra", width="stretch")

        st.markdown("##### 🔬 Analýza architektury:")
        st.markdown(
            """
            1. **Vliv `max_features=6`:** Při každém větvení strom náhodně vybírá podmnožinu 6 příznaků z celkových 8. Tento princip zabraňuje dominantním příznakům maskovat ostatní proměnné.
            2. **Rozhodovací kritérium Poisson:** Kritérium `poisson` modeluje rozdělení četností a kladných hodnot, což se pro pevnost betonu ($csMPa > 0$) ukázalo jako vysoce účinné.
            3. **Kořenový split:** Větvení začíná na klíčových komponentách – stáří betonu (`age`) a obsahu cementu (`cement`).
            """
        )

    # TAB 3: Důležitost příznaků (Plně interaktivní Plotly)
    with tab3:
        st.subheader("3. Relativní důležitost příznaků (Feature Importances)")
        st.markdown("Jak jednotlivé složky betonové směsi přispívají ke snížení chyby predikce:")

        fi_df = pd.DataFrame(fi)
        fig_fi = px.bar(
            fi_df.sort_values(by="importance", ascending=True),
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Viridis",
            labels={"importance": "Důležitost (poměr)", "feature": "Vstupní příznak"},
            text_auto=".2%"
        )
        fig_fi.update_layout(height=450, margin=dict(l=20, r=20, t=30, b=20), coloraxis_showscale=False)
        st.plotly_chart(fig_fi, width="stretch")

        fi_table = fi_df.copy()
        fi_table["Důležitost (%)"] = fi_table["importance"].apply(lambda x: f"{x * 100:.2f} %")
        fi_table = fi_table[["feature", "description", "Důležitost (%)"]].rename(columns={
            "feature": "Příznak",
            "description": "Popis složky betonu"
        })
        st.dataframe(fi_table, width="stretch", hide_index=True)

    # TAB 4: Analýza RandomizedSearch (Plně interaktivní Plotly Scatter)
    with tab4:
        st.subheader("4. Interaktivní analýza průběhu náhodného vyhledávání (RandomizedSearchCV)")
        st.markdown(
            "Vizualizace validační chyby (5-Fold CV MAE) pro jednotlivé kombinace parametrů. "
            "Pohybem myši zobrazíte konkrétní parametry každého pokusu:"
        )

        top_trials = pd.DataFrame(data.get("top_trials", []))
        if not top_trials.empty:
            fig_trials = px.scatter(
                top_trials,
                x=top_trials.index + 1,
                y="mean_mae",
                color="rank_test_score",
                color_continuous_scale="Plasma_r",
                size=[14 - r for r in top_trials["rank_test_score"]],
                labels={"x": "Index nejlepšího kandidáta", "mean_mae": "Validační MAE (MPa)", "rank_test_score": "Pořadí"},
                title="Top 10 kandidátů RandomizedSearchCV seřazených dle MAE",
                hover_data={"param_max_depth": True, "param_criterion": True, "param_min_samples_leaf": True, "param_max_features": True}
            )
            fig_trials.add_hline(y=data["cv_score_mae"], line_dash="dash", line_color="red", annotation_text=f"Nejlepší CV MAE = {data['cv_score_mae']:.2f} MPa")
            fig_trials.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_trials, width="stretch")

            st.markdown("##### 🏆 Tabulka Top 10 nejlepších kombinací parametrů v RandomizedSearch:")
            top_trials_tbl = top_trials.rename(columns={
                "rank_test_score": "Pořadí",
                "mean_mae": "Validační MAE (MPa)",
                "param_max_depth": "max_depth",
                "param_criterion": "criterion",
                "param_min_samples_leaf": "min_samples_leaf",
                "param_max_features": "max_features"
            })
            top_trials_tbl["Validační MAE (MPa)"] = top_trials_tbl["Validační MAE (MPa)"].apply(lambda x: f"{x:.4f}")
            st.dataframe(top_trials_tbl, width="stretch", hide_index=True)

    # TAB 5: Srovnání modelů (Plně interaktivní Plotly Grouped Bar)
    with tab5:
        st.subheader("5. Interaktivní srovnání modelů: OLS vs. GridSearchCV vs. RandomizedSearchCV")
        st.markdown("Porovnání dosažených výsledků napříč všemi vytvořenými regresními modely pro pevnost betonu:")

        grid_test = comp_models.get("grid_tree", {"r2": 0.8475, "rmse": 6.6206, "mae": 4.4025})

        comp_plot_data = pd.DataFrame([
            {"Model": "OLS Lineární regrese", "Metrika": "Test R² (%)", "Hodnota": ols_test["r2"] * 100},
            {"Model": "Strom (GridSearch)", "Metrika": "Test R² (%)", "Hodnota": grid_test["r2"] * 100},
            {"Model": "Strom (RandomizedSearch)", "Metrika": "Test R² (%)", "Hodnota": test_m["r2"] * 100},

            {"Model": "OLS Lineární regrese", "Metrika": "Test RMSE (MPa)", "Hodnota": ols_test["rmse"]},
            {"Model": "Strom (GridSearch)", "Metrika": "Test RMSE (MPa)", "Hodnota": grid_test["rmse"]},
            {"Model": "Strom (RandomizedSearch)", "Metrika": "Test RMSE (MPa)", "Hodnota": test_m["rmse"]},

            {"Model": "OLS Lineární regrese", "Metrika": "Test MAE (MPa)", "Hodnota": ols_test["mae"]},
            {"Model": "Strom (GridSearch)", "Metrika": "Test MAE (MPa)", "Hodnota": grid_test["mae"]},
            {"Model": "Strom (RandomizedSearch)", "Metrika": "Test MAE (MPa)", "Hodnota": test_m["mae"]}
        ])

        fig_comp_bar = px.bar(
            comp_plot_data,
            x="Metrika",
            y="Hodnota",
            color="Model",
            barmode="group",
            text_auto=".2f",
            title="Porovnání klíčových metrik (R², RMSE, MAE) mezi modely",
            color_discrete_sequence=["#636EFA", "#00CC96", "#AB63FA"]
        )
        fig_comp_bar.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_comp_bar, width="stretch")

        comp_table = pd.DataFrame([
            {
                "Model": "1. Lineární regrese (OLS)",
                "Metoda optimalizace": "Analytické řešení (normální rovnice)",
                "Hyperparametry": "Žádné",
                "Test R²": f"{ols_test['r2'] * 100:.2f} %",
                "Test RMSE (MPa)": f"{ols_test['rmse']:.4f}",
                "Test MAE (MPa)": f"{ols_test['mae']:.4f}"
            },
            {
                "Model": "2. Rozhodovací strom (Cvičení 1)",
                "Metoda optimalizace": "GridSearchCV (úplná mřížka 3 parametrů)",
                "Hyperparametry": "depth=None, leaf=1, crit='squared_error'",
                "Test R²": f"{grid_test['r2'] * 100:.2f} %",
                "Test RMSE (MPa)": f"{grid_test['rmse']:.4f}",
                "Test MAE (MPa)": f"{grid_test['mae']:.4f}"
            },
            {
                "Model": "3. Rozhodovací strom (Cvičení 2)",
                "Metoda optimalizace": "RandomizedSearchCV (60 náhodných iterací, 4 parametry)",
                "Hyperparametry": f"depth={best_hp['max_depth']}, leaf={best_hp['min_samples_leaf']}, feat={best_hp['max_features']}, crit='{best_hp['criterion']}'",
                "Test R²": f"{test_m['r2'] * 100:.2f} %",
                "Test RMSE (MPa)": f"{test_m['rmse']:.4f}",
                "Test MAE (MPa)": f"{test_m['mae']:.4f}"
            }
        ])
        st.dataframe(comp_table, width="stretch", hide_index=True)

        st.markdown(
            r"""
            #### 💡 Klíčové poznatky ze srovnání:
            1. **RandomizedSearchCV dosáhl nejnižší MAE:** Na testovací sadě dosáhl model ze Cvičení 2 průměrné absolutní chyby **4.39 MPa** (oproti 4.40 MPa u GridSearch a 8.98 MPa u OLS).
            2. **Časová úspora:** Místo testování všech 5 148 kombinací stačilo 60 vzorků (cca 1 % prostoru), které nalezly srovnatelně či více optimální model.
            3. **Výhoda `max_features`:** Omezení počtu příznaků na 6 v každém rozdělení zlepšilo robustnost stromu proti lokálnímu šumu v trénovacích datech.
            """
        )

    # TAB 6: Simulátor pevnosti (Live interactive)
    with tab6:
        st.subheader("6. 🎛️ Živý simulátor pevnosti betonu (RandomizedSearch Tree)")
        st.markdown(
            "Vyzkoušejte si interaktivní predikci optimálního modelu ze Cvičení 2 nastavením složení směsi:"
        )

        sim_rc1, sim_rc2 = st.columns(2)
        with sim_rc1:
            rc_cement = st.slider("Cement (kg/m³):", 100.0, 550.0, 300.0, 5.0, key="rnd_cement")
            rc_slag = st.slider("Vysokopecní struska (kg/m³):", 0.0, 360.0, 60.0, 5.0, key="rnd_slag")
            rc_flyash = st.slider("Popílek (kg/m³):", 0.0, 200.0, 20.0, 5.0, key="rnd_flyash")
            rc_water = st.slider("Záměsová voda (kg/m³):", 120.0, 250.0, 180.0, 2.0, key="rnd_water")
        with sim_rc2:
            rc_sp = st.slider("Superplastifikátor (kg/m³):", 0.0, 35.0, 7.0, 0.5, key="rnd_sp")
            rc_coarse = st.slider("Hrubé kamenivo (kg/m³):", 800.0, 1150.0, 960.0, 10.0, key="rnd_coarse")
            rc_fine = st.slider("Jemné kamenivo / písek (kg/m³):", 590.0, 1000.0, 770.0, 10.0, key="rnd_fine")
            rc_age = st.slider("Doba zrání (dny):", 1, 365, 28, 1, key="rnd_age")

        means = {"cement": 281.17, "slag": 73.90, "flyash": 54.19, "water": 181.57, "superplasticizer": 6.20, "coarseaggregate": 972.92, "fineaggregate": 773.58, "age": 45.66}
        stds = {"cement": 104.51, "slag": 86.28, "flyash": 63.99, "water": 21.36, "superplasticizer": 5.97, "coarseaggregate": 77.75, "fineaggregate": 80.18, "age": 63.17}

        feature_cols = [
            "cement", "slag", "flyash", "water",
            "superplasticizer", "coarseaggregate", "fineaggregate", "age"
        ]
        input_df = pd.DataFrame([[
            (rc_cement - means["cement"]) / stds["cement"],
            (rc_slag - means["slag"]) / stds["slag"],
            (rc_flyash - means["flyash"]) / stds["flyash"],
            (rc_water - means["water"]) / stds["water"],
            (rc_sp - means["superplasticizer"]) / stds["superplasticizer"],
            (rc_coarse - means["coarseaggregate"]) / stds["coarseaggregate"],
            (rc_fine - means["fineaggregate"]) / stds["fineaggregate"],
            (rc_age - means["age"]) / stds["age"]
        ]], columns=feature_cols)

        if tree_model is not None:
            pred_rnd = tree_model.predict(input_df)[0]
            st.markdown("---")
            m_col1, m_col2 = st.columns(2)
            m_col1.metric("Odhadnutá pevnost (csMPa)", f"{pred_rnd:.2f} MPa")
            m_col2.metric("Vodní součinitel w/c poměr", f"{rc_water / rc_cement:.2f}")


if __name__ == "__main__":
    render_concrete_decision_tree_random_view()
