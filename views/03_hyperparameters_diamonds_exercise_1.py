"""
Den 3 / Session 2: Cvičení 1 – Optimalizace hyperparametrů na cenách diamantů
===========================================================================
Cvičení navazuje na rozhodovací strom z 1. dne:
1. Načtení diamonds.csv, LabelEncoder pro cut, color, clarity, rozdělení 70/30.
2. GridSearchCV pro dvojici (max_depth, criterion).
3. Úprava funkce train_and_test_decision_tree(hyperparameters: dict).
4. Ověření zlepšení metrik (R2, RMSE, redukce přeučení).
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def load_diamonds_ex1_precomputed():
    base_dir = Path(__file__).resolve().parent.parent
    cache_path = base_dir / "03_Advanced_ML_Neural_Networks" / "data" / "diamonds_hyperparameters_exercise_1_precomputed.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def render_diamonds_hyperparameters_exercise_1_view():
    st.title("🎯 Cvičení 1: Diamanty – Výsledky zadání")
    st.markdown(
        """
        **Zadání cvičení:**  
        Načtěte data o diamantech (`diamonds.csv`), zakódujte kategorie pomocí `LabelEncoder` a rozdělte data v poměru 70/30.
        Pomocí třídy **`GridSearchCV`** nalezněte optimální dvojici hyperparametrů (`max_depth`, `criterion`).
        Upravte funkci `train_and_test_decision_tree(hyperparameters: dict)` tak, aby přijímala slovník parametrů,
        a ověřte, zda se metriky zlepšily oproti modelu z 1. dne kurzu (`max_depth=13`).
        """
    )

    data = load_diamonds_ex1_precomputed()
    if not data:
        st.error("Předpočtená data nebyla nalezena. Spusťte prosím skript `02_hyperparameters_diamonds_exercise_1.py`.")
        return

    ds = data["dataset_summary"]
    base = data["baseline_model"]
    opt = data["optimal_model"]
    comp = data["comparison_metrics"]
    sweep = pd.DataFrame(data["depth_sweep"])

    # Rychlý přehled klíčových metrik
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Počet vzorků (Train / Test)", f"{ds['train_size']:,} / {ds['test_size']:,}")
    c2.metric(
        "Optimální Test RMSE",
        f"{opt['test_rmse']:.2f} $",
        delta=f"-{comp['delta_rmse']:.2f} $ (lepší přesnost)",
    )
    c3.metric(
        "Optimální Test R²",
        f"{opt['test_r2']:.4f}",
        delta=f"+{comp['delta_r2']:.4f} vs 1. den",
    )
    c4.metric(
        "Redukce přeučení (Gap)",
        f"{comp['overfitting_gap_reduction_pct']:.1f} %",
        delta="Menší Train-Test propad",
    )

    st.markdown("---")

    tab_results, tab_sim, tab_diag = st.tabs(
        [
            "📋 Výsledky zadání & Kód",
            "🎛️ Interaktivní simulátor hloubky",
            "📈 Diagnostika chyb & Křivky přeučení",
        ]
    )

    # ---------------------------------------------------------
    # TAB 1: Výsledky zadání & Kód
    # ---------------------------------------------------------
    with tab_results:
        st.subheader("Detailní vyhodnocení kroků zadání")

        col_l, col_r = st.columns([1.2, 1])
        with col_l:
            st.markdown("#### 1. Porovnání: Model z 1. dne vs. GridSearchCV optimum")

            comp_table = [
                {
                    "Metrika / Vlastnost": "Nastavené hyperparametry",
                    "Původní model (1. den)": f"max_depth={base['max_depth']}, criterion='{base['criterion']}'",
                    "GridSearchCV Optimum (3. den)": f"max_depth={opt['hyperparameters']['max_depth']}, criterion='{opt['hyperparameters']['criterion']}'",
                },
                {
                    "Metrika / Vlastnost": "Trénovací R² (Train Score)",
                    "Původní model (1. den)": f"{base['train_r2']:.4f}",
                    "GridSearchCV Optimum (3. den)": f"{opt['train_r2']:.4f}",
                },
                {
                    "Metrika / Vlastnost": "Testovací R² (Test Score)",
                    "Původní model (1. den)": f"{base['test_r2']:.4f}",
                    "GridSearchCV Optimum (3. den)": f"{opt['test_r2']:.4f} (zlepšení)",
                },
                {
                    "Metrika / Vlastnost": "Testovací RMSE (Root Mean Squared Error)",
                    "Původní model (1. den)": f"{base['test_rmse']:.2f} $",
                    "GridSearchCV Optimum (3. den)": f"{opt['test_rmse']:.2f} $ (pokles chyb)",
                },
                {
                    "Metrika / Vlastnost": "Generalization Gap (Train R² - Test R²)",
                    "Původní model (1. den)": f"{base['train_r2'] - base['test_r2']:.4f} (přeučeno)",
                    "GridSearchCV Optimum (3. den)": f"{opt['train_r2'] - opt['test_r2']:.4f} (-54 % přeučení)",
                },
                {
                    "Metrika / Vlastnost": "Počet koncových listů",
                    "Původní model (1. den)": f"{base['n_leaves']:,} listů",
                    "GridSearchCV Optimum (3. den)": f"{opt['n_leaves']:,} listů (jednodušší)",
                },
            ]
            st.dataframe(pd.DataFrame(comp_table), width="stretch", hide_index=True)

            st.success(
                r"""
                ✅ **Závěr ověření metrik:**
                - Model nalezený pomocí `GridSearchCV` s `max_depth=10` dosahuje **nižší testovací chyby RMSE ($641.51 oproti $642.67)**.
                - Zásadním benefitem je však **redukce přeučení**: rozdíl mezi trénovacím a testovacím $R^2$ 
                  klesl z $0.0166$ na $0.0076$ (pokles o **54.2 %**). Model má poloviční počet listů a je mnohem robustnější!
                """
            )

        with col_r:
            st.markdown("#### 2. Kód upravené funkce dle zadání:")
            st.code(
                """def train_and_test_decision_tree(hyperparameters: dict):
    # Předání slovníku do konstruktoru
    reg_tree = DecisionTreeRegressor(**hyperparameters, random_state=42)
    reg_tree.fit(X_train, y_train)

    y_pred = reg_tree.predict(X_test)
    y_pred_train = reg_tree.predict(X_train)

    print(f"R2 score on train dataset: {r2_score(y_train, y_pred_train)}")
    print(f"R2 score on test dataset: {r2_score(y_test, y_pred)}")
    print(f"Mean absolute error on test dataset: {mean_absolute_error(y_test, y_pred)}")
    print(f"Mean squared error on test dataset: {np.sqrt(mean_squared_error(y_test, y_pred))}")
    
    return reg_tree

# Volání s optimálními parametry z GridSearchCV:
train_and_test_decision_tree(grid_search.best_params_)
""",
                language="python",
            )

            st.markdown("#### 3. Kód pro `GridSearchCV`:")
            st.code(
                """from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [5, 8, 10, 11, 12, 13, 14, 15, 16, 18],
    'criterion': ['squared_error', 'poisson']
}

grid_search = GridSearchCV(
    estimator=DecisionTreeRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
""",
                language="python",
            )

    # ---------------------------------------------------------
    # TAB 2: Interaktivní simulátor
    # ---------------------------------------------------------
    with tab_sim:
        st.subheader("🎛️ Interaktivní simulátor: Prozkoumejte výsledky mřížky")
        st.markdown(
            "Vyberte hloubku a kritérium a prohlédněte si, jak se mění 5-fold křížová validační chyba RMSE:"
        )

        sim_c1, sim_c2 = st.columns(2)
        with sim_c1:
            sel_depth = st.slider("Zvolte maximální hloubku (`max_depth`):", min_value=5, max_value=18, value=10, step=1)
        with sim_c2:
            sel_crit = st.selectbox("Zvolte kritérium štěpení (`criterion`):", ["squared_error", "poisson"])

        # Najít v sweep
        row = sweep[sweep["max_depth"] == sel_depth]
        if not row.empty:
            r = row.iloc[0]
            val_rmse = r[f"{sel_crit}_rmse"]
            tr_rmse = r[f"{sel_crit}_train_rmse"]

            m1, m2, m3 = st.columns(3)
            m1.metric("5-Fold Validační RMSE", f"{val_rmse:.2f} $")
            m2.metric("Trénovací RMSE", f"{tr_rmse:.2f} $")
            m3.metric(
                "Rozdíl (Validační - Trénovací)",
                f"{val_rmse - tr_rmse:.2f} $",
                delta="Riziko přeučení při velké hloubce",
            )

        # Plotly interaktivní graf
        fig_sweep = go.Figure()
        fig_sweep.add_trace(
            go.Scatter(
                x=sweep["max_depth"],
                y=sweep["squared_error_rmse"],
                mode="lines+markers",
                name="Validační RMSE (squared_error)",
                line=dict(color="#1f77b4", width=3),
            )
        )
        fig_sweep.add_trace(
            go.Scatter(
                x=sweep["max_depth"],
                y=sweep["poisson_rmse"],
                mode="lines+markers",
                name="Validační RMSE (poisson)",
                line=dict(color="#2ca02c", width=2, dash="dash"),
            )
        )
        fig_sweep.add_trace(
            go.Scatter(
                x=sweep["max_depth"],
                y=sweep["squared_error_train_rmse"],
                mode="lines",
                name="Trénovací RMSE (squared_error)",
                line=dict(color="gray", dash="dot"),
            )
        )
        fig_sweep.add_vline(x=10, line_width=2, line_dash="dash", line_color="red", annotation_text="Optimum (depth=10)")
        fig_sweep.add_vline(x=13, line_width=1.5, line_dash="dot", line_color="black", annotation_text="Původní (depth=13)")

        fig_sweep.update_layout(
            title="Průběh chyb 5-Fold Cross-Validation v závislosti na max_depth",
            xaxis_title="Maximální hloubka stromu (max_depth)",
            yaxis_title="5-Fold CV RMSE ($)",
            template="plotly_white",
            height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_sweep, width="stretch")

    # ---------------------------------------------------------
    # TAB 3: Diagnostika reziduí
    # ---------------------------------------------------------
    with tab_diag:
        st.subheader("📈 Diagnostika predikčních chyb (Rezidua)")
        st.markdown(
            """
            Proč je optimální strom lepší? Porovnání distribuce chyb $(y_{\\text{test}} - \\hat{y}_{\\text{test}})$ 
            ukazuje, že model s `max_depth=10` má užší rozptyl chyb a méně extrémních odchylek než přeučený model s hloubkou 13.
            """
        )

        st.markdown("##### 🔬 Interaktivní křivka přeučení (Overfitting U-Curve: Train vs. Test Error)")
        sweep_data = data.get("depth_sweep", [])
        if sweep_data:
            sweep_df = pd.DataFrame(sweep_data)
            fig_ucurve = go.Figure()
            fig_ucurve.add_trace(go.Scatter(
                x=sweep_df["max_depth"],
                y=sweep_df["squared_error_train_rmse"],
                mode="lines+markers",
                name="Train RMSE (Trénovací data)",
                line=dict(color="#1f77b4", width=2.5)
            ))
            fig_ucurve.add_trace(go.Scatter(
                x=sweep_df["max_depth"],
                y=sweep_df["squared_error_rmse"],
                mode="lines+markers",
                name="Validation RMSE (5-Fold CV)",
                line=dict(color="#d62728", width=2.5)
            ))
            fig_ucurve.add_vline(x=10, line_dash="dash", line_color="green", annotation_text="Optimum (depth=10, min RMSE)")
            fig_ucurve.add_vline(x=13, line_dash="dot", line_color="orange", annotation_text="Původní baseline (depth=13, přeučený)")
            fig_ucurve.update_layout(
                title="Důkaz přeučení: S rostoucí hloubkou trénovací chyba stále klesá, ale validační od hloubky 10 roste!",
                xaxis_title="Maximální hloubka stromu (max_depth)",
                yaxis_title="RMSE ($)",
                height=450,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_ucurve, width="stretch")

        st.markdown("##### 📊 Srovnání složitosti modelu: Počet koncových listů (Leaves)")
        comp_df = pd.DataFrame([
            {"Model": "Původní strom (Den 1)", "Hloubka": 13, "Počet listů": data["baseline_model"]["n_leaves"], "Test RMSE ($)": data["baseline_model"]["test_rmse"], "Train R²": data["baseline_model"]["train_r2"]},
            {"Model": "Optimalizovaný strom (GridSearchCV)", "Hloubka": 10, "Počet listů": data["optimal_model"]["n_leaves"], "Test RMSE ($)": data["optimal_model"]["test_rmse"], "Train R²": data["optimal_model"]["train_r2"]},
            {"Model": "Alternativní Poisson strom", "Hloubka": 12, "Počet listů": data["alternative_poisson_model"]["n_leaves"], "Test RMSE ($)": data["alternative_poisson_model"]["test_rmse"], "Train R²": data["alternative_poisson_model"]["train_r2"]}
        ])

        fig_leaves = px.bar(
            comp_df,
            x="Model",
            y="Počet listů",
            color="Model",
            text_auto=True,
            title="Dramatické zjednodušení modelu: Pokles počtu listů z 4 410 na 929 (-79 % složitosti!)",
            color_discrete_sequence=["#ff7f0e", "#2ca02c", "#9467bd"]
        )
        fig_leaves.update_layout(showlegend=False, height=380, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_leaves, width="stretch")

        st.dataframe(comp_df, width="stretch", hide_index=True)


if __name__ == "__main__":
    render_diamonds_hyperparameters_exercise_1_view()
