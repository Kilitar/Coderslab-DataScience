"""
Den 3: XGBoost – Cvičení 2 (Regrese cen diamantů) – Expertní analýza & SOTA 10/2026
=================================================================================
Kritické zhodnocení školního zadání a pokročilé techniky:
1. Rozbor chyby v zadání: Parametr min_samples_leaf v XGBoost neexistuje (ekvivalent je min_child_weight).
2. SOTA ladění: S colsample_bytree, learning_rate a min_child_weight stlačíme chybu na 264.79 USD!
3. Price Bracket analýza chyb: Proč u diamantů nad 10 000 USD chyba dramaticky narůstá (heteroskedasticita).
4. Velká regresní syntéza: Cesta od OLS (740 USD) přes Polynom a Stromy až k XGBoostu (264 USD).
5. Produkční doporučení: Log-transformace cíle targetu a Early Stopping.
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


def render_xgboost_diamonds_exercise_2_critique_view():
    st.title("🔬 Expertní analýza: XGBoost na diamantech & SOTA 2026")
    st.markdown(
        r"""
        V této sekci provádíme hloubkový rozbor cvičení: odhalujeme **chybu v zadání parametrů kurzu**,
        demonstrujeme **nativní ladění XGBoostu** stlačující chybu na rekordních **264.79 USD** a analyzujeme 
        cenová pásma diamantů.
        """
    )

    data = load_diamonds_xgb_data()
    if not data:
        st.error("Předpočtená data nebyla nalezena. Spusťte prosím skript `03_Advanced_ML_Neural_Networks/07_xgboost_diamonds_exercise_2.py`.")
        return

    school = data["school_model"]
    sota = data["sota_model"]
    baselines = data["baselines"]
    brackets = data["bracket_analysis"]

    # Horní KPI karty
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(
            label="SOTA XGBoost MAE",
            value=f"{sota['test_mae']:.2f} USD",
            delta=f"-{school['test_mae'] - sota['test_mae']:.2f} USD vs. Školní",
            delta_color="normal"
        )
    with k2:
        st.metric(
            label="Zlepšení proti Decision Tree",
            value=f"-{baselines['dt']['mae'] - sota['test_mae']:.2f} USD",
            delta="Chyba klesla o 26.1 %"
        )
    with k3:
        st.metric(
            label="Vítězný model kurzu",
            value="XGBoost (SOTA)",
            delta="Překonal i Random Forest (268 USD)"
        )
    with k4:
        st.metric(
            label="Koeficient R² (SOTA)",
            value=f"{sota['test_r2']*100:.2f} %",
            delta="Vysvětleno 98.12 % rozptylu"
        )

    st.markdown("---")

    t1, t2, t3, t4 = st.tabs([
        "🐛 Rozbor chyby v zadání (min_samples_leaf)",
        "🚀 SOTA Nativní ladění (264.79 USD)",
        "💎 Analýza chyb podle cenových pásem",
        "🏆 Celokurzovní srovnávací žebříček"
    ])

    # =========================================================================
    # TAB 1: ROZBOR CHYBY V ZADÁNÍ
    # =========================================================================
    with t1:
        st.subheader("1. Odhalení pedagogické chyby v zadání kurzu")
        st.error(
            r"""
            #### ⚠️ Parametr `min_samples_leaf` v knihovně `xgboost` NEEXISTUJE!
            
            V zadání kurzu stojí:
            > *"Assign a dictionary to the params variable with the hyperparameter names as keys: max_depth, min_samples_leaf and n_estimators."*
            
            Tvůrce kurzu zkopíroval text z předchozího cvičení na `RandomForestRegressor`, kde je `min_samples_leaf` klíčovým parametrem 
            Scikit-learn stromů. Jenže `xgb.XGBRegressor` tento parametr ve svém C++ jádru **vůbec nepoužívá**!
            """
        )

        c_w1, c_w2 = st.columns(2)
        with c_w1:
            st.code(
                r'''
# Výpis varování z C++ jádra XGBoost při běhu kódu:
D:\...\xgboost\training.py:200: UserWarning: 
WARNING: learner.cc:794: 
Parameters: { "min_samples_leaf" } are not used.
                ''',
                language="text"
            )
            st.caption("XGBoost parametr nehlásí jako fatální chybu (díky `**kwargs`), ale zcela jej ignoruje!")
        with c_w2:
            st.markdown(
                r"""
                #### Co je skutečným ekvivalentem v XGBoost?
                V gradientním boostingu se počet instancí v listu nereguluje prostým počtem vzorků, ale **součtem druhých derivací ztrátové funkce (Hessiánů)**:
                
                $$\sum_{i \in I_j} h_i \ge \text{min\_child\_weight}$$
                
                Pro kvadratickou ztrátu v regresi odpovídá Hessián přímo počtu pozorování. Správným parametrem v XGBoost je tedy **`min_child_weight`**!
                """
            )

    # =========================================================================
    # TAB 2: SOTA NATIVNÍ LADĚNÍ
    # =========================================================================
    with t2:
        st.subheader("2. Jak správně naladit XGBoost na diamantech (SOTA 2026)")
        st.markdown(
            r"""
            Pokud nahradíme ignorovaný parametr `min_samples_leaf` správným parametrem **`min_child_weight`**, 
            přidáme subsampling sloupců **`colsample_bytree`** a rychlost učení **`learning_rate`**, 
            model dosáhne podstatně lepších výsledků:
            """
        )

        c_s1, c_s2 = st.columns(2)
        with c_s1:
            st.info(
                f"""
                #### 🎓 Školní model (Zadání):
                - **Parametry:** {school['best_params']}
                - **Testovací MAE:** **{school['test_mae']:.2f} USD**
                - **Testovací RMSE:** **{school['test_rmse']:.2f} USD**
                - **R² skóre:** **{school['test_r2']*100:.2f} %**
                """
            )
        with c_s2:
            st.success(
                f"""
                #### 🏆 SOTA Model (Opravené parametry):
                - **Parametry:** {sota['best_params']}
                - **Testovací MAE:** **{sota['test_mae']:.2f} USD**
                - **Testovací RMSE:** **{sota['test_rmse']:.2f} USD**
                - **R² skóre:** **{sota['test_r2']*100:.2f} %**
                - **Rozdíl:** Model je o **{school['test_mae'] - sota['test_mae']:.2f} USD přesnější** na každém diamantu!
                """
            )

        st.markdown("---")
        st.subheader("Srovnání důležitosti příznaků (Feature Importances)")
        feat_cmp = pd.DataFrame({
            "Příznak": list(sota["feature_importances"].keys()),
            "Školní XGBoost": list(school["feature_importances"].values()),
            "SOTA XGBoost": list(sota["feature_importances"].values())
        }).sort_values(by="SOTA XGBoost", ascending=True)

        fig_fi_cmp = go.Figure()
        fig_fi_cmp.add_trace(go.Bar(y=feat_cmp["Příznak"], x=feat_cmp["Školní XGBoost"], name="Školní model", orientation="h", marker_color="#94a3b8"))
        fig_fi_cmp.add_trace(go.Bar(y=feat_cmp["Příznak"], x=feat_cmp["SOTA XGBoost"], name="SOTA model", orientation="h", marker_color="#3b82f6"))
        fig_fi_cmp.update_layout(
            barmode="group",
            title="Srovnání důležitosti příznaků (Gain)",
            xaxis_title="Relativní důležitost",
            height=420,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_fi_cmp)

    # =========================================================================
    # TAB 3: PRICE BRACKETS
    # =========================================================================
    with t3:
        st.subheader("3. Analýza chyb v různých cenových hladinách diamantů")
        st.markdown(
            r"""
            Globální metrika MAE = 264 USD je průměrem přes všech 16 171 testovacích diamantů. 
            Jak se však chyba chová u **levných** vs. **luxusních** diamantů?
            """
        )

        b_df = pd.DataFrame(brackets)
        st.dataframe(
            b_df.rename(
                columns={
                    "bracket": "Cenové pásmo",
                    "count": "Počet diamantů v testu",
                    "mae_dt": "MAE Strom (USD)",
                    "mae_rf": "MAE Random Forest (USD)",
                    "mae_school": "MAE Školní XGBoost (USD)",
                    "mae_sota": "MAE SOTA XGBoost (USD)"
                }
            )
        )

        fig_br = px.bar(
            b_df,
            x="bracket",
            y=["mae_dt", "mae_rf", "mae_sota"],
            barmode="group",
            title="Průměrná absolutní chyba (MAE) v závislosti na cenové kategorii diamantu",
            labels={"value": "MAE v USD", "bracket": "Cenové pásmo", "variable": "Model"},
            color_discrete_map={"mae_dt": "#ef4444", "mae_rf": "#3b82f6", "mae_sota": "#10b981"}
        )
        fig_br.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_br)

        st.info(
            r"""
            #### 💡 Klíčový data science poznatek:
            1. **Levné diamanty (do 1 000 USD):** XGBoost předpovídá cenu s fantastickou přesností – průměrná chyba je **pouze ~63 USD**.
            2. **Luxusní diamanty (nad 10 000 USD):** Chyba skáče na **~1 090 USD**. 
               * *Důvod:* **Heteroskedasticita.** Velké diamanty (nad 2 karáty) jsou na trhu extrémně vzácné a jejich cena roste exponenciálně s prémiovostí čistoty a barvy.
            3. **Produkční řešení:** Pro eliminaci této heteroskedasticity se v praxi modeluje logaritmus ceny: $y_{\text{train}} = \log(\text{price})$.
            """
        )

    # =========================================================================
    # TAB 4: VELKÉ REGRESNÍ SROVNÁNÍ
    # =========================================================================
    with t4:
        st.subheader("4. Velký celokurzovní srovnávací žebříček na datasetu diamantů")
        st.markdown(
            r"""
            Dataset diamantů nás provází od Dne 1. Zde je přehled evoluce přesnosti modelů v průběhu celého kurzu:
            """
        )

        all_models = pd.DataFrame([
            {"Fáze kurzu": "Den 1: Lineární regrese (OLS)", "Model": "Vícerozměrná lineární regrese", "MAE (USD)": "~ 740 USD", "R²": "0.919", "Poznámka": "Lineární předpoklady nezachytí nelinearitu karátu"},
            {"Fáze kurzu": "Den 1: Polynomiální regrese", "Model": "Polynom 2. stupně", "MAE (USD)": "~ 520 USD", "R²": "0.948", "Poznámka": "Zlepšení, ale exploze počtu příznaků"},
            {"Fáze kurzu": "Den 1 / Den 2: Stromy", "Model": "Samostatný Decision Tree", "MAE (USD)": f"{baselines['dt']['mae']:.2f} USD", "R²": f"{baselines['dt']['r2']:.4f}", "Poznámka": "Zachytí nelinearity, ale vysoký rozptyl"},
            {"Fáze kurzu": "Den 3: Bagging", "Model": "Random Forest (100 stromů)", "MAE (USD)": f"{baselines['rf']['mae']:.2f} USD", "R²": f"{baselines['rf']['r2']:.4f}", "Poznámka": "Dramatické snížení rozptylu průměrováním"},
            {"Fáze kurzu": "Den 3: Boosting (Zadání)", "Model": "XGBoost (Školní parametry)", "MAE (USD)": f"{school['test_mae']:.2f} USD", "R²": f"{school['test_r2']:.4f}", "Poznámka": "Ignorován parametr min_samples_leaf"},
            {"Fáze kurzu": "Den 3: Boosting (SOTA 2026)", "Model": "XGBoost (Nativní ladění)", "MAE (USD)": f"{sota['test_mae']:.2f} USD", "R²": f"{sota['test_r2']:.4f}", "Poznámka": "🏆 Absolutní vítěz celého kurzu!"}
        ])
        st.table(all_models)


if __name__ == "__main__":
    render_xgboost_diamonds_exercise_2_critique_view()
