"""
Den 3: Boosting a XGBoost – Teorie, Matematika & SOTA 10/2026
============================================================
Interaktivní výukový modul pro druhou část teorie Dne 3 (Sekce Boosting):
1. Teoretický rozbor sekvenčního učení: AdaBoost (vážení chyb) vs. Gradient Boosting (minimalizace reziduí).
2. Živá matematická simulace krok za krokem podle slajdů kurzu (F0, H1, F1, H2, F2, H3, F3 na 10 pozorováních).
3. Kompletní rozbor hyperparametrů a regularizací knihovny XGBoost (gamma, alpha, lambda, colsample_bytree).
4. Melbourne Housing Benchmark: Samostatný strom vs. Random Forest vs. XGBoost.
5. Expertní kritické zhodnocení: Citlivost na odlehlé hodnoty, nutnost Early Stopping a learning rate decay.
6. Velký souboj na tabulkových datech k 10/2026: XGBoost vs. LightGBM vs. CatBoost.
7. Interaktivní vědomostní kvíz.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render_boosting_xgboost_theory_view():
    st.title("🚀 Den 3: Boosting & XGBoost – Teorie, Matematika & SOTA 2026")
    st.markdown(
        r"""
        **Druhá sekce Dne 3:** Přechod od paralelního Baggingu k sekvenčnímu **Boostingu**. 
        Jak zřetězením slabých modelů (*Weak Learners*) redukovat vychýlení (**Bias**), matematický mechanismus 
        gradientního sestupu v prostoru reziduí a proč algoritmus **XGBoost** kraluje soutěžím na Kaggle.
        """
    )

    # Horní KPI karty
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Primární cíl Boostingu", "Redukce vychýlení (Bias)", delta="Oprava předchozích chyb")
    with k2:
        st.metric("Typ trénování", "Sekvenční zřetězení", delta="Modely se učí z reziduí")
    with k3:
        st.metric("XGBoost regularizace", "L1 (Lasso) + L2 (Ridge) + Gamma", delta="Ochrana před přeučením")
    with k4:
        st.metric("Melbourne Benchmark", "260 891 AUD", delta="Vyrovnaný train & test MAE")

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🚀 Princip Boostingu",
        "📐 Matematická simulace (Slajdy)",
        "⚙️ XGBoost API & Parametry",
        "🏡 Melbourne Benchmark",
        "🔬 Expertní kritika & Úskalí",
        "⚔️ Souboj: XGB vs LGBM vs CatBoost",
        "📝 Vědomostní kvíz"
    ])

    # =========================================================================
    # TAB 1: PRINCIP BOOSTINGU
    # =========================================================================
    with tab1:
        st.subheader("1. Co je Boosting a v čem se liší od Baggingu?")
        st.markdown(
            r"""
            Zatímco **Bagging** trénuje paralelně mnoho hlubokých stromů za účelem snížení rozptylu, 
            **Boosting** staví na sekvenci **slabých modelů (*Weak Learners*)**, typicky velmi mělkých stromů:
            """
        )

        c_bag, c_boost = st.columns(2)
        with c_bag:
            st.info(
                r"""
                #### 🌲 Bagging (Paralelní)
                - **Bázové modely:** Hluboké stromy (Low Bias, High Variance).
                - **Trénování:** Nezávislé, paralelní na bootstrapových vzorcích dat.
                - **Hlavní cíl:** **Snížení rozptylu (Variance)** bez zvýšení vychýlení.
                - **Chyba jednotlivce:** Zaniká v davovém průměru.
                """
            )
        with c_boost:
            st.success(
                r"""
                #### 🚀 Boosting (Sekvenční)
                - **Bázové modely:** Mělké stromy (High Bias, Low Variance).
                - **Trénování:** Sekvenční – každý model se učí z chyb předchozího.
                - **Hlavní cíl:** **Snížení vychýlení (Bias)** při udržení nízkého rozptylu.
                - **Chyba jednotlivce:** Je předána dalšímu modelu k nápravě.
                """
            )

        st.markdown("---")
        st.subheader("Srovnání dvou hlavních strategií: AdaBoost vs. Gradient Boosting")
        c_ada, c_grad = st.columns(2)
        with c_ada:
            st.markdown("##### 1. AdaBoost (Adaptive Boosting)")
            st.markdown(
                r"""
                - **Strategie:** Vážení pozorování.
                - Všechny vzorky začínají se stejnou vahou $w_i = 1/N$.
                - Špatně klasifikovaným vzorkům se **zvýší váha**, správně zařazeným klesne.
                - Další strom se soustředí na problémové body s nejvyšší váhou.
                """
            )
        with c_grad:
            st.markdown("##### 2. Gradient Boosting")
            st.markdown(
                r"""
                - **Strategie:** Trénování přímo na reziduích (chybách).
                - První prediktor $F_0$ je pouhý průměr $\bar{y}$.
                - Každý další model $h_m(x)$ se trénuje na rozdíl $r_i = y_i - \hat{y}_i$.
                - Nová predikce: $F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$ (kde $\eta$ je learning rate).
                """
            )

    # =========================================================================
    # TAB 2: MATEMATICKÁ SIMULACE ZE SLAJDŮ
    # =========================================================================
    with tab2:
        st.subheader("2. Matematický výpočet krok za krokem (Přednáška slajdy 14–17)")
        st.markdown(
            r"""
            Přednáška uvádí názorný výpočet na 10 pozorováních. Podívejme se na přesná čísla, jak se predikce 
            $F_0 \to F_1 \to F_2 \to F_3$ postupně zpřesňuje:
            """
        )

        # Přesná data ze slajdů 14, 15, 16 a 17
        data_math = {
            "X": [5, 7, 12, 23, 25, 28, 29, 34, 35, 40],
            "Y": [82, 80, 103, 118, 172, 127, 204, 189, 99, 166],
            "F0": [134.0] * 10,
            "Y - F0": [-52.0, -54.0, -31.0, -16.0, 38.0, -7.0, 70.0, 55.0, -35.0, 32.0],
            "H1": [-38.25, -38.25, -38.25, -38.25, 25.5, 25.5, 25.5, 25.5, 25.5, 25.5],
            "F1": [95.75, 95.75, 95.75, 95.75, 159.5, 159.5, 159.5, 159.5, 159.5, 159.5],
            "Y - F1": [-13.75, -15.75, 7.25, 22.25, 12.5, -32.5, 44.5, 29.5, -60.5, 6.5],
            "H2": [6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, -27.0, -27.0],
            "F2": [102.5, 102.5, 102.5, 102.5, 166.25, 166.25, 166.25, 166.25, 132.5, 132.5],
            "Y - F2": [-20.5, -22.5, 0.5, 15.5, 5.75, -39.25, 37.75, 22.75, -33.5, 33.5],
            "H3": [-10.083, -10.083, -10.083, -10.083, -10.083, -10.083, 15.125, 15.125, 15.125, 15.125],
            "F3": [92.417, 92.417, 92.417, 92.417, 156.167, 156.167, 181.375, 181.375, 147.625, 147.625]
        }
        df_math = pd.DataFrame(data_math)
        st.dataframe(df_math, width="stretch", hide_index=True)

        st.markdown("---")
        st.markdown("#### 📈 Vizuální průběh: Jak stromy H1, H2 a H3 postupně aproximují data")

        fig_math = go.Figure()
        fig_math.add_trace(go.Scatter(
            x=df_math["X"],
            y=df_math["Y"],
            mode="markers",
            name="Skutečné body (Y)",
            marker=dict(size=12, color="#ef4444")
        ))
        fig_math.add_trace(go.Scatter(
            x=df_math["X"],
            y=df_math["F0"],
            mode="lines",
            name="F0: Průměr (134.0)",
            line=dict(color="#94a3b8", dash="dash")
        ))
        fig_math.add_trace(go.Scatter(
            x=df_math["X"],
            y=df_math["F1"],
            mode="lines+markers",
            name="F1: Po 1. stromu (split X ≤ 23)",
            line=dict(color="#3b82f6", width=2)
        ))
        fig_math.add_trace(go.Scatter(
            x=df_math["X"],
            y=df_math["F2"],
            mode="lines+markers",
            name="F2: Po 2. stromu (split X ≤ 34)",
            line=dict(color="#f59e0b", width=2)
        ))
        fig_math.add_trace(go.Scatter(
            x=df_math["X"],
            y=df_math["F3"],
            mode="lines+markers",
            name="F3: Po 3. stromu (split X ≤ 28)",
            line=dict(color="#10b981", width=3)
        ))
        fig_math.update_layout(
            title="Postupné zpřesňování modelu Gradient Boostingu (F0 → F1 → F2 → F3)",
            xaxis_title="Nezávislá proměnná X",
            yaxis_title="Hodnota Y",
            height=360,
            margin=dict(l=30, r=30, t=35, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_math, width="stretch")

        st.caption(
            "💡 **Důkaz funkčnosti:** Každý přidaný strom vykrývá zbývající rezidua a zmenšuje vzdálenost modelu od skutečných bodů."
        )

    # =========================================================================
    # TAB 3: XGBOOST API & PARAMETRY
    # =========================================================================
    with tab3:
        st.subheader("3. Kompletní přehled hyperparametrů knihovny XGBoost")
        st.markdown(
            r"""
            XGBoost podporuje Scikit-learn API třídami `xgb.XGBClassifier` a `xgb.XGBRegressor`. 
            Zde je přehled parametrů řídících architekturu a regularizaci:
            """
        )

        params_xgb = [
            ("n_estimators", "100", "Počet stromů (sekvenčních kol učení)."),
            ("learning_rate (eta)", "0.3 (v praxi 0.01–0.1)", "Krok gradientu. Nižší hodnota vyžaduje více stromů, ale brání přeučení."),
            ("max_depth", "6 (typicky 3–8)", "Maximální hloubka stromu. Nižší než u Random Forestu, protože stromy stavíme sekvenčně."),
            ("min_child_weight", "1", "Minimální součet vah v listu. Zvýšení výrazně tlumí přeučení."),
            ("gamma (min_split_loss)", "0", "Globální penalizace: štěpení proběhne jen při zisku ztrátové funkce větším než gamma."),
            ("subsample", "1.0 (v praxi 0.7–0.9)", "Podíl náhodně vybraných řádků pro každé kolo (stochastický gradient boosting)."),
            ("colsample_bytree", "1.0 (v praxi 0.6–0.8)", "Podíl náhodně vybraných sloupců pro každý strom (Feature Subspacing)."),
            ("reg_alpha", "0", "L1 regularizace vah listů (Lasso). Vynulovává zbytečné váhy a redukuje komplexitu."),
            ("reg_lambda", "1", "L2 regularizace vah listů (Ridge). Tlumí extrémní skoky v predikcích listů."),
            ("scale_pos_weight", "1", "Poměr (počet negativních / počet pozitivních vzorků) pro nevyvážené třídy."),
            ("tree_method", "'auto' / 'hist'", "Způsob konstrukce stromů. 'hist' je moderní rychlá histogramová metoda."),
        ]
        df_p_xgb = pd.DataFrame(params_xgb, columns=["Hyperparametr", "Default / Doporučení", "Význam & Popis"])
        st.dataframe(df_p_xgb, width="stretch", hide_index=True)

        st.markdown("---")
        st.markdown("#### 🛠️ Kódové syntaxe: Scikit-learn vs. Native XGBoost")
        c_code_sk, c_code_nat = st.columns(2)
        with c_code_sk:
            st.markdown("##### Scikit-learn API (`XGBRegressor`)")
            st.code(
                """
import xgboost as xgb

model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=5,
    gamma=1.0,
    reg_lambda=2.0,
    n_jobs=-1,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
                """,
                language="python"
            )
        with c_code_nat:
            st.markdown("##### Nativní XGBoost API (`xgb.train`)")
            st.code(
                """
import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'learning_rate': 0.05,
    'max_depth': 5,
    'gamma': 1.0,
    'lambda': 2.0,
    'objective': 'reg:squarederror'
}
bst = xgb.train(params, dtrain, num_boost_round=100)
y_pred = bst.predict(dtest)
                """,
                language="python"
            )

    # =========================================================================
    # TAB 4: MELBOURNE HOUSING BENCHMARK
    # =========================================================================
    with tab4:
        st.subheader("4. Benchmark z přednášky: Reality v Melbourne (`melb_house_data.csv`)")
        st.markdown(
            r"""
            V prezentaci byl natrénován `XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.2, gamma=1)`:
            """
        )

        col_melb_kpi1, col_melb_kpi2, col_melb_kpi3 = st.columns(3)
        with col_melb_kpi1:
            st.metric("XGBoost Trénovací MAE", "254 450 AUD", delta="Žádné známky přeučení")
        with col_melb_kpi2:
            st.metric("XGBoost Testovací MAE", "260 891 AUD", delta="-11 218 AUD vs Random Forest default")
        with col_melb_kpi3:
            st.metric("Rozdíl Train vs. Test", "6 441 AUD (2.5 %)", delta="Dokonalá generalizace")

        # Velké srovnání Melbourne
        df_melb_all = pd.DataFrame({
            "Model": ["Decision Tree (1 strom)", "Random Forest (100 stromů)", "XGBoost (50 stromů, depth=3)"],
            "Testovací MAE (AUD)": [376988, 272109, 260891]
        })

        fig_melb_comp = go.Figure()
        fig_melb_comp.add_trace(go.Bar(
            x=df_melb_all["Model"],
            y=df_melb_all["Testovací MAE (AUD)"],
            text=[f"{v:,} AUD".replace(",", " ") for v in df_melb_all["Testovací MAE (AUD)"]],
            textposition="auto",
            marker_color=["#ef4444", "#3b82f6", "#10b981"]
        ))
        fig_melb_comp.update_layout(
            title="Vývoj chyby ocenění nemovitostí v Melbourne (Přednáškový benchmark)",
            yaxis_title="Střední absolutní chyba MAE (AUD)",
            height=340,
            margin=dict(l=30, r=30, t=35, b=30)
        )
        st.plotly_chart(fig_melb_comp, width="stretch")

        st.caption(
            "💡 **Komentář ze slajdu 17:** Model dává téměř identické výsledky na trénovací i testovací sadě. "
            "To dokazuje stabilitu a absenci overfitingu, avšak průměrná chyba 260 tisíc AUD signalizuje prostor pro snížení vychýlení lepším feature engineeringem."
        )

    # =========================================================================
    # TAB 5: EXPERTNÍ KRITIKA & ÚSKALÍ
    # =========================================================================
    with tab5:
        st.subheader("5. Expertní kritická analýza: Úskalí a rizika Boostingu")

        c_crit1, c_crit2 = st.columns(2)
        with c_crit1:
            st.error(
                r"""
                #### ⚠️ 1. Extrémní citlivost na odlehlé hodnoty a šum
                - **Rozdíl od Random Forestu:** Random Forest odlehlé body zprůměruje $\to$ málo citlivý na šum.
                - **Problém Boostingu:** Boosting se snaží **opravit každou chybu**.
                - Pokud je v datech extrémní outlier (např. překlep v ceně 100 000 000 AUD), 
                  všechny následující stromy budou plýtvat kapacitou na vysvětlení tohoto chybného bodu.
                - **Řešení:** Robustní loss funkce (Huber loss) a důkladný preprocessing.
                """
            )
            st.warning(
                r"""
                #### 📉 2. Riziko přeučení při vysokém počtu stromů
                - U Random Forestu můžete nastavit 1 000 stromů a model se nepřeučí.
                - U Boostingu vede příliš mnoho kol k **memorování trénovacích dat**.
                - **Povinný standard v praxi:** Vždy používat **Early Stopping** s validační sadou (`early_stopping_rounds=15`).
                """
            )

        with c_crit2:
            st.warning(
                r"""
                #### ⚙️ 3. Zlaté pravidlo: Learning Rate vs. n_estimators
                - Zmenšení `learning_rate` (např. z 0.2 na 0.03) vyhladí kroky a zlepší generalizaci.
                - Vyžaduje to však **úměrně zvýšit `n_estimators`** (např. z 50 na 300).
                - Nelze změnit jeden parametr bez úpravy druhého!
                """
            )
            st.error(
                r"""
                #### 🛑 4. Neschopnost extrapolace
                - Stejně jako všechny stromové modely, ani XGBoost **nedokáže predikovat hodnoty mimo trénovací rozsah**.
                - Pokud data vykazují rostoucí trend v čase, samotný XGBoost bude v budoucnosti předpovídat vodorovnou přímku.
                """
            )

    # =========================================================================
    # TAB 6: SOUBOJ: XGBOOST VS LIGHTGBM VS CATBOOST (10/2026)
    # =========================================================================
    with tab6:
        st.subheader("6. Moderní SOTA kontext: Velký souboj tabulkových gigantů k 10/2026")
        st.markdown(
            r"""
            K říjnu 2026 tvoří absolutní špičku na tabulkových datech trojice knihoven:
            """
        )

        df_battle = pd.DataFrame({
            "Vlastnost": ["Strategie růstu stromů", "Histogramové binování", "Kategorické proměnné", "Rychlost tréninku", "Odolnost proti přeučení", "Kdy zvolit v praxi"],
            "XGBoost (Tianqi Chen)": ["Depth-wise (po patrech)", "Weighted Quantile Sketch / Hist", "One-Hot / Experimental", "Vysoká (GPU akcelerace)", "Vysoká (L1/L2 + Gamma)", "Produkční MLOps, C++ integrace"],
            "LightGBM (Microsoft)": ["Leaf-wise (podle zisku)", "GOSS (One-Side Sampling)", "Integer binning", "Extrémní (nejrychlejší)", "Střední (nutno hlídat hloubku)", "Obrovské datasety (> 100k řádků)"],
            "CatBoost (Yandex)": ["Oblivious / Symmetric Trees", "Ordered Boosting", "Nativní Target Encoding bez úniku", "Střední na CPU, rychlá na GPU", "Nejvyšší (out-of-the-box)", "Mnoho kategorií, minimum ladění"]
        })
        st.dataframe(df_battle, width="stretch", hide_index=True)

        st.success(
            r"""
            #### 💡 Doporučení pro produkční Data Science:
            - **XGBoost 2.x:** Zůstává standardem pro spolehlivost, C++ API a matematickou robustnost s explicitní L1/L2 penalizací.
            - **LightGBM:** Neporazitelný v rychlosti při rozsáhlých datasetech.
            - **CatBoost:** Vítězí na datech s vysokým podílem textových a kategorických sloupců.
            """
        )

    # =========================================================================
    # TAB 7: VĚDOMOSTNÍ KVÍZ
    # =========================================================================
    with tab7:
        st.subheader("📝 Rychlý vědomostní kvíz: Boosting & XGBoost")

        q1 = st.radio(
            "1. Na co se zaměřují rozhodovací stromy v Gradient Boostingu v každém dalším kroku učení?",
            [
                "Na náhodně vylosované vzorky s vracením (bootstrap).",
                "Na rezidua (rozdíly mezi skutečnou hodnotou a predikcí předchozích stromů).",
                "Na nově vytvořené polynomiální příznaky.",
                "Na normalizaci dat."
            ],
            key="xgb_q1"
        )

        q2 = st.radio(
            "2. Jaký je klíčový rozdíl mezi původním Gradient Boostingem a algoritmem XGBoost?",
            [
                "XGBoost funguje pouze pro textová data.",
                "XGBoost integruje do účelové funkce explicitní L1/L2 regularizaci a penalizaci složitosti stromů (gamma).",
                "XGBoost nepoužívá rozhodovací stromy.",
                "Mezi nimi není žádný rozdíl."
            ],
            key="xgb_q2"
        )

        q3 = st.radio(
            "3. Co se stane v Boostingu, pokud v datech existuje extrémní odlehlá hodnota (outlier / šum)?",
            [
                "Model ji automaticky smaže.",
                "Model ji zprůměruje jako v Random Forestu.",
                "Model jí přiřadí obrovské reziduum a následující stromy budou plýtvat kapacitou na vysvětlení tohoto šumu.",
                "Trénink okamžitě zhavaruje."
            ],
            key="xgb_q3"
        )

        q4 = st.radio(
            "4. Pokud snížíme hodnotu hyperparametru `learning_rate` (např. z 0.3 na 0.05), co musíme udělat s `n_estimators`?",
            [
                "Musíme n_estimators snížit na 1.",
                "Musíme n_estimators adekvátně zvýšit, aby model stihl konvergovat k optimu.",
                "Hodnota n_estimators na to nemá žádný vliv.",
                "Musíme vypnout regularizaci."
            ],
            key="xgb_q4"
        )

        if st.button("Vyhodnotit kvíz Boosting & XGBoost", width="stretch"):
            score = 0
            if "Na rezidua" in q1:
                score += 1
            if "explicitní L1/L2 regularizaci" in q2:
                score += 1
            if "přiřadí obrovské reziduum" in q3:
                score += 1
            if "Musíme n_estimators adekvátně zvýšit" in q4:
                score += 1

            if score == 4:
                st.balloons()
                st.success("🎉 Skvěle! 4 ze 4 správně! Dokonale rozumíte teoretickým i praktickým aspektům Boostingu a XGBoostu.")
            else:
                st.warning(f"Získali jste {score} ze 4 bodů. Projděte si záložky s teorií a matematickou simulací.")


render_boosting_xgboost_theory_view()
