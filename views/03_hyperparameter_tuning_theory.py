"""
Den 3 / Session 2: Teorie optimalizace hyperparametrů a křížové validace
======================================================================
Interaktivní výukový modul vysvětlující:
1. Rozdíl mezi vnitřními parametry a hyperparametry modelů.
2. K-násobnou křížovou validaci (K-Fold Cross-Validation) a rizika data leakage.
3. Strategie prohledávání: Grid Search vs. Random Search vs. Bayesovská optimalizace.
4. Interaktivní kalkulátor kombinací a vizualizace efektu dimenzionality (Bergstra & Bengio).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def render_hyperparameter_tuning_theory_view():
    st.title("📖 Teorie: Optimalizace hyperparametrů & Křížová validace")
    st.markdown(
        """
        V první části kurzu jsme ladili hyperparametry ručně pomocí jednoduchých `for` cyklů.
        Tento modul otevírá **druhou část kurzu (Session 2)** a představuje systematické, 
        rigorózní metody pro nalezení optimální konfigurace modelů pomocí **křížové validace (Cross-Validation)** 
        a moderních optimalizačních strategií (**Grid Search**, **Random Search**, **Bayesian Optimization**).
        """
    )

    tab_params, tab_cv, tab_strategies, tab_calc = st.tabs(
        [
            "⚙️ Parametry vs. Hyperparametry",
            "🔄 K-Fold křížová validace",
            "🔍 Strategie: Grid vs. Random vs. Bayes",
            "🧮 Interaktivní kalkulátor & Simulátor",
        ]
    )

    # ---------------------------------------------------------
    # TAB 1: Parametry vs. Hyperparametry
    # ---------------------------------------------------------
    with tab_params:
        st.subheader("Zásadní rozdíl: Co se učí model a co určuje člověk?")

        col_left, col_right = st.columns(2)
        with col_left:
            with st.container(border=True):
                st.markdown("### 🧩 Vnitřní parametry (Model Parameters)")
                st.markdown(
                    """
                    - **Co to je:** Vnitřní proměnné modelu, které algoritmus **sám optimalizuje z trénovacích dat**.
                    - **Kdy vznikají:** Během fáze trénování (`model.fit(X, y)`).
                    - **Kdo je určuje:** Matematický optimalizátor (analytické OLS, Gradient Descent, CART algoritmus, SMO).
                    - **Jak k nim přistoupit v Scikit-learn:** Atributy končící podtržítkem (např. `coef_`, `intercept_`, `tree_`).
                    """
                )
        with col_right:
            with st.container(border=True):
                st.markdown("### 🎛️ Hyperparametry (Model Hyperparameters)")
                st.markdown(
                    """
                    - **Co to je:** Konfigurační „knoflíky“, které **nastavuje uživatel před zahájením trénování**.
                    - **Účel:** Řídí chování algoritmu, složitost modelu, kapacitu a sílu regularizace.
                    - **Kdo je určuje:** Datový vědec, Grid Search, Random Search nebo Bayesovský optimalizátor.
                    - **Jak k nim přistoupit v Scikit-learn:** Argumenty konstruktoru modelu (např. `C=1.0`, `max_depth=3`).
                    """
                )

        st.markdown("---")
        st.subheader("Přehled parametrů a hyperparametrů u modelů z kurzu")

        model_selector = st.selectbox(
            "Vyberte algoritmus pro detailní rozbor:",
            [
                "Lineární regrese (OLS)",
                "Ridge / Lasso regrese",
                "k-Nearest Neighbors (k-NN)",
                "Logistická regrese",
                "Rozhodovací strom (Decision Tree)",
                "Support Vector Machine (SVM)",
            ],
        )

        model_details = {
            "Lineární regrese (OLS)": {
                "params": "Směrnice (koeficienty) $w_1, \\dots, w_p$ a posun (intercept) $b$.",
                "hyperparams": "`fit_intercept` (True/False), `positive` (True/False).",
                "note": "Základní OLS nemá téměř žádné hyperparametry k ladění kapacity – model je rigidní.",
            },
            "Ridge / Lasso regrese": {
                "params": "Zregularizované váhy $\\mathbf{w}$ a intercept $b$.",
                "hyperparams": "`alpha` (síla L1/L2 penalizace), `fit_intercept`, `max_iter`, `tol`.",
                "note": "Alpha řídí kompromis mezi zkreslením (bias) a rozptylem (variance).",
            },
            "k-Nearest Neighbors (k-NN)": {
                "params": "Žádné explicitní váhy – model ukládá celá trénovací data do paměti (tzv. lazy learner).",
                "hyperparams": "`n_neighbors` ($k$), `weights` ('uniform', 'distance'), `metric` ('euclidean', 'manhattan'), `p`.",
                "note": "Výběr $k$ dramaticky ovlivňuje vyhlazení rozhodovací hranice.",
            },
            "Logistická regrese": {
                "params": "Váhy logistické křivky $\\mathbf{w}$ a posun $b$.",
                "hyperparams": "`C` (inverzní síla regularizace), `penalty` ('l1', 'l2', 'elasticnet'), `solver` ('lbfgs', 'saga'), `class_weight`.",
                "note": "Menší hodnota $C$ znamená silnější regularizaci (plošší křivku, jednodušší model).",
            },
            "Rozhodovací strom (Decision Tree)": {
                "params": "Struktura stromu: konkrétní štěpící příznaky, prahové hodnoty $\\theta_j$ a četnosti tříd v listech.",
                "hyperparams": "`max_depth`, `min_samples_split`, `min_samples_leaf`, `criterion` ('gini', 'entropy'), `max_features`.",
                "note": "Klíčové pro prevenci přeučení. Neomezený strom se přeučí na 100 % trénovací přesnosti.",
            },
            "Support Vector Machine (SVM)": {
                "params": "Lagrangeovy multiplikátory $\\alpha_i$, podpůrné vektory (Support Vectors) a bias $b$.",
                "hyperparams": "`C` (tolerance porušení marže), `kernel` ('linear', 'rbf', 'poly'), `gamma` (dosah RBF jádra), `degree`.",
                "note": "Kombinace $C$ a $\\gamma$ definuje rovnováhu mezi hladkostí a detailním obepnutím tříd.",
            },
        }

        m_info = model_details[model_selector]
        c1, c2 = st.columns(2)
        with c1:
            st.info(f"**Vnitřní parametry modelu:**\n\n{m_info['params']}")
        with c2:
            st.success(f"**Nastavitelné hyperparametry:**\n\n{m_info['hyperparams']}")
        st.caption(f"💡 *Poznámka k ladění:* {m_info['note']}")

    # ---------------------------------------------------------
    # TAB 2: K-Fold Cross-Validation
    # ---------------------------------------------------------
    with tab_cv:
        st.subheader("Křížová validace (K-Fold Cross-Validation)")
        st.markdown(
            r"""
            Proč nestačí pouhý jednorázový `train_test_split`?
            1. **Závislost na náhodném rozdělení:** U menších datasetů může změna `random_state` změnit metriku o 5–10 %.
            2. **Plýtvání vzorky:** Testovací část dat leží ladem a model se z ní neučí.
            3. **Únik informací (Data Leakage) při ladění:** Pokud hyperparametry optimalizujeme tak, 
               aby dosáhly maxima na testovací sadě, testovací sada se stává součástí trénovacího procesu!
            """
        )

        st.markdown("#### Jak funguje $K$-násobná křížová validace v praxi?")
        k_val = st.slider("Zvolte počet foldů ($K$):", min_value=2, max_value=8, value=4, step=1)

        # Interaktivní vizualizace foldů
        fold_rows = []
        for i in range(k_val):
            row = {"Iterace": f"Krok {i+1}"}
            for j in range(k_val):
                row[f"Fold {j+1}"] = "Validační sada" if j == i else "Trénovací sada"
            fold_rows.append(row)

        df_folds = pd.DataFrame(fold_rows)
        st.dataframe(df_folds, width="stretch", hide_index=True)

        st.markdown(
            r"""
            **Matematické vyhodnocení:**
            Každá z $K$ iterací poskytne skóre $S_i$. Výsledné validační skóre a jeho spolehlivost jsou vyjádřeny jako:
            """
        )
        st.latex(r"\mu_S = \frac{1}{K}\sum_{i=1}^K S_i, \qquad \sigma_S = \sqrt{\frac{1}{K}\sum_{i=1}^K (S_i - \mu_S)^2}")
        st.info(
            """
            📌 **Stratified K-Fold:** U klasifikace vždy používáme stratifikované foldy, 
            které zaručují identický poměr tříd v každém foldu i v celém datasetu.
            """
        )

    # ---------------------------------------------------------
    # TAB 3: Strategie: Grid vs. Random vs. Bayes
    # ---------------------------------------------------------
    with tab_strategies:
        st.subheader("Tři hlavní přístupy k vyhledávání hyperparametrů")

        st.markdown(
            """
            | Strategie | Princip | Výhody | Nevýhody |
            | :--- | :--- | :--- | :--- |
            | **Grid Search (Mřížka)** | Vyzkouší všechny kombinace ze zadaného kartézského součinu. | Systematický, zaručeně najde nejlepší bod z mřížky. | Trpí kombinatorickou explozí. Netestuje mezihodnoty. |
            | **Random Search (Náhoda)** | Náhodně vzorkuje z definovaných rozsahů/distribucí po $N$ kroků. | Mnohem efektivnější v objevování vlivných dimenzí. | Nemá záruku pokrytí konkrétního bodu. |
            | **Bayesovská optimalizace** | Staví pravděpodobnostní model závislosti metriky a chytře vybírá další bod. | Extrémně efektivní, „hledání pokladu na základě stop“. | Složitější konfigurace, sekvenční povaha (hůře se paralelizuje). |
            """
        )

        st.markdown("---")
        st.subheader("💡 Proč je Random Search efektivnější než Grid Search?")
        st.markdown(
            """
            *Podle slavného článku J. Bergstra & Y. Bengio (2012): Random Search for Hyper-Parameter Optimization.*
            Většina reálných problémů má jen **1 nebo 2 dominantní hyperparametry** (např. $C$ u SVM nebo `max_depth` u stromu), 
            zatímco ostatní mají zanedbatelný vliv.
            """
        )

        # Vizuální demonstrace Bergstra & Bengio
        np.random.seed(42)
        grid_x = np.repeat([1, 2, 3], 3)
        grid_y = np.tile([1, 2, 3], 3)

        rand_x = np.random.uniform(0.5, 3.5, 9)
        rand_y = np.random.uniform(0.5, 3.5, 9)

        fig_bb = go.Figure()
        fig_bb.add_trace(
            go.Scatter(
                x=grid_x,
                y=grid_y,
                mode="markers",
                marker=dict(size=14, color="crimson", symbol="square"),
                name="Grid Search (9 bodů = 3 unikátní hodnoty X)",
            )
        )
        fig_bb.add_trace(
            go.Scatter(
                x=rand_x,
                y=rand_y,
                mode="markers",
                marker=dict(size=14, color="royalblue", symbol="circle"),
                name="Random Search (9 bodů = 9 unikátních hodnot X!)",
            )
        )

        fig_bb.update_layout(
            title="Srovnání pokrytí prostoru: Grid Search vs. Random Search (9 pokusů)",
            xaxis_title="Důležitý hyperparametr (např. C)",
            yaxis_title="Méně důležitý hyperparametr (např. random_state)",
            width=800,
            height=450,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_bb, width="stretch")
        st.caption(
            "Při mřížce 3×3 otestujeme pouze 3 různé hodnoty důležitého parametru. "
            "Při 9 náhodných pokusech otestujeme 9 různých hodnot důležitého parametru!"
        )

    # ---------------------------------------------------------
    # TAB 4: Interaktivní kalkulátor kombinací
    # ---------------------------------------------------------
    with tab_calc:
        st.subheader("🧮 Kalkulátor výpočetní náročnosti tuningu")
        st.markdown(
            """
            Vyzkoušejte si, jak rychle roste počet modelů při **Grid Search** oproti **Random Search** 
            s rostoucím počtem hyperparametrů a foldů křížové validace:
            """
        )

        col_in1, col_in2, col_in3 = st.columns(3)
        with col_in1:
            n_params = st.slider("Počet laděných hyperparametrů:", min_value=1, max_value=8, value=3, step=1)
        with col_in2:
            n_values = st.slider("Počet hodnot na parametr (pro Grid):", min_value=2, max_value=10, value=4, step=1)
        with col_in3:
            k_folds = st.slider("Počet foldů v Cross-Validation:", min_value=2, max_value=10, value=5, step=1)

        time_per_fit_ms = st.number_input(
            "Odhadovaný čas natrénování 1 modelu (v milisekundách):",
            min_value=1,
            max_value=10000,
            value=25,
            step=5,
        )

        # Výpočty
        grid_combinations = n_values**n_params
        grid_total_fits = grid_combinations * k_folds
        grid_time_sec = (grid_total_fits * time_per_fit_ms) / 1000.0

        random_budget = st.slider("Rozpočet iterací pro Random Search:", min_value=10, max_value=200, value=50, step=10)
        random_total_fits = random_budget * k_folds
        random_time_sec = (random_total_fits * time_per_fit_ms) / 1000.0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Kombinací v Grid Search", f"{grid_combinations:,}")
        m2.metric("Tréninků pro Grid Search", f"{grid_total_fits:,}")
        m3.metric("Čas Grid Search", f"{grid_time_sec:.2f} s" if grid_time_sec < 60 else f"{grid_time_sec/60.0:.1f} min")
        m4.metric(
            "Čas Random Search",
            f"{random_time_sec:.2f} s" if random_time_sec < 60 else f"{random_time_sec/60.0:.1f} min",
            delta=f"{(1 - random_total_fits / max(1, grid_total_fits))*100:.0f} % úspora",
        )

        if grid_combinations > 10000:
            st.warning(
                f"⚠️ **Pozor na kombinatorickou explozi!** Grid Search musí natrénovat {grid_total_fits:,} modelů. "
                "V tomto scénáři je použití `GridSearchCV` nevhodné – doporučuje se `RandomizedSearchCV` nebo Bayesovská optimalizace."
            )
        else:
            st.success(
                f"✅ Tento prostor ({grid_combinations} kombinací) je pro moderní počítač bez problémů zvládnutelný i s Grid Search."
            )

        st.markdown("---")
        st.subheader("💡 3 zlatá pravidla z praxe:")
        st.markdown(
            r"""
            1. **Nikdy netunit na finální testovací sadě:** 
               Data nejprve rozdělte na `X_train` a `X_test`. Celý proces ladění a křížové validace provádějte 
               **výhradně na `X_train`**. Finální model pak jednou otestujte na `X_test`.
            2. **Pipeline proti Data Leakage:**
               Normalizace a škálování příznaků (`StandardScaler`, `Normalizer`) musí probíhat **uvnitř křížové validace** 
               pomocí `sklearn.pipeline.Pipeline`, aby nedošlo k ovlivnění trénovacího foldu daty z validačního foldu!
            3. **Logaritmická mřížka pro multiplikativní parametry:**
               Parametry jako $C, \alpha, \gamma$ se zásadně ladí v mocninách desítky: `[0.001, 0.01, 0.1, 1, 10, 100]`.
            """
        )


if __name__ == "__main__":
    render_hyperparameter_tuning_theory_view()
