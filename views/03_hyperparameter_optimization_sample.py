"""
Den 3 / Session 2: Ukázková implementace optimalizace hyperparametrů
===================================================================
Praktická demonstrace 3 stěžejních optimalizačních nástrojů:
1. GridSearchCV (Scikit-learn) - DecisionTreeClassifier
2. RandomizedSearchCV (Scikit-learn) - Support Vector Machine (SVC)
3. Bayesian Optimization (Hyperopt / TPE) - DecisionTreeClassifier
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def load_sample_precomputed():
    base_dir = Path(__file__).resolve().parent.parent
    cache_path = base_dir / "03_Advanced_ML_Neural_Networks" / "data" / "hyperparameter_optimization_sample_precomputed.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def render_hyperparameter_optimization_sample_view():
    st.title("🔬 Ukázka: Implementace optimalizace hyperparametrů")
    st.markdown(
        """
        Tato interaktivní laboratoř představuje konkrétní implementaci tří optimalizačních technik v Pythonu 
        podle výukových materiálů: **GridSearchCV** a **RandomizedSearchCV** ze Scikit-learn 
        a pokročilou **Bayesovskou optimalizaci** pomocí knihovny **Hyperopt**.
        
        *Všechny modely jsou trénovány a validovány na normalizovaném datasetu onemocnění páteře (310 pacientů, 75/25 split).*
        """
    )

    data = load_sample_precomputed()
    if not data:
        st.error("Předpočtená data nebyla nalezena. Spusťte prosím skript `01_hyperparameter_optimization_sample.py`.")
        return

    ds = data["dataset_summary"]
    grid = data["grid_search"]
    rand = data["random_search"]
    bayes = data["bayesian_search"]

    # Rychlý přehled v metrikách nahoře
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Počet vzorků (Train / Test)", f"{ds['train_size']} / {ds['test_size']}")
    c2.metric("GridSearch Time (54 combos)", f"{grid['time_seconds']:.2f} s", delta=f"{grid['test_metrics']['recall']*100:.1f} % Recall")
    c3.metric("RandomSearch Time (20 iter)", f"{rand['time_seconds']:.2f} s", delta=f"{rand['test_metrics']['precision']*100:.1f} % Precision")
    c4.metric("Hyperopt Time (35 evals)", f"{bayes['time_seconds']:.2f} s", delta=f"{bayes['best_cv_recall']*100:.1f} % Max CV")

    st.markdown("---")

    tab_grid, tab_rand, tab_bayes, tab_bench = st.tabs(
        [
            "1️⃣ GridSearchCV (Mřížka)",
            "2️⃣ RandomizedSearchCV (Náhoda)",
            "3️⃣ Bayesian Search (Hyperopt)",
            "📊 Velké srovnání & Benchmark",
        ]
    )

    # ---------------------------------------------------------
    # TAB 1: GridSearchCV
    # ---------------------------------------------------------
    with tab_grid:
        st.subheader("1. Vyčerpávající mřížka: `GridSearchCV` na DecisionTreeClassifier")
        st.markdown(
            """
            `GridSearchCV` systematicky otestuje **kartézský součin všech hodnot** zadaných ve slovníku `param_grid`.
            V medicínské diagnostice páteře je klíčové minimalizovat False Negatives, proto optimalizujeme na **`scoring='recall'`**.
            """
        )

        with st.expander("💻 Zobrazit vzorový Python kód (dle přednášky)", expanded=False):
            st.code(
                """import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# 1. Definice mřížky parametrů
params_grid = {
    'max_depth': np.arange(3, 21, 2),  # celá čísla od 3 do 19 s krokem 2
    'criterion': ["entropy", "gini"],
    'max_features': [None, "log2", "sqrt"],
}

# 2. Vytvoření základního modelu a GridSearchCV objektu
model = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(model, params_grid, cv=5, scoring="recall", n_jobs=-1)

# 3. Trénování a vyhodnocení
grid_search.fit(X_train, y_train)

# 4. Nejlepší nalezené parametry a model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
test_recall = best_model.score(X_test, y_test)
""",
                language="python",
            )

        g_col1, g_col2 = st.columns([1.2, 1])
        with g_col1:
            st.markdown("#### Nalezené optimální hyperparametry:")
            st.json(grid["best_params"])

            st.markdown("#### Výsledky na nezávislé testovací sadě ($N=78$):")
            tm = grid["test_metrics"]
            m_c1, m_c2, m_c3, m_c4 = st.columns(4)
            m_c1.metric("Recall (Záchyt)", f"{tm['recall']*100:.2f} %")
            m_c2.metric("Precision", f"{tm['precision']*100:.2f} %")
            m_c3.metric("F1-Skóre", f"{tm['f1']:.4f}")
            m_c4.metric("Accuracy", f"{tm['accuracy']*100:.2f} %")

            st.markdown("#### Top 5 kombinací z `cv_results_`:")
            top_rows = []
            for item in grid["top_5_models"]:
                p = item["params"]
                top_rows.append({
                    "Criterion": p.get("criterion"),
                    "Max Depth": p.get("max_depth"),
                    "Max Features": p.get("max_features"),
                    "Mean CV Recall": f"{item['mean_test_recall']*100:.2f} %",
                    "Std CV": f"±{item['std_test_recall']*100:.2f} %"
                })
            st.dataframe(pd.DataFrame(top_rows), width="stretch", hide_index=True)

        with g_col2:
            st.markdown("#### Matice záměn nejlepšího stromu:")
            cm_grid = np.array(grid["confusion_matrix"])
            fig_cm_g = px.imshow(
                cm_grid,
                text_auto=True,
                color_continuous_scale="Blues",
                labels=dict(x="Predikce", y="Skutečnost", color="Počet"),
                x=["Normal (0)", "Abnormal (1)"],
                y=["Normal (0)", "Abnormal (1)"],
            )
            fig_cm_g.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_cm_g, width="stretch")

            plots_dir = Path(__file__).resolve().parent.parent / "03_Advanced_ML_Neural_Networks" / "plots"
            img_heat = plots_dir / "hyperopt_grid_search_heatmap.png"
            if img_heat.exists():
                st.image(str(img_heat), caption="Heatmapa validačního Recall (max_depth vs criterion)", width="stretch")

    # ---------------------------------------------------------
    # TAB 2: RandomizedSearchCV
    # ---------------------------------------------------------
    with tab_rand:
        st.subheader("2. Náhodné prohledávání: `RandomizedSearchCV` na SVC")
        st.markdown(
            """
            `RandomizedSearchCV` netestuje všechny kombinace, ale náhodně vzorkuje ze spojitých pravděpodobnostních 
            rozdělení (zde `uniform` ze `scipy.stats`). Zde optimalizujeme **`scoring='precision'`** pro model Support Vector Machine.
            """
        )

        with st.expander("💻 Zobrazit vzorový Python kód (dle přednášky)", expanded=False):
            st.code(
                """from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

# 1. Definice spojitých distribucí parametrů
param_dist = {
    'C': uniform(loc=0.01, scale=4.0),       # rovnoměrné rozdělení v intervalu [0.01, 4.01]
    'gamma': uniform(loc=0.001, scale=0.1),  # rovnoměrné rozdělení v intervalu [0.001, 0.101]
    'kernel': ['rbf', 'linear']
}

# 2. Vytvoření modelu a RandomizedSearchCV
model = SVC(random_state=42)
random_search = RandomizedSearchCV(
    model, 
    param_distributions=param_dist, 
    n_iter=20,       # pevný počet náhodných pokusů
    cv=5, 
    scoring="precision", 
    random_state=42, 
    n_jobs=-1
)

# 3. Trénování
random_search.fit(X_train, y_train)

# 4. Nejlepší parametry
best_params = random_search.best_params_
""",
                language="python",
            )

        r_col1, r_col2 = st.columns([1.2, 1])
        with r_col1:
            st.markdown("#### Nalezené optimální hyperparametry pro SVC:")
            st.json(rand["best_params"])

            st.markdown("#### Výsledky na nezávislé testovací sadě ($N=78$):")
            rtm = rand["test_metrics"]
            rm_c1, rm_c2, rm_c3, rm_c4 = st.columns(4)
            rm_c1.metric("Precision (Přesnost)", f"{rtm['precision']*100:.2f} %")
            rm_c2.metric("Recall (Záchyt)", f"{rtm['recall']*100:.2f} %")
            rm_c3.metric("F1-Skóre", f"{rtm['f1']:.4f}")
            rm_c4.metric("Accuracy", f"{rtm['accuracy']*100:.2f} %")

            st.markdown(f"#### Vzorkování {rand['n_iter']} pokusů ze spojitých distribucí:")
            sampled_df = pd.DataFrame(rand["trials_sampled"])
            st.dataframe(sampled_df, width="stretch", hide_index=True)

        with r_col2:
            st.markdown("#### Matice záměn nejlepšího SVC:")
            cm_rand = np.array(rand["confusion_matrix"])
            fig_cm_r = px.imshow(
                cm_rand,
                text_auto=True,
                color_continuous_scale="Greens",
                labels=dict(x="Predikce", y="Skutečnost", color="Počet"),
                x=["Normal (0)", "Abnormal (1)"],
                y=["Normal (0)", "Abnormal (1)"],
            )
            fig_cm_r.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_cm_r, width="stretch")

            plots_dir = Path(__file__).resolve().parent.parent / "03_Advanced_ML_Neural_Networks" / "plots"
            img_scatter = plots_dir / "hyperopt_random_search_scatter.png"
            if img_scatter.exists():
                st.image(str(img_scatter), caption="Prostor náhodně vzorkovaných bodů (C vs gamma)", width="stretch")

    # ---------------------------------------------------------
    # TAB 3: Bayesian Optimization (Hyperopt)
    # ---------------------------------------------------------
    with tab_bayes:
        st.subheader("3. Bayesovská optimalizace: `hyperopt` (TPE) na DecisionTreeClassifier")
        st.markdown(
            """
            Knihovna `hyperopt` přistupuje k hledání hyperparametrů jako k **minimalizaci ztrátové funkce**. 
            Pomocí pravděpodobnostního modelu TPE (*Tree-structured Parzen Estimator*) se z každého předchozího pokusu 
            učí, kde se pravděpodobně nachází globální optimum.
            """
        )

        with st.expander("💻 Zobrazit vzorový Python kód (dle přednášky)", expanded=False):
            st.code(
                """from hyperopt import hp, fmin, tpe, Trials, space_eval
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# 1. Definice prostoru parametrů
space = {
    'min_samples_split': hp.uniform('min_samples_split', 5, 25),
    'min_samples_leaf': hp.choice('min_samples_leaf', [5, 10, 15, 20, 25, 30]),
    'max_depth': hp.choice('max_depth', [None, 3, 5, 7, 10]),
    'criterion': hp.choice('criterion', ['gini', 'entropy'])
}

# 2. Cílová funkce (minimalizujeme, proto vracíme zápornou hodnotu skóre!)
def objective(params):
    model = DecisionTreeClassifier(
        min_samples_split=int(params['min_samples_split']),
        min_samples_leaf=int(params['min_samples_leaf']),
        max_depth=params['max_depth'],
        criterion=params['criterion'],
        random_state=42
    )
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="recall")
    return -scores.mean()

# 3. Spuštění optimalizace s objektem Trials pro historii
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=35, trials=trials)

# 4. Převod na čitelné parametry
best_params = space_eval(space, best)
""",
                language="python",
            )

        b_col1, b_col2 = st.columns([1.2, 1])
        with b_col1:
            st.markdown("#### Nalezené optimální parametry pomocí Hyperopt:")
            st.json(bayes["best_params"])

            st.markdown("#### Výsledky na nezávislé testovací sadě ($N=78$):")
            btm = bayes["test_metrics"]
            bm_c1, bm_c2, bm_c3, bm_c4 = st.columns(4)
            bm_c1.metric("Recall (Záchyt)", f"{btm['recall']*100:.2f} %")
            bm_c2.metric("Precision", f"{btm['precision']*100:.2f} %")
            bm_c3.metric("F1-Skóre", f"{btm['f1']:.4f}")
            bm_c4.metric("Accuracy", f"{btm['accuracy']*100:.2f} %")

            st.info(
                f"""
                **Průběh učení TPE:**
                Nejvyšší dosažený 5-fold CV Recall: **{bayes['best_cv_recall']*100:.2f} %**.  
                Algoritmus prozkoumal {bayes['max_evals']} kombinací za pouhých **{bayes['time_seconds']:.2f} s**.
                """
            )

        with b_col2:
            st.markdown("#### Křivka konvergence v čase:")
            history = bayes["convergence_history"]
            cum_best = bayes["cumulative_best"]
            iters = list(range(1, len(history) + 1))

            fig_conv = go.Figure()
            fig_conv.add_trace(
                go.Scatter(
                    x=iters,
                    y=history,
                    mode="lines+markers",
                    name="Iterační Recall",
                    line=dict(color="lightgray", dash="dot"),
                    marker=dict(size=6),
                )
            )
            fig_conv.add_trace(
                go.Scatter(
                    x=iters,
                    y=cum_best,
                    mode="lines+markers",
                    name="Kumulativní maximum",
                    line=dict(color="#d95f02", width=3),
                    marker=dict(size=8, symbol="square"),
                )
            )
            fig_conv.update_layout(
                height=320,
                xaxis_title="Iterace pokusu (Trial index)",
                yaxis_title="5-Fold CV Recall",
                template="plotly_white",
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_conv, width="stretch")

    # ---------------------------------------------------------
    # TAB 4: Benchmark & Porovnání
    # ---------------------------------------------------------
    with tab_bench:
        st.subheader("📊 Souhrnný benchmark a srovnání metod")

        bench_data = [
            {
                "Metoda": "GridSearchCV",
                "Knihovna": "scikit-learn",
                "Model": "DecisionTreeClassifier",
                "Počet pokusů": f"{grid['total_combinations']} kombinací",
                "Čas běhu": f"{grid['time_seconds']:.2f} s",
                "Cílová metrika": "Recall",
                "Nejlepší CV skóre": f"{grid['best_cv_recall']*100:.2f} %",
                "Testovací skóre": f"{grid['test_metrics']['recall']*100:.2f} %",
                "Doporučení": "Vhodné pro 1-2 parametry s jasnými hranicemi",
            },
            {
                "Metoda": "RandomizedSearchCV",
                "Knihovna": "scikit-learn",
                "Model": "SVC",
                "Počet pokusů": f"{rand['n_iter']} vzorků",
                "Čas běhu": f"{rand['time_seconds']:.2f} s",
                "Cílová metrika": "Precision",
                "Nejlepší CV skóre": f"{rand['best_cv_precision']*100:.2f} %",
                "Testovací skóre": f"{rand['test_metrics']['precision']*100:.2f} %",
                "Doporučení": "Rychlý průzkum pro spojitá rozdělení (C, gamma)",
            },
            {
                "Metoda": "Hyperopt (TPE)",
                "Knihovna": "hyperopt",
                "Model": "DecisionTreeClassifier",
                "Počet pokusů": f"{bayes['max_evals']} iterací",
                "Čas běhu": f"{bayes['time_seconds']:.2f} s",
                "Cílová metrika": "Recall",
                "Nejlepší CV skóre": f"{bayes['best_cv_recall']*100:.2f} %",
                "Testovací skóre": f"{bayes['test_metrics']['recall']*100:.2f} %",
                "Doporučení": "Inteligentní volba pro drahé ansámbly a neuronové sítě",
            },
        ]
        st.dataframe(pd.DataFrame(bench_data), width="stretch", hide_index=True)

        st.markdown("---")
        st.success(
            r"""
            💡 **Závěrečné doporučení pro nadcházející cvičení:**
            - Pokud znáte malý rozsah možností (např. 2-3 hloubky a 2 kritéria): použijte **`GridSearchCV`**.
            - Pokud ladíte reálná čísla v řádech velikostí (např. $C \in [0.01, 100]$): použijte **`RandomizedSearchCV`** se spojitou distribucí ze `scipy.stats`.
            - Pokud stavíte komplexní pipeline s mnoha navzájem závislými parametry: použijte **`hyperopt`** nebo **`optuna`**.
            """
        )


if __name__ == "__main__":
    render_hyperparameter_optimization_sample_view()
