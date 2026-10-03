"""
Den 3 / Session 2: Cvičení 2 – Optimalizace hyperparametrů SVM na druzích tučňáků
================================================================================
Cvičení navazuje na klasifikační model SVM z 2. dne:
1. Načtení dat penguins_df_normalized.csv, kontrola prvních 10 řádků, split 70/30 (stratifikováno).
2. Původní model ze zadání: SVC(kernel="rbf", gamma=100, C=100, decision_function_shape="ovo").
3. RandomizedSearchCV pro 4 hyperparametry: kernel, C, gamma, degree.
4. Natrénování modelu s optimálními parametry pomocí best_params_.
5. Ověření zlepšení metrik a detailní interpretace výsledků.
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def load_penguins_ex2_precomputed():
    base_dir = Path(__file__).resolve().parent.parent
    cache_path = base_dir / "03_Advanced_ML_Neural_Networks" / "data" / "penguins_hyperparameters_exercise_2_precomputed.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def render_penguins_hyperparameters_exercise_2_view():
    st.title("🎯 Cvičení 2: Tučňáci – Výsledky zadání")
    st.markdown(
        """
        **Zadání cvičení:**  
        Načtěte normalizovaná data tučňáků (`penguins_df_normalized.csv`), zobrazte prvních 10 řádků a rozdělte data na trénovací a testovací sadu v poměru 70/30.
        Pomocí třídy **`RandomizedSearchCV`** nalezněte nejlepší čtveřici hyperparametrů: `kernel`, `C`, `gamma`, `degree`.
        Natrénujte model s optimálními parametry získanými z atributu `best_params_`, ověřte zlepšení metrik a **interpretujte dosažené výsledky**.
        """
    )

    data = load_penguins_ex2_precomputed()
    if not data:
        st.error("Předpočtená data nebyla nalezena. Spusťte prosím skript `03_hyperparameters_penguins_exercise_2.py`.")
        return

    ds = data["dataset_summary"]
    base = data["baseline_model"]
    opt = data["optimal_model"]
    comp = data["comparison"]
    trials_df = pd.DataFrame(data["sampled_trials"])

    # Metriky nahoře
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vzorky (Train / Test)", f"{ds['train_size']} / {ds['test_size']}")
    c2.metric(
        "Test Precision (Micro)",
        f"{opt['test_precision_micro']*100:.2f} %",
        delta=f"+{comp['delta_precision_micro']*100:.2f} % vs baseline",
    )
    c3.metric(
        "Test Precision (Macro)",
        f"{opt['test_precision_macro']*100:.2f} %",
        delta=f"+{comp['delta_precision_macro']*100:.2f} % vs baseline",
    )
    c4.metric(
        "Počet Support Vectors",
        f"{opt['n_support_vectors']} (z 233)",
        delta=f"-{comp['delta_support_vectors']} SVs (-{comp['support_vector_reduction_pct']:.1f} %)",
    )

    st.markdown("---")

    tab_results, tab_interp, tab_trials = st.tabs(
        [
            "📋 Výsledky zadání & Kód",
            "💡 Interpretace výsledků (Klíčová otázka zadání)",
            "📊 Rozdělení 50 pokusů & Matice záměn",
        ]
    )

    # ---------------------------------------------------------
    # TAB 1: Výsledky zadání & Kód
    # ---------------------------------------------------------
    with tab_results:
        st.subheader("1. Porovnání: Původní model ze zadání vs. RandomizedSearchCV optimum")

        col1, col2 = st.columns([1.2, 1])
        with col1:
            comp_table = [
                {
                    "Vlastnost / Metrika": "Zvolené jádro (kernel)",
                    "Původní model (Baseline)": "rbf",
                    "Optimální model (RandomizedSearch)": opt["hyperparameters"]["kernel"],
                },
                {
                    "Vlastnost / Metrika": "Parametr C (penalizace marže)",
                    "Původní model (Baseline)": "100.0 (extrémně tvrdá marže)",
                    "Optimální model (RandomizedSearch)": f"{opt['hyperparameters']['C']:.2f}",
                },
                {
                    "Vlastnost / Metrika": "Koeficient gamma",
                    "Původní model (Baseline)": "100.0 (hyperlokální vliv)",
                    "Optimální model (RandomizedSearch)": f"{opt['hyperparameters']['gamma']:.2f}",
                },
                {
                    "Vlastnost / Metrika": "Stupeň polynomu (degree)",
                    "Původní model (Baseline)": "3 (nepoužito pro rbf)",
                    "Optimální model (RandomizedSearch)": str(opt["hyperparameters"]["degree"]),
                },
                {
                    "Vlastnost / Metrika": "Test Precision (Micro / Accuracy)",
                    "Původní model (Baseline)": f"{base['test_precision_micro']*100:.2f} %",
                    "Optimální model (RandomizedSearch)": f"{opt['test_precision_micro']*100:.2f} % (+{comp['delta_precision_micro']*100:.2f} %)",
                },
                {
                    "Vlastnost / Metrika": "Test Precision (Macro)",
                    "Původní model (Baseline)": f"{base['test_precision_macro']*100:.2f} %",
                    "Optimální model (RandomizedSearch)": f"{opt['test_precision_macro']*100:.2f} % (+{comp['delta_precision_macro']*100:.2f} %)",
                },
                {
                    "Vlastnost / Metrika": "Počet podpůrných vektorů (SVs)",
                    "Původní model (Baseline)": f"{base['n_support_vectors']} z 233 ({base['n_support_vectors']/233*100:.1f} % dat!)",
                    "Optimální model (RandomizedSearch)": f"{opt['n_support_vectors']} z 233 ({opt['n_support_vectors']/233*100:.1f} % dat)",
                },
            ]
            st.dataframe(pd.DataFrame(comp_table), width="stretch", hide_index=True)

            st.success(
                r"""
                ✅ **Shrnutí zlepšení:**  
                - **Testovací Precision vzrostla na 99.01 %** (pouze 1 chybně klasifikovaný tučňák ze 101!).
                - **Počet podpůrných vektorů klesl ze 108 na 17** (pokles o **84.3 %**).
                - Model odstranil drastické přeučení a získal hladkou rozhodovací nadrovinu.
                """
            )

        with col2:
            st.markdown("#### Kód implementace dle zadání:")
            st.code(
                """from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from scipy.stats import uniform

# 1. Definice spojitých distribucí pro 4 hyperparametry
param_distributions = {
    'kernel': ['linear', 'rbf', 'poly'],
    'C': uniform(loc=0.1, scale=50.0),
    'gamma': uniform(loc=0.01, scale=5.0),
    'degree': [2, 3, 4, 5]
}

# 2. Spuštění RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=SVC(decision_function_shape="ovo", random_state=42),
    param_distributions=param_distributions,
    n_iter=50,
    cv=5,
    scoring="precision_macro",
    random_state=42,
    n_jobs=-1
)
random_search.fit(X_train, y_train)

# 3. Natrénování modelu s parametry z best_params_
best_params = random_search.best_params_
optimal_svc = SVC(**best_params, decision_function_shape="ovo", random_state=42)
optimal_svc.fit(X_train, y_train)
""",
                language="python",
            )

    # ---------------------------------------------------------
    # TAB 2: Interpretace výsledků
    # ---------------------------------------------------------
    with tab_interp:
        st.subheader("💡 Jak interpretovat dosažené výsledky?")
        st.markdown(
            """
            Otázka v zadání: *„Check if the metrics have improved. How would you interpret the results obtained?“*
            """
        )

        i_col1, i_col2 = st.columns(2)
        with i_col1:
            with st.container(border=True):
                st.markdown("### ⚠️ 1. Proč původní model selhával?")
                st.markdown(
                    r"""
                    Původní model měl parametry zadané „od boku“:
                    - **$\gamma = 100$:** U RBF jádra ($K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2)$) 
                      koeficient $\gamma$ řídí šířku Gaussovského zvonu.
                      Při $\gamma = 100$ má každý bod extrémně úzký dosah. Kolem každého vzorku vzniká izolovaný ostrůvek, 
                      místo souvislé rozhodovací oblasti.
                    - **$C = 100$:** Extrémně tvrdá penalizace za překročení marže. Model nepovolí žádnou chybu a křečovitě obepíná šum.
                    - **Důsledek:** Téměř polovina trénovacích dat (**108 vzorků ze 233, tj. 46.4 %**) se musela stát 
                      podpůrnými vektory, aby vytvořila tuto členitou hranici. Model trpěl silným **přeučením (overfittingem)**.
                    """
                )

        with i_col2:
            with st.container(border=True):
                st.markdown("### 🚀 2. Proč RandomizedSearchCV uspěl?")
                st.markdown(
                    r"""
                    RandomizedSearchCV prozkoumal 50 náhodných bodů v prostoru čtyř parametrů:
                    - **Hladká rozhodovací plocha:** Optimalizovaný model (např. polynom stupně 5 s přiměřeným $\gamma$ nebo lineární jádro) 
                      dokáže oddělit druhy tučňáků pomocí elegantní, globální geometrie.
                    - **Dramatický úbytek podpůrných vektorů:** Počet podpůrných vektorů klesl ze **108 na pouhých 17** (pokles o **84.3 %**).
                    - **Proč je to důležité:** Méně podpůrných vektorů znamená, že rozhodovací hranice je robustní, 
                      stojí pouze na skutečně kritických hraničních bodech a neřídí se náhodným šumem.
                    - **Výsledek:** Testovací Precision stoupla na **99.01 %**, přičemž ze 101 testovacích vzorků došlo k jediné chybě.
                    """
                )

        st.info(
            """
            🧠 **Závěrečný teoretický vhled:**  
            Počet podpůrných vektorů je u SVM jedním z nejspolehlivějších indikátorů složitosti a přeučení modelu. 
            Pokud počet podpůrných vektorů překročí 30–40 % velikosti trénovací sady, model téměř jistě memoruje trénovací data. 
            Díky křížové validaci v `RandomizedSearchCV` jsme nalezli model, který vyžaduje pouze 7.3 % vzorků jako podpůrné vektory.
            """
        )

    # ---------------------------------------------------------
    # TAB 3: Rozdělení pokusů & Matice záměn
    # ---------------------------------------------------------
    with tab_trials:
        st.subheader("📊 Diagnostika: Matice záměn a 50 pokusů v prostoru hyperparametrů")

        plots_dir = Path(__file__).resolve().parent.parent / "03_Advanced_ML_Neural_Networks" / "plots"
        d_c1, d_c2 = st.columns(2)
        with d_c1:
            img1 = plots_dir / "penguins_svm_cm_comparison.png"
            if img1.exists():
                st.image(str(img1), caption="Srovnání matic záměn (Původní vs. Optimalizovaný model)", width="stretch")
        with d_c2:
            img2 = plots_dir / "penguins_random_search_distribution.png"
            if img2.exists():
                st.image(str(img2), caption="Rozmístění 50 náhodných pokusů v prostoru C a gamma", width="stretch")

        st.markdown("---")
        st.subheader("Interaktivní tabulka všech 50 náhodných pokusů:")
        st.dataframe(trials_df, width="stretch", hide_index=True)


if __name__ == "__main__":
    render_penguins_hyperparameters_exercise_2_view()
