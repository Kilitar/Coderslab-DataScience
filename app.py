import streamlit as st

st.set_page_config(
    page_title="Data Science & ML Portfolio | Coderslab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Definice stránek
p_home = st.Page("views/00_home.py", title="Přehled kurzu a sylabus", icon="🏠", default=True)

# Téma 0: Prework – Základy Data Science & ML
p_prework_foundations = st.Page("views/00_prework_foundations.py", title="01. Co je Data Science & ML", icon="🚀")
p_prework_stats = st.Page("views/00_prework_statistics.py", title="02. Statistika & Python stack", icon="📊")
p_prework_workflow = st.Page("views/00_prework_workflow.py", title="03. Příprava dat & Scikit-learn", icon="🛠️")

# Téma 1: Lineární regrese & OLS
p_theory = st.Page("views/01_theory.py", title="Teorie: Lineární regrese & OLS", icon="📖")
p_kc_model = st.Page("views/01_kc_housing.py", title="Cvičení 1: Reality – Výsledky zadání", icon="🎯")
p_kc_time = st.Page("views/01_kc_time_analysis.py", title="Cvičení 1: Reality – Expertní analýza", icon="🔬")
p_kc_nb = st.Page("views/01_kc_notebook.py", title="Cvičení 1: Reality – Notebook", icon="🐍")
p_diamonds_model = st.Page("views/01_diamonds.py", title="Cvičení 2: Diamanty – Výsledky zadání", icon="🎯")
p_diamonds_gemology = st.Page("views/01_diamonds_gemology_analysis.py", title="Cvičení 2: Diamanty – Expertní analýza", icon="🔬")
p_diamonds_nb = st.Page("views/01_diamonds_notebook.py", title="Cvičení 2: Diamanty – Notebook", icon="🐍")

# Téma 2: Metriky regresních modelů
p_metrics_theory = st.Page("views/01_metrics_theory.py", title="Teorie: Metriky regrese", icon="📖", url_path="regression_metrics_theory")
p_metrics_kc_model = st.Page("views/01_metrics_exercise_1.py", title="Cvičení 3: Metriky reality – Výsledky zadání", icon="🎯")
p_metrics_kc_critique = st.Page("views/01_metrics_critique_analysis.py", title="Cvičení 3: Metriky reality – Expertní analýza", icon="🔬")
p_metrics_kc_nb = st.Page("views/01_metrics_notebook.py", title="Cvičení 3: Metriky reality – Notebook", icon="🐍")
p_metrics_diam_model = st.Page("views/01_diamonds_metrics.py", title="Cvičení 4: Metriky diamantů – Výsledky zadání", icon="🎯")
p_metrics_diam_critique = st.Page("views/01_diamonds_metrics_critique.py", title="Cvičení 4: Metriky diamantů – Expertní analýza", icon="🔬")
p_metrics_diam_nb = st.Page("views/01_diamonds_metrics_nb.py", title="Cvičení 4: Metriky diamantů – Notebook", icon="🐍")

# Téma 3: Regularizace (Lasso, Ridge, Elastic Net)
p_reg_theory = st.Page("views/01_regularization_theory.py", title="Teorie: Regularizace (Lasso & Ridge)", icon="📖")
p_elastic_theory = st.Page("views/01_elastic_net_theory.py", title="Teorie: Elastic Net regrese", icon="📖")
p_reg_kc_model = st.Page("views/01_regularization_exercise_1.py", title="Cvičení 5: Regularizace reality – Výsledky zadání", icon="🎯")
p_reg_kc_critique = st.Page("views/01_regularization_critique.py", title="Cvičení 5: Regularizace reality – Expertní analýza", icon="🔬")
p_reg_kc_nb = st.Page("views/01_regularization_nb.py", title="Cvičení 5: Regularizace reality – Notebook", icon="🐍")
p_reg_diam_model = st.Page("views/01_diamonds_regularization.py", title="Cvičení 6: Regularizace diamantů – Výsledky zadání", icon="🎯")
p_reg_diam_critique = st.Page("views/01_diamonds_regularization_critique.py", title="Cvičení 6: Regularizace diamantů – Expertní analýza", icon="🔬")
p_reg_diam_nb = st.Page("views/01_diamonds_regularization_nb.py", title="Cvičení 6: Regularizace diamantů – Notebook", icon="🐍")

# Téma 4: Polynomiální regrese
p_poly_theory = st.Page("views/01_polynomial_theory.py", title="Teorie: Polynomiální regrese", icon="📖")
p_poly_diam_model = st.Page("views/01_diamonds_polynomial.py", title="Cvičení 7: Polynom diamantů – Výsledky zadání", icon="🎯")
p_poly_diam_critique = st.Page("views/01_diamonds_polynomial_critique.py", title="Cvičení 7: Polynom diamantů – Expertní analýza", icon="🔬")
p_poly_diam_nb = st.Page("views/01_diamonds_polynomial_nb.py", title="Cvičení 7: Polynom diamantů – Notebook", icon="🐍")

# Téma 5: Rozhodovací stromy & Ansámbly
p_tree_theory = st.Page("views/01_decision_tree_theory.py", title="Teorie: Rozhodovací strom v regresi", icon="📖")
p_tree_kc = st.Page("views/01_decision_tree_kc.py", title="Cvičení 8: Reality – Výsledky zadání", icon="🎯")
p_tree_kc_critique = st.Page("views/01_decision_tree_kc_critique.py", title="Cvičení 8: Reality – Expertní analýza", icon="🔬")
p_tree_kc_nb = st.Page("views/01_decision_tree_kc_nb.py", title="Cvičení 8: Reality – Notebook", icon="🐍")
p_tree_diam = st.Page("views/01_decision_tree_diam.py", title="Cvičení 9: Diamanty – Výsledky zadání", icon="🎯")
p_tree_diam_critique = st.Page("views/01_decision_tree_diam_critique.py", title="Cvičení 9: Diamanty – Expertní analýza", icon="🔬")
p_tree_diam_nb = st.Page("views/01_decision_tree_diam_nb.py", title="Cvičení 9: Diamanty – Notebook", icon="🐍")
p_ensemble_theory = st.Page("views/01_ensemble_theory.py", title="Teorie: Ansámblové metody (RF & Boosting)", icon="📖")

# Závěr Dne 1: Celkové shrnutí & Glosář
p_day1_summary = st.Page("views/01_day_1_summary.py", title="Závěrečné shrnutí Dne 1: Velká syntéza regrese & Glosář", icon="🎓")

# Day 1 Extras: Expertní rozšíření & Interaktivní nástroje
p_extra_whatif = st.Page("views/01_extras_whatif.py", title="Kalkulátor cen: What-If simulátor inference", icon="🎯")
p_extra_geo = st.Page("views/01_extras_geo_residuals.py", title="Mapa nemovitostí: Kde OLS a stromy chybují v Seattlu", icon="🗺️")
p_extra_gauss = st.Page("views/01_extras_gauss_markov.py", title="Diagnostika OLS: Gauss-Markov & VIF multikolinearita", icon="🔬")
p_extra_ccp = st.Page("views/01_extras_ccp_pruning.py", title="Prořezávání stromů: Cost-Complexity Pruning (alpha)", icon="🔬")
p_extra_chooser = st.Page("views/01_extras_model_chooser.py", title="Průvodce výběrem: Jak volit model v praxi", icon="🔬")
p_extra_cheat = st.Page("views/01_extras_cheatsheet.py", title="Tahák do kapsy: Scikit-learn Cheatsheet ke stažení", icon="📖")

# =============================================================================
# DEN 2: KLASIFIKACE (CLASSIFICATION)
# =============================================================================
# Téma 8: K-Nearest Neighbors (k-NN)
p_knn_theory = st.Page("views/02_knn_theory.py", title="Teorie: K-Nearest Neighbors (k-NN)", icon="📖")
p_knn_impl = st.Page("views/02_knn_impl_guide.py", title="Teorie: k-NN v Scikit-learn & Škálování", icon="🛠️")
p_knn_lumbar = st.Page("views/02_knn_lumbar.py", title="Cvičení 1: Bederní páteř (k-NN) – Výsledky & Diagnostika", icon="🎯")
p_knn_lumbar_nb = st.Page("views/02_knn_lumbar_nb.py", title="Cvičení 1: Bederní páteř – Notebook", icon="🐍")
p_knn_penguins = st.Page("views/02_knn_penguins.py", title="Cvičení 2: Tučňáci (k-NN) – Výsledky & Simulátor", icon="🎯")
p_knn_penguins_nb = st.Page("views/02_knn_penguins_nb.py", title="Cvičení 2: Tučňáci – Notebook", icon="🐍")

# Téma 9: Metriky klasifikačních modelů
p_metrics_class_theory = st.Page("views/02_classification_metrics_theory.py", title="Teorie: Metriky klasifikace", icon="📖", url_path="classification_metrics_theory")
p_metrics_lumbar = st.Page("views/02_metrics_lumbar.py", title="Cvičení 1: Bederní páteř – Metriky & Optimalizace k", icon="📊")
p_metrics_lumbar_nb = st.Page("views/02_metrics_lumbar_nb.py", title="Cvičení 1: Bederní páteř – Notebook", icon="🐍")
p_metrics_penguins = st.Page("views/02_metrics_penguins.py", title="Cvičení 2: Tučňáci – Multiclass metriky & k-sweep", icon="🎯")
p_metrics_penguins_nb = st.Page("views/02_metrics_penguins_nb.py", title="Cvičení 2: Tučňáci – Notebook", icon="🐍")

# Hierarchická navigace přehledně členěná dle tematických bloků
nav = st.navigation(
    {
        "Úvod & Přehled kurzu": [
            p_home,
        ],
        "00. Prework: Úvod do Data Science & ML": [
            p_prework_foundations,
            p_prework_stats,
            p_prework_workflow,
        ],
        "01. Lineární regrese (OLS)": [
            p_theory,
            p_kc_model,
            p_kc_time,
            p_kc_nb,
            p_diamonds_model,
            p_diamonds_gemology,
            p_diamonds_nb,
        ],
        "02. Metriky regresních modelů": [
            p_metrics_theory,
            p_metrics_kc_model,
            p_metrics_kc_critique,
            p_metrics_kc_nb,
            p_metrics_diam_model,
            p_metrics_diam_critique,
            p_metrics_diam_nb,
        ],
        "03. Regularizace (Lasso, Ridge, Elastic Net)": [
            p_reg_theory,
            p_elastic_theory,
            p_reg_kc_model,
            p_reg_kc_critique,
            p_reg_kc_nb,
            p_reg_diam_model,
            p_reg_diam_critique,
            p_reg_diam_nb,
        ],
        "04. Polynomiální regrese": [
            p_poly_theory,
            p_poly_diam_model,
            p_poly_diam_critique,
            p_poly_diam_nb,
        ],
        "05. Rozhodovací stromy & Ansámbly": [
            p_tree_theory,
            p_tree_kc,
            p_tree_kc_critique,
            p_tree_kc_nb,
            p_tree_diam,
            p_tree_diam_critique,
            p_tree_diam_nb,
            p_ensemble_theory,
        ],
        "06. Závěr Dne 1 (Day 1 Summary)": [
            p_day1_summary,
        ],
        "07. Day 1 Extras (Expertní laboratoř)": [
            p_extra_whatif,
            p_extra_geo,
            p_extra_gauss,
            p_extra_ccp,
            p_extra_chooser,
            p_extra_cheat,
        ],
        "08. Den 2: K-Nearest Neighbors (k-NN)": [
            p_knn_theory,
            p_knn_impl,
            p_knn_lumbar,
            p_knn_lumbar_nb,
            p_knn_penguins,
            p_knn_penguins_nb,
        ],
        "09. Den 2: Metriky klasifikačních modelů": [
            p_metrics_class_theory,
            p_metrics_lumbar,
            p_metrics_lumbar_nb,
            p_metrics_penguins,
            p_metrics_penguins_nb,
        ],
    }
)

nav.run()


