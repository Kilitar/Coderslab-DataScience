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
p_knn_lumbar_critique = st.Page("views/02_knn_lumbar_critique.py", title="Cvičení 1: Bederní páteř – Expertní analýza", icon="🔬")
p_knn_penguins = st.Page("views/02_knn_penguins.py", title="Cvičení 2: Tučňáci (k-NN) – Výsledky & Simulátor", icon="🎯")
p_knn_penguins_critique = st.Page("views/02_knn_penguins_critique.py", title="Cvičení 2: Tučňáci – Expertní analýza", icon="🔬")
p_knn_penguins_nb = st.Page("views/02_knn_penguins_nb.py", title="Cvičení 2: Tučňáci – Notebook", icon="🐍")

# Téma 9: Metriky klasifikačních modelů
p_metrics_class_theory = st.Page("views/02_classification_metrics_theory.py", title="Teorie: Metriky klasifikace", icon="📖", url_path="classification_metrics_theory")
p_metrics_lumbar = st.Page("views/02_metrics_lumbar.py", title="Cvičení 1: Bederní páteř – Metriky & Optimalizace k", icon="📊")
p_metrics_lumbar_critique = st.Page("views/02_metrics_lumbar_critique.py", title="Cvičení 1: Bederní páteř – Expertní analýza & Náklady", icon="🔬")
p_metrics_lumbar_nb = st.Page("views/02_metrics_lumbar_nb.py", title="Cvičení 1: Bederní páteř – Notebook", icon="🐍")
p_metrics_penguins = st.Page("views/02_metrics_penguins.py", title="Cvičení 2: Tučňáci – Multiclass metriky & k-sweep", icon="🎯")
p_metrics_penguins_critique = st.Page("views/02_metrics_penguins_critique.py", title="Cvičení 2: Tučňáci – Expertní analýza & OvR ROC", icon="🔬")
p_metrics_penguins_nb = st.Page("views/02_metrics_penguins_nb.py", title="Cvičení 2: Tučňáci – Notebook", icon="🐍")

# Téma 10: Logistická regrese (Logistic Regression)
p_logreg_theory = st.Page("views/02_logistic_regression_theory.py", title="Teorie: Logistická regrese", icon="📖", url_path="logistic_regression_theory")
p_logreg_lumbar_ex1 = st.Page("views/02_logistic_regression_lumbar_exercise_1.py", title="Cvičení 1: Páteř – Výsledky zadání", icon="🎯", url_path="logistic_regression_lumbar_ex1")
p_logreg_lumbar_nb = st.Page("views/02_logistic_regression_lumbar_nb.py", title="Cvičení 1: Páteř – Notebook", icon="🐍", url_path="logistic_regression_lumbar_nb")
p_logreg_penguins_ex2 = st.Page("views/02_logistic_regression_penguins_exercise_2.py", title="Cvičení 2: Tučňáci – Výsledky", icon="🎯", url_path="logistic_regression_penguins_ex2")
p_logreg_penguins_nb = st.Page("views/02_logistic_regression_penguins_nb.py", title="Cvičení 2: Tučňáci – Notebook", icon="🐍", url_path="logistic_regression_penguins_nb")
p_logreg_sample = st.Page("views/02_logistic_regression_sample.py", title="Ukázka: Simulátor & Laboratoř", icon="🔬", url_path="logistic_regression_sample")
p_logreg_nb = st.Page("views/02_logistic_regression_nb.py", title="Ukázka: Syntetika – Notebook", icon="🐍", url_path="logistic_regression_notebook")

# Téma 11: Rozhodovací stromy (Decision Trees - Classification)
p_tree_class_theory = st.Page("views/02_decision_tree_theory.py", title="Teorie: Rozhodovací stromy", icon="📖", url_path="decision_tree_classification_theory")
p_tree_lumbar_ex1 = st.Page("views/02_decision_tree_lumbar_exercise_1.py", title="Cvičení 1: Páteř – Výsledky zadání", icon="🎯", url_path="decision_tree_lumbar_ex1")
p_tree_lumbar_nb = st.Page("views/02_decision_tree_lumbar_nb.py", title="Cvičení 1: Páteř – Notebook", icon="🐍", url_path="decision_tree_lumbar_nb")
p_tree_penguins_ex2 = st.Page("views/02_decision_tree_penguins_exercise_2.py", title="Cvičení 2: Tučňáci – Výsledky", icon="🎯", url_path="decision_tree_penguins_ex2")
p_tree_penguins_nb = st.Page("views/02_decision_tree_penguins_nb.py", title="Cvičení 2: Tučňáci – Notebook", icon="🐍", url_path="decision_tree_penguins_nb")
p_tree_class_sample = st.Page("views/02_decision_tree_sample.py", title="Ukázka: Strom & Laboratoř", icon="🔬", url_path="decision_tree_classification_sample")
p_tree_class_nb = st.Page("views/02_decision_tree_nb.py", title="Ukázka: Syntetika – Notebook", icon="🐍", url_path="decision_tree_classification_nb")

# Téma 12: Support Vector Machines (SVM)
p_svm_theory = st.Page("views/02_svm_theory.py", title="Teorie: Support Vector Machines", icon="📖", url_path="svm_theory")
p_svm_lumbar_ex1 = st.Page("views/02_svm_lumbar_exercise_1.py", title="Cvičení 1: Páteř – Výsledky zadání", icon="🎯", url_path="svm_lumbar_ex1")
p_svm_lumbar_nb = st.Page("views/02_svm_lumbar_nb.py", title="Cvičení 1: Páteř – Notebook", icon="🐍", url_path="svm_lumbar_nb")
p_svm_sample = st.Page("views/02_svm_sample.py", title="Ukázka: Klasifikační SVM", icon="🔬", url_path="svm_classification_sample")
p_svm_nb = st.Page("views/02_svm_nb.py", title="Ukázka: SVM – Notebook", icon="🐍", url_path="svm_classification_nb")

# Téma 13: Shrnutí 2. dne & Znalostní test
p_day2_summary = st.Page("views/02_day_2_summary.py", title="Shrnutí 2. dne & Kvíz", icon="🎓", url_path="day_2_summary")

# =============================================================================
# HOMEWORK: MEZI DNY 2 A 3 (TUNING & REGRESE)
# =============================================================================
# Část A: Optimalizace hyperparametrů & Křížová validace
p_hyperopt_theory = st.Page("views/03_hyperparameter_tuning_theory.py", title="Teorie: Hyperparametry & CV", icon="📖", url_path="hyperparameter_tuning_theory")
p_hyperopt_diam_ex1 = st.Page("views/03_hyperparameters_diamonds_exercise_1.py", title="Tuning 1: Diamanty – Výsledky", icon="🎯", url_path="hyperparameters_diamonds_ex1")
p_hyperopt_diam_nb = st.Page("views/03_hyperparameters_diamonds_nb.py", title="Tuning 1: Diamanty – Notebook", icon="🐍", url_path="hyperparameters_diamonds_nb")
p_hyperopt_peng_ex2 = st.Page("views/03_hyperparameters_penguins_exercise_2.py", title="Tuning 2: Tučňáci – Výsledky", icon="🎯", url_path="hyperparameters_penguins_ex2")
p_hyperopt_peng_nb = st.Page("views/03_hyperparameters_penguins_nb.py", title="Tuning 2: Tučňáci – Notebook", icon="🐍", url_path="hyperparameters_penguins_nb")
p_hyperopt_sample = st.Page("views/03_hyperparameter_optimization_sample.py", title="Tuning: Ukázka implementace", icon="🔬", url_path="hyperparameter_optimization_sample")
p_hyperopt_nb = st.Page("views/03_hyperparameter_optimization_nb.py", title="Tuning: Ukázka – Notebook", icon="🐍", url_path="hyperparameter_optimization_nb")

# Část B: Regrese – Pevnost betonu (Concrete Compressive Strength)
p_hw_concrete_prep = st.Page("views/01_homework_concrete_preprocessing.py", title="Regrese: Příprava dat (Beton)", icon="🧱", url_path="hw_concrete_prep")
p_hw_concrete_nb = st.Page("views/01_homework_concrete_preprocessing_nb.py", title="Regrese: Příprava dat – Notebook", icon="🐍", url_path="hw_concrete_nb")
p_hw_concrete_lr = st.Page("views/01_homework_concrete_linear_regression.py", title="Regrese: Lineární model (Beton)", icon="📈", url_path="hw_concrete_lr")
p_hw_concrete_lr_nb = st.Page("views/01_homework_concrete_linear_regression_nb.py", title="Regrese: Lineární model – Notebook", icon="🐍", url_path="hw_concrete_lr_nb")
p_hw_concrete_reg = st.Page("views/01_homework_concrete_regularization.py", title="Regrese: Regularizace (Beton)", icon="🎯", url_path="hw_concrete_reg")
p_hw_concrete_reg_nb = st.Page("views/01_homework_concrete_regularization_nb.py", title="Regrese: Regularizace – Notebook", icon="🐍", url_path="hw_concrete_reg_nb")
p_hw_concrete_tree = st.Page("views/01_homework_concrete_decision_tree.py", title="Regrese: Rozhodovací strom (Beton)", icon="🌳", url_path="hw_concrete_tree")
p_hw_concrete_tree_nb = st.Page("views/01_homework_concrete_decision_tree_nb.py", title="Regrese: Strom (Beton) – Notebook", icon="🐍", url_path="hw_concrete_tree_nb")
p_hw_concrete_tree_rnd = st.Page("views/01_homework_concrete_decision_tree_random.py", title="Regrese: Strom (Randomized)", icon="🎲", url_path="hw_concrete_tree_rnd")
p_hw_concrete_tree_rnd_nb = st.Page("views/01_homework_concrete_decision_tree_random_nb.py", title="Regrese: Random Strom – Notebook", icon="🐍", url_path="hw_concrete_tree_rnd_nb")

# Část C: Klasifikace – Diabetes (Příprava dat a klasifikační modely)
p_hw_diabetes_prep = st.Page("views/02_homework_diabetes_preprocessing.py", title="Klasifikace: Příprava (Diabetes)", icon="🩺", url_path="hw_diabetes_prep")
p_hw_diabetes_nb = st.Page("views/02_homework_diabetes_preprocessing_nb.py", title="Klasifikace: Příprava – Notebook", icon="🐍", url_path="hw_diabetes_nb")
p_hw_diabetes_knn = st.Page("views/02_homework_diabetes_knn.py", title="Klasifikace: k-NN (Diabetes)", icon="🎯", url_path="hw_diabetes_knn")
p_hw_diabetes_knn_nb = st.Page("views/02_homework_diabetes_knn_nb.py", title="Klasifikace: k-NN – Notebook", icon="🐍", url_path="hw_diabetes_knn_nb")
p_hw_diabetes_lr = st.Page("views/02_homework_diabetes_logistic_regression.py", title="Klasifikace: LogReg (Diabetes)", icon="🎯", url_path="hw_diabetes_lr")
p_hw_diabetes_lr_nb = st.Page("views/02_homework_diabetes_logistic_regression_nb.py", title="Klasifikace: LogReg – Notebook", icon="🐍", url_path="hw_diabetes_lr_nb")
p_hw_diabetes_svm = st.Page("views/02_homework_diabetes_svm.py", title="Klasifikace: SVM (Diabetes)", icon="🎯", url_path="hw_diabetes_svm")
p_hw_diabetes_svm_nb = st.Page("views/02_homework_diabetes_svm_nb.py", title="Klasifikace: SVM – Notebook", icon="🐍", url_path="hw_diabetes_svm_nb")
p_hw_session1_synthesis = st.Page("views/02_homework_session1_modern_synthesis.py", title="Syntéza Session 1: SOTA 2026", icon="🚀", url_path="hw_session1_synthesis")

# =============================================================================
# PREWORK SESSION 2: PŘÍPRAVA PŘED DNEM 3 (NEURONOVÉ SÍTĚ)
# =============================================================================
p_prework_nn_intro = st.Page("views/03_prework_neural_networks_intro.py", title="Prework: Neuronové sítě", icon="🧠", url_path="prework_nn_intro")

# Hierarchická navigace přehledně členěná dle tematických bloků
nav = st.navigation(
    {
        "Úvod & Přehled kurzu": [
            p_home,
        ],
        "00. Den 0: Prework (Úvod do Data Science & ML)": [
            p_prework_foundations,
            p_prework_stats,
            p_prework_workflow,
        ],
        "01. Den 1: Lineární regrese (OLS)": [
            p_theory,
            p_kc_model,
            p_kc_time,
            p_kc_nb,
            p_diamonds_model,
            p_diamonds_gemology,
            p_diamonds_nb,
        ],
        "02. Den 1: Metriky regresních modelů": [
            p_metrics_theory,
            p_metrics_kc_model,
            p_metrics_kc_critique,
            p_metrics_kc_nb,
            p_metrics_diam_model,
            p_metrics_diam_critique,
            p_metrics_diam_nb,
        ],
        "03. Den 1: Regularizace (Lasso, Ridge, Elastic Net)": [
            p_reg_theory,
            p_elastic_theory,
            p_reg_kc_model,
            p_reg_kc_critique,
            p_reg_kc_nb,
            p_reg_diam_model,
            p_reg_diam_critique,
            p_reg_diam_nb,
        ],
        "04. Den 1: Polynomiální regrese": [
            p_poly_theory,
            p_poly_diam_model,
            p_poly_diam_critique,
            p_poly_diam_nb,
        ],
        "05. Den 1: Rozhodovací stromy & Ansámbly": [
            p_tree_theory,
            p_tree_kc,
            p_tree_kc_critique,
            p_tree_kc_nb,
            p_tree_diam,
            p_tree_diam_critique,
            p_tree_diam_nb,
            p_ensemble_theory,
        ],
        "06. Den 1: Závěr Dne 1 (Velká syntéza regrese)": [
            p_day1_summary,
        ],
        "07. Den 1: Extras (Expertní laboratoř)": [
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
            p_knn_lumbar_critique,
            p_knn_lumbar_nb,
            p_knn_penguins,
            p_knn_penguins_critique,
            p_knn_penguins_nb,
        ],
        "09. Den 2: Metriky klasifikačních modelů": [
            p_metrics_class_theory,
            p_metrics_lumbar,
            p_metrics_lumbar_critique,
            p_metrics_lumbar_nb,
            p_metrics_penguins,
            p_metrics_penguins_critique,
            p_metrics_penguins_nb,
        ],
        "10. Den 2: Logistická regrese": [
            p_logreg_theory,
            p_logreg_lumbar_ex1,
            p_logreg_lumbar_nb,
            p_logreg_penguins_ex2,
            p_logreg_penguins_nb,
            p_logreg_sample,
            p_logreg_nb,
        ],
        "11. Den 2: Rozhodovací stromy (Classification)": [
            p_tree_class_theory,
            p_tree_lumbar_ex1,
            p_tree_lumbar_nb,
            p_tree_penguins_ex2,
            p_tree_penguins_nb,
            p_tree_class_sample,
            p_tree_class_nb,
        ],
        "12. Den 2: Support Vector Machines (SVM)": [
            p_svm_theory,
            p_svm_lumbar_ex1,
            p_svm_lumbar_nb,
            p_svm_sample,
            p_svm_nb,
        ],
        "13. Den 2: Shrnutí klasifikace": [
            p_day2_summary,
        ],
        "14. Homework: Mezi Dny 2 a 3": [
            p_hyperopt_theory,
            p_hyperopt_diam_ex1,
            p_hyperopt_diam_nb,
            p_hyperopt_peng_ex2,
            p_hyperopt_peng_nb,
            p_hyperopt_sample,
            p_hyperopt_nb,
            p_hw_concrete_prep,
            p_hw_concrete_nb,
            p_hw_concrete_lr,
            p_hw_concrete_lr_nb,
            p_hw_concrete_reg,
            p_hw_concrete_reg_nb,
            p_hw_concrete_tree,
            p_hw_concrete_tree_nb,
            p_hw_concrete_tree_rnd,
            p_hw_concrete_tree_rnd_nb,
            p_hw_diabetes_prep,
            p_hw_diabetes_nb,
            p_hw_diabetes_knn,
            p_hw_diabetes_knn_nb,
            p_hw_diabetes_lr,
            p_hw_diabetes_lr_nb,
            p_hw_diabetes_svm,
            p_hw_diabetes_svm_nb,
            p_hw_session1_synthesis,
        ],
        "15. Prework Session 2: Neuronové sítě": [
            p_prework_nn_intro,
        ],
    }
)

nav.run()


