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
# PREWORK SESSION 2: PŘÍPRAVA NA DNY 3 A 4 (SÍTĚ & NLP)
# =============================================================================
p_prework_nn_intro = st.Page("views/03_prework_neural_networks_intro.py", title="Prework: Neuronové sítě", icon="🧠", url_path="prework_nn_intro")
p_prework_nlp_intro = st.Page("views/04_prework_nlp_intro.py", title="Prework: Úvod do NLP", icon="🗣️", url_path="prework_nlp_intro")
p_prework_nltk_spacy = st.Page("views/04_prework_nltk_spacy.py", title="Prework: NLTK a spaCy", icon="🛠️", url_path="prework_nltk_spacy")

# =============================================================================
# DEN 3: ADVANCED MACHINE LEARNING MODELS
# Sekce: 1. Bagging, 2. Boosting, 3. Neural Networks
# =============================================================================
p_ensemble_bagging_theory = st.Page("views/03_ensemble_bagging_theory.py", title="Teorie: Ansámbly & Bagging", icon="📖", url_path="ensemble_bagging_theory")
p_rf_theory = st.Page("views/03_random_forest_theory_and_implementation.py", title="Teorie: Náhodný les (RF)", icon="📖", url_path="random_forest_theory")
p_rf_heart = st.Page("views/03_random_forest_heart_exercise_1.py", title="Cvičení 1: Nemoc srdce – Výsledky zadání", icon="🎯", url_path="rf_heart_exercise_1")
p_rf_heart_critique = st.Page("views/03_random_forest_heart_exercise_1_critique.py", title="Cvičení 1: Nemoc srdce – Expertní analýza", icon="🔬", url_path="rf_heart_critique")
p_rf_heart_nb = st.Page("views/03_random_forest_heart_nb.py", title="Cvičení 1: Nemoc srdce – Notebook", icon="🐍", url_path="rf_heart_notebook")
p_rf_diam = st.Page("views/03_random_forest_diamonds_exercise_2.py", title="Cvičení 2: Diamanty – Výsledky zadání", icon="🎯", url_path="rf_diamonds_exercise_2")
p_rf_diam_critique = st.Page("views/03_random_forest_diamonds_exercise_2_critique.py", title="Cvičení 2: Diamanty – Expertní analýza", icon="🔬", url_path="rf_diamonds_critique")
p_rf_diam_nb = st.Page("views/03_random_forest_diamonds_nb.py", title="Cvičení 2: Diamanty – Notebook", icon="🐍", url_path="rf_diamonds_notebook")
p_boosting_theory = st.Page("views/03_boosting_xgboost_theory.py", title="Teorie: Boosting & XGBoost", icon="📖", url_path="boosting_xgboost_theory")
p_xgb_heart = st.Page("views/03_xgboost_heart_exercise_1.py", title="Cvičení 1: Nemoc srdce – Výsledky zadání", icon="🎯", url_path="xgb_heart_exercise_1")
p_xgb_heart_critique = st.Page("views/03_xgboost_heart_exercise_1_critique.py", title="Cvičení 1: Nemoc srdce – Expertní analýza", icon="🔬", url_path="xgb_heart_critique")
p_xgb_heart_nb = st.Page("views/03_xgboost_heart_nb.py", title="Cvičení 1: Nemoc srdce – Notebook", icon="🐍", url_path="xgb_heart_notebook")
p_xgb_diam = st.Page("views/03_xgboost_diamonds_exercise_2.py", title="Cvičení 2: Diamanty – Výsledky zadání", icon="🎯", url_path="xgb_diamonds_exercise_2")
p_xgb_diam_critique = st.Page("views/03_xgboost_diamonds_exercise_2_critique.py", title="Cvičení 2: Diamanty – Expertní analýza", icon="🔬", url_path="xgb_diamonds_critique")
p_xgb_diam_nb = st.Page("views/03_xgboost_diamonds_nb.py", title="Cvičení 2: Diamanty – Notebook", icon="🐍", url_path="xgb_diamonds_notebook")
p_nn_theory = st.Page("views/03_neural_networks_theory.py", title="Teorie: Neuronové sítě & Keras", icon="📖", url_path="neural_networks_theory")
p_nn_mnist = st.Page("views/03_neural_network_mnist_exercise_1.py", title="Cvičení 1: MNIST Číslice – Výsledky zadání", icon="🎯", url_path="nn_mnist_exercise_1")
p_nn_mnist_critique = st.Page("views/03_neural_network_mnist_exercise_1_critique.py", title="Cvičení 1: MNIST Číslice – Expertní analýza", icon="🔬", url_path="nn_mnist_critique")
p_nn_mnist_nb = st.Page("views/03_neural_network_mnist_nb.py", title="Cvičení 1: MNIST Číslice – Notebook", icon="🐍", url_path="nn_mnist_notebook")
p_nn_kc = st.Page("views/03_neural_network_kc_housing_exercise_2.py", title="Cvičení 2: KC Housing – Výsledky zadání", icon="🎯", url_path="nn_kc_exercise_2")
p_nn_kc_critique = st.Page("views/03_neural_network_kc_housing_exercise_2_critique.py", title="Cvičení 2: KC Housing – Expertní analýza", icon="🔬", url_path="nn_kc_critique")
p_nn_kc_nb = st.Page("views/03_neural_network_kc_housing_nb.py", title="Cvičení 2: KC Housing – Notebook", icon="🐍", url_path="nn_kc_notebook")
p_day3_summary = st.Page("views/03_day_3_summary.py", title="Závěr Dne 3: Shrnutí & Kvíz", icon="🎓", url_path="day_3_summary")

# Den 3 Extras: Expertní laboratoř & Interaktivní nástroje
p_extra_mnist_draw = st.Page("views/03_extras_mnist_draw.py", title="MNIST: Živé kreslení & Rozpoznávání číslic", icon="✍️", url_path="day3_extras_mnist_draw")
p_extra_nn_playground = st.Page("views/03_extras_nn_playground.py", title="Neuronové sítě: 2D Interaktivní hřiště", icon="🧠", url_path="day3_extras_nn_playground")
p_extra_backprop = st.Page("views/03_extras_backprop_calc.py", title="Backpropagation: Krok za krokem & Errata slidů", icon="🔢", url_path="day3_extras_backprop_calc")
p_extra_bagging_var = st.Page("views/03_extras_bagging_variance.py", title="Bagging & RF: Variance & Korelace Lab", icon="🌲", url_path="day3_extras_bagging_variance")
p_extra_boosting_anim = st.Page("views/03_extras_boosting_animator.py", title="Boosting: Sekvenční simulátor reziduí", icon="📈", url_path="day3_extras_boosting_animator")
p_extra_condorcet = st.Page("views/03_extras_condorcet_jury.py", title="Ansámbly: Condorcetova porota (Jury Theorem)", icon="🗳️", url_path="day3_extras_condorcet_jury")
p_extra_nn_arch = st.Page("views/03_extras_nn_architecture_calc.py", title="NN Architekt: Počítadlo tvarů & parametrů", icon="📐", url_path="day3_extras_nn_architecture_calc")
p_extra_cost_thresh = st.Page("views/03_extras_cost_threshold_calc.py", title="Kalkulátor prahu: Cost-Sensitive analýza (Srdce)", icon="⚖️", url_path="day3_extras_cost_threshold_calc")
p_extra_model_chooser = st.Page("views/03_extras_model_chooser.py", title="Průvodce výběrem: Jak volit model v praxi", icon="🧭", url_path="day3_extras_model_chooser")
p_extra_hyperparam = st.Page("views/03_extras_hyperparam_cheatsheet.py", title="Tahák & Diagnostika: Symptom-to-Fix matice", icon="📚", url_path="day3_extras_hyperparam_cheatsheet")

# Den 4: NLP (Natural Language Processing)
p_nlp_principles_theory = st.Page("views/04_nlp_text_preprocessing_theory.py", title="Teorie: Klíčové principy práce s textem", icon="📖", url_path="day4_nlp_principles_theory")
p_nlp_bow_tfidf_theory = st.Page("views/04_nlp_bow_tfidf_theory.py", title="Teorie: Bag of Words & TF-IDF (Vektorizace)", icon="🎒", url_path="day4_nlp_bow_tfidf_theory")
p_nlp_ex1 = st.Page("views/04_nlp_exercise_1.py", title="Cvičení 1: Předzpracování textu (IMDb) – Výsledky zadání", icon="🎯", url_path="day4_nlp_exercise_1")
p_nlp_ex1_critique = st.Page("views/04_nlp_exercise_1_critique.py", title="Cvičení 1: Předzpracování textu – Expertní analýza", icon="🔬", url_path="day4_nlp_exercise_1_critique")
p_nlp_ex1_nb = st.Page("views/04_nlp_exercise_1_nb.py", title="Cvičení 1: Předzpracování textu – Notebook", icon="🐍", url_path="day4_nlp_exercise_1_nb")
p_nlp_ex2 = st.Page("views/04_nlp_exercise_2.py", title="Cvičení 2: Lemmatizace textu (IMDb) – Výsledky zadání", icon="🎯", url_path="day4_nlp_exercise_2")
p_nlp_ex2_critique = st.Page("views/04_nlp_exercise_2_critique.py", title="Cvičení 2: Lemmatizace & Subwords – Expertní analýza", icon="🔬", url_path="day4_nlp_exercise_2_critique")
p_nlp_ex2_nb = st.Page("views/04_nlp_exercise_2_nb.py", title="Cvičení 2: Lemmatizace textu – Notebook", icon="🐍", url_path="day4_nlp_exercise_2_nb")
p_nlp_bow_ex = st.Page("views/04_nlp_exercise_bow.py", title="Cvičení 3: Bag of Words & Klasifikace (IMDb)", icon="🎯", url_path="day4_nlp_exercise_bow")
p_nlp_bow_ex_critique = st.Page("views/04_nlp_exercise_bow_critique.py", title="Cvičení 3: Bag of Words – Expertní analýza", icon="🔬", url_path="day4_nlp_exercise_bow_critique")
p_nlp_bow_ex_nb = st.Page("views/04_nlp_exercise_bow_nb.py", title="Cvičení 3: Bag of Words – Notebook", icon="🐍", url_path="day4_nlp_exercise_bow_nb")
p_nlp_tfidf_ex = st.Page("views/04_nlp_exercise_tfidf.py", title="Cvičení 4: TF-IDF & Srovnání modelů (IMDb)", icon="🎯", url_path="day4_nlp_exercise_tfidf")
p_nlp_tfidf_ex_critique = st.Page("views/04_nlp_exercise_tfidf_critique.py", title="Cvičení 4: TF-IDF – Expertní analýza", icon="🔬", url_path="day4_nlp_exercise_tfidf_critique")
p_nlp_tfidf_ex_nb = st.Page("views/04_nlp_exercise_tfidf_nb.py", title="Cvičení 4: TF-IDF – Notebook", icon="🐍", url_path="day4_nlp_exercise_tfidf_nb")
p_nlp_word2vec_theory = st.Page("views/04_nlp_word2vec_theory.py", title="Teorie: Word2Vec & Slovní vnoření", icon="🧬", url_path="day4_nlp_word2vec_theory")
p_nlp_word2vec_impl = st.Page("views/04_nlp_word2vec_implementation.py", title="Implementace: Word2Vec v knihovně Gensim", icon="🛠️", url_path="day4_nlp_word2vec_implementation")
p_nlp_w2v_ex = st.Page("views/04_nlp_exercise_word2vec.py", title="Cvičení 5: Word2Vec – Trénování & Klasifikace", icon="🎯", url_path="day4_nlp_exercise_word2vec")
p_nlp_w2v_ex_critique = st.Page("views/04_nlp_exercise_word2vec_critique.py", title="Cvičení 5: Word2Vec – Expertní analýza", icon="🔬", url_path="day4_nlp_exercise_word2vec_critique")
p_nlp_w2v_ex_nb = st.Page("views/04_nlp_exercise_word2vec_nb.py", title="Cvičení 5: Word2Vec – Notebook", icon="🐍", url_path="day4_nlp_exercise_word2vec_nb")
p_nlp_transformer_theory = st.Page("views/04_nlp_transformer_theory.py", title="Teorie: Architektura Transformer & Self-Attention", icon="⚡", url_path="day4_nlp_transformer_theory")
p_nlp_bert_theory = st.Page("views/04_nlp_bert_theory.py", title="Teorie & Demo: BERT & Hugging Face Pipeline", icon="🤖", url_path="day4_nlp_bert_theory")
p_nlp_day4_summary = st.Page("views/04_nlp_day_4_summary.py", title="Závěr Dne 4: Shrnutí & Kvíz", icon="🎓", url_path="day_4_summary")

# Domácí úkoly (Session 2 / Mezi Dnem 4 a 5): Random Forest
p_hw_titanic_rf = st.Page("views/04_homework_titanic_rf.py", title="DÚ 1: Titanic (Klasifikace) – Výsledky zadání", icon="🎯", url_path="hw_titanic_rf_results")
p_hw_titanic_rf_critique = st.Page("views/04_homework_titanic_rf_critique.py", title="DÚ 1: Titanic (Klasifikace) – Expertní analýza", icon="🔬", url_path="hw_titanic_rf_critique")
p_hw_titanic_rf_nb = st.Page("views/04_homework_titanic_rf_nb.py", title="DÚ 1: Titanic (Klasifikace) – Notebook", icon="🐍", url_path="hw_titanic_rf_nb")

p_hw_car_price_rf = st.Page("views/04_homework_car_price_rf.py", title="DÚ 2: Ceny aut (Regrese) – Výsledky zadání", icon="🎯", url_path="hw_car_price_rf_results")
p_hw_car_price_rf_critique = st.Page("views/04_homework_car_price_rf_critique.py", title="DÚ 2: Ceny aut (Regrese) – Expertní analýza", icon="🔬", url_path="hw_car_price_rf_critique")
p_hw_car_price_rf_nb = st.Page("views/04_homework_car_price_rf_nb.py", title="DÚ 2: Ceny aut (Regrese) – Notebook", icon="🐍", url_path="hw_car_price_rf_nb")

p_hw_diabetes_xgb = st.Page("views/04_homework_diabetes_xgb.py", title="DÚ 3: Diabetes (XGBoost) – Výsledky zadání", icon="🎯", url_path="hw_diabetes_xgb_results")
p_hw_diabetes_xgb_critique = st.Page("views/04_homework_diabetes_xgb_critique.py", title="DÚ 3: Diabetes (XGBoost) – Expertní analýza", icon="🔬", url_path="hw_diabetes_xgb_critique")
p_hw_diabetes_xgb_nb = st.Page("views/04_homework_diabetes_xgb_nb.py", title="DÚ 3: Diabetes (XGBoost) – Notebook", icon="🐍", url_path="hw_diabetes_xgb_nb")

p_hw_calories_xgb = st.Page("views/04_homework_calories_xgb.py", title="DÚ 4: Kalorie při cvičení (XGBoost) – Výsledky zadání", icon="🎯", url_path="hw_calories_xgb_results")
p_hw_calories_xgb_critique = st.Page("views/04_homework_calories_xgb_critique.py", title="DÚ 4: Kalorie při cvičení (XGBoost) – Expertní analýza", icon="🔬", url_path="hw_calories_xgb_critique")
p_hw_calories_xgb_nb = st.Page("views/04_homework_calories_xgb_nb.py", title="DÚ 4: Kalorie při cvičení (XGBoost) – Notebook", icon="🐍", url_path="hw_calories_xgb_nb")

p_hw_sonar_nn = st.Page("views/04_homework_sonar_nn.py", title="DÚ 5: Sonar (Neuronové sítě) – Výsledky zadání", icon="🎯", url_path="hw_sonar_nn_results")
p_hw_sonar_nn_critique = st.Page("views/04_homework_sonar_nn_critique.py", title="DÚ 5: Sonar (Neuronové sítě) – Expertní analýza", icon="🔬", url_path="hw_sonar_nn_critique")
p_hw_sonar_nn_nb = st.Page("views/04_homework_sonar_nn_nb.py", title="DÚ 5: Sonar (Neuronové sítě) – Notebook", icon="🐍", url_path="hw_sonar_nn_nb")

p_hw_auto_mpg_nn = st.Page("views/04_homework_auto_mpg_nn.py", title="DÚ 6: Auto MPG (Neuronové sítě) – Výsledky zadání", icon="🎯", url_path="hw_auto_mpg_nn_results")
p_hw_auto_mpg_nn_critique = st.Page("views/04_homework_auto_mpg_nn_critique.py", title="DÚ 6: Auto MPG (Neuronové sítě) – Expertní analýza", icon="🔬", url_path="hw_auto_mpg_nn_critique")
p_hw_auto_mpg_nn_nb = st.Page("views/04_homework_auto_mpg_nn_nb.py", title="DÚ 6: Auto MPG (Neuronové sítě) – Notebook", icon="🐍", url_path="hw_auto_mpg_nn_nb")

p_hw_mcdonalds_w2v = st.Page("views/04_homework_mcdonalds_w2v.py", title="DÚ 7: McDonald's (Word2Vec + SVM) – Výsledky zadání", icon="🎯", url_path="hw_mcdonalds_w2v_results")
p_hw_mcdonalds_w2v_critique = st.Page("views/04_homework_mcdonalds_w2v_critique.py", title="DÚ 7: McDonald's (Word2Vec + SVM) – Expertní analýza", icon="🔬", url_path="hw_mcdonalds_w2v_critique")
p_hw_mcdonalds_w2v_nb = st.Page("views/04_homework_mcdonalds_w2v_nb.py", title="DÚ 7: McDonald's (Word2Vec + SVM) – Notebook", icon="🐍", url_path="hw_mcdonalds_w2v_nb")

p_hw_kaggle_guide = st.Page("views/04_homework_kaggle_guide.py", title="Průvodce: Jak nahrát notebook na Kaggle", icon="🌐", url_path="hw_kaggle_guide")

# Prework Session 3: Neřízené učení (Unsupervised Learning)
p_prework_unsupervised_intro = st.Page("views/05_prework_unsupervised_learning_intro.py", title="Úvod do neřízeného učení (Unsupervised Learning)", icon="🌐", url_path="prework_unsupervised_intro")

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
        "15. Prework Session 2: Příprava (Dny 3–4)": [
            p_prework_nn_intro,
            p_prework_nlp_intro,
            p_prework_nltk_spacy,
        ],
        "16. Den 3: Advanced ML – Bagging": [
            p_ensemble_bagging_theory,
            p_rf_theory,
            p_rf_heart,
            p_rf_heart_critique,
            p_rf_heart_nb,
            p_rf_diam,
            p_rf_diam_critique,
            p_rf_diam_nb,
        ],
        "17. Den 3: Advanced ML – Boosting": [
            p_boosting_theory,
            p_xgb_heart,
            p_xgb_heart_critique,
            p_xgb_heart_nb,
            p_xgb_diam,
            p_xgb_diam_critique,
            p_xgb_diam_nb,
        ],
        "18. Den 3: Advanced ML – Neural Networks": [
            p_nn_theory,
            p_nn_mnist,
            p_nn_mnist_critique,
            p_nn_mnist_nb,
            p_nn_kc,
            p_nn_kc_critique,
            p_nn_kc_nb,
        ],
        "19. Den 3: Shrnutí & Závěr Dne 3": [
            p_day3_summary,
        ],
        "20. Den 3: Extras (Expertní laboratoř & Interaktivní nástroje)": [
            p_extra_mnist_draw,
            p_extra_nn_playground,
            p_extra_backprop,
            p_extra_bagging_var,
            p_extra_boosting_anim,
            p_extra_condorcet,
            p_extra_nn_arch,
            p_extra_cost_thresh,
            p_extra_model_chooser,
            p_extra_hyperparam,
        ],
        "21. Den 4: NLP – Zpracování textu & Reprezentace": [
            p_nlp_principles_theory,
            p_nlp_ex1,
            p_nlp_ex1_critique,
            p_nlp_ex1_nb,
            p_nlp_ex2,
            p_nlp_ex2_critique,
            p_nlp_ex2_nb,
            p_nlp_bow_tfidf_theory,
            p_nlp_bow_ex,
            p_nlp_bow_ex_critique,
            p_nlp_bow_ex_nb,
            p_nlp_tfidf_ex,
            p_nlp_tfidf_ex_critique,
            p_nlp_tfidf_ex_nb,
            p_nlp_word2vec_theory,
            p_nlp_word2vec_impl,
            p_nlp_w2v_ex,
            p_nlp_w2v_ex_critique,
            p_nlp_w2v_ex_nb,
        ],
        "22. Den 4: NLP – Transformery & BERT": [
            p_nlp_transformer_theory,
            p_nlp_bert_theory,
        ],
        "23. Den 4: Shrnutí & Závěr Dne 4": [
            p_nlp_day4_summary,
        ],
        "24. Domácí úkoly (Session 2): Sítě & Ensembly": [
            p_hw_titanic_rf,
            p_hw_titanic_rf_critique,
            p_hw_titanic_rf_nb,
            p_hw_car_price_rf,
            p_hw_car_price_rf_critique,
            p_hw_car_price_rf_nb,
            p_hw_diabetes_xgb,
            p_hw_diabetes_xgb_critique,
            p_hw_diabetes_xgb_nb,
            p_hw_calories_xgb,
            p_hw_calories_xgb_critique,
            p_hw_calories_xgb_nb,
            p_hw_sonar_nn,
            p_hw_sonar_nn_critique,
            p_hw_sonar_nn_nb,
            p_hw_auto_mpg_nn,
            p_hw_auto_mpg_nn_critique,
            p_hw_auto_mpg_nn_nb,
            p_hw_mcdonalds_w2v,
            p_hw_mcdonalds_w2v_critique,
            p_hw_mcdonalds_w2v_nb,
            p_hw_kaggle_guide,
        ],
        "25. Prework Session 3: Neřízené učení (Unsupervised Learning)": [
            p_prework_unsupervised_intro,
        ],
    }
)

nav.run()


