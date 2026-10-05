from views.notebook_renderer import render_jupyter_notebook

render_jupyter_notebook(
    nb_rel_path="04_Homework/03_diabetes_xgboost.ipynb",
    title="📓 DÚ: XGBoost (Diabetes) – Notebook",
    description="Kompletní vypracované řešení cvičení XGBoost Classifier na datech pacientek kmene Pima (diabetes.csv) s inspekcí přes ydata_profiling, analýzou biologicky nemožných nul, rozdělením 80:20 (random_state=21), laděním hyperparametrů pomocí GridSearchCV podle metriky precision a finálním vyhodnocením na testovací sadě."
)
