from views.notebook_renderer import render_jupyter_notebook

render_jupyter_notebook(
    nb_rel_path="01_Regression/12_homework_concrete_regularization.ipynb",
    title="📓 DÚ: Regularizace (Beton) – Notebook",
    description="Kompletní vypracované řešení cvičení Regularizace na datech betonu: ladění ElasticNet (L1 + L2) přes RandomizedSearchCV s regresními metrikami (R2, RMSE) i doslovná implementace LogisticRegression(C, penalty) na binarizované normě pevnosti betonu."
)
