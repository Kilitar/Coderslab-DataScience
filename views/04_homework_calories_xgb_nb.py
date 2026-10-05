from views.notebook_renderer import render_jupyter_notebook

render_jupyter_notebook(
    nb_rel_path="04_Homework/04_calories_xgboost.ipynb",
    title="📓 DÚ: XGBoost (Kalorie) – Notebook",
    description="Kompletní vypracované řešení cvičení XGBoost Regressor na datech sportovních tréninků (calories_exercise_data.csv) s inspekcí přes ydata_profiling, odstraněním user_id, kódováním pohlaví, normalizací, rozdělením 70:30 (random_state=42), laděním hyperparametrů pomocí RandomizedSearchCV podle metriky MAE a finálním vyhodnocením na testovací sadě."
)
