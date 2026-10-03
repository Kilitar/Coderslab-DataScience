from views.notebook_renderer import render_jupyter_notebook

render_jupyter_notebook(
    nb_rel_path="02_Classification/17_homework_diabetes_logistic_regression.ipynb",
    title="📓 DÚ: LogReg (Diabetes) – Notebook",
    description="Kompletní vypracované řešení cvičení Logistická regrese na datech diabetu: split 70/30, ladění inverzní regularizace C přes RandomizedSearchCV, evaluace na testovací sadě s metrikou F1/Recall, matice záměn a analýza vah a Odds Ratios."
)
