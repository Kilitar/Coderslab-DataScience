from views.notebook_renderer import render_jupyter_notebook

render_jupyter_notebook(
    nb_rel_path="02_Classification/15_homework_diabetes_preprocessing.ipynb",
    title="📓 DÚ: Příprava dat (Diabetes) – Notebook",
    description="Kompletní vypracované řešení cvičení Příprava dat pro klasifikační modely na datasetu diabetes.csv: normalizace sloupců, odhalení biologicky nemožných nul, imputace průměrem/mediánem dle šikmosti, korelační analýza a standardizace StandardScalerem."
)
