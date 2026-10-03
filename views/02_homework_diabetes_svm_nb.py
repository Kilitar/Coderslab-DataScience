from views.notebook_renderer import render_jupyter_notebook

render_jupyter_notebook(
    nb_rel_path="02_Classification/18_homework_diabetes_svm.ipynb",
    title="📓 DÚ: SVM (Diabetes) – Notebook",
    description="Kompletní vypracované řešení cvičení Support Vector Machine na datech diabetu: split 70/30, Bayesovská optimalizace hyperparametrů (kernel, C, gamma) přes Hyperopt (TPE) na metrice F1-Score a velké srovnání všech tří klasifikátorů."
)
