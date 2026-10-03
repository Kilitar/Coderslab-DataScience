from views.notebook_renderer import render_jupyter_notebook

render_jupyter_notebook(
    nb_rel_path="02_Classification/16_homework_diabetes_knn.ipynb",
    title="📓 DÚ: k-NN (Diabetes) – Notebook",
    description="Kompletní vypracované řešení cvičení K-Nearest Neighbors na škálovaných datech diabetes_scaled.csv: dělení 70/30, GridSearchCV pro n_neighbors a metric, medicínská volba optimalizační metriky (F1 vs Recall) a evaluace na testovací sadě s maticí záměn."
)
