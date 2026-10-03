from views.notebook_renderer import render_jupyter_notebook

render_jupyter_notebook(
    nb_rel_path="02_Classification/10_decision_tree_lumbar_exercise_1.ipynb",
    title="📓 Cvičení 1: Bederní páteř (Rozhodovací strom)",
    description="Kompletní vypracované cvičení 1: Načtení lumbar_normalized_df.csv, train/test split 75/25, výchozí DecisionTreeClassifier, vyhodnocení Precision a experimenty s laděním max_depth a min_samples_leaf pro překonání výchozího modelu."
)
