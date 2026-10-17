from views.notebook_renderer import render_jupyter_notebook

render_jupyter_notebook(
    nb_rel_path="05_Unsupervised_Learning/04_dbscan_moons_exercise.ipynb",
    title="🐍 Cvičení: Moons – Notebook",
    description="Kompletní vypracovaný a spuštěný Jupyter sešit: Hustotní shlukování DBSCAN na datech dvou půlměsíců (make_moons, 500 vzorků, noise=0.10), "
    "standardizace, analytické stanovení eps přes k-NN elbow metodu, přiřazení do proměnné dbscan_labels a srovnání se selháním K-Means."
)
