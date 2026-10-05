from views.notebook_renderer import render_jupyter_notebook

render_jupyter_notebook(
    nb_rel_path="04_Homework/05_sonar_neural_network.ipynb",
    title="📓 DÚ: Neuronové sítě (Sonar) – Notebook",
    description="Kompletní vypracované řešení cvičení Neuronové sítě v Kerasu na datech odrazů sonaru (sonar.csv) z UCI repository: Příprava dat, normalizace StandardScaler, architektura Sequential MLP s Dropoutem, kompilace s optimalizátorem Adam a ztrátovou funkcí binary_crossentropy, trénování s validačním rozdělením a finální vyhodnocení na testovací sadě."
)
