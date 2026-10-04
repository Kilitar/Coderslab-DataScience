from views.notebook_renderer import render_jupyter_notebook

render_jupyter_notebook(
    nb_rel_path="04_NLP/05_word2vec_exercise.ipynb",
    title="📓 Cvičení: Word2Vec – Trénování & Klasifikace – Jupyter Notebook",
    description="Kompletní vypracované řešení cvičení Word2Vec - exercise: Trénování modelu Word2Vec v Gensim (vector_size=100, window=5, min_count=3, sg=0, epochs=15), nalezení 10 nejpodobnějších slov pro [cast, movie, comedy, watch, interesting], výpočet podobnosti movie vs comedy, zobrazení vektoru pro show, průměrování vektorů přes .apply() a lambda výraz a klasifikace sentimentu pomocí Logistické regrese."
)
