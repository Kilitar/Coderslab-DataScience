from views.notebook_renderer import render_jupyter_notebook

render_jupyter_notebook(
    nb_rel_path="04_Homework/07_mcdonalds_word2vec_svm.ipynb",
    title="📓 DÚ: NLP McDonald's (Word2Vec + SVM) – Notebook",
    description="Kompletní vypracované řešení cvičení NLP na recenzích McDonald's (mcdonalds_reviews.csv): Čištění textu, stop-slova, lemmatizace s WordNetem, trénování 100D Word2Vec modelu, průměrování vektorů vět přes .apply() a lambda, split 70:30 a trénování Support Vector Machine (LinearSVC) pro 3-třídní sentiment analýzu."
)
