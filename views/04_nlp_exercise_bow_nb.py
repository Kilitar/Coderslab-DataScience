from views.notebook_renderer import render_jupyter_notebook

render_jupyter_notebook(
    nb_rel_path="04_NLP/03_bow_exercise.ipynb",
    title="📓 Cvičení: Bag of Words & Klasifikace sentimentu – Jupyter Notebook",
    description="Kompletní vypracované řešení cvičení Bag of Words - exercise: Načtení imdb_reviews_lemmatized.csv, vektorizace pomocí CountVectorizer(max_features=10000), rozdělení train_test_split, trénování LogisticRegression(max_iter=1000), predikce na testovací sadě, classification_report a analýza vah naučených koeficientů."
)
