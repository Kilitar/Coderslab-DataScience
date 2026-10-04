from views.notebook_renderer import render_jupyter_notebook

render_jupyter_notebook(
    nb_rel_path="04_NLP/01_text_preprocessing_exercise_1.ipynb",
    title="📓 Cvičení 1: Předzpracování textu (IMDb) – Jupyter Notebook",
    description="Kompletní vypracované řešení cvičení Processing textual data - exercise 1: Načtení imdb_reviews.csv, definice funkce clean_review (lowercase, regulární výraz pro písmena, odstranění NLTK stop-slov), aplikace přes .apply() a export do imdb_reviews_preprocessed_1.csv."
)
