"""
Word2Vec - Exercise (Cvičení: Trénování Word2Vec modelu & Klasifikace recenzí)
=============================================================================
1. Load dataset imdb_reviews_preprocessed.csv via Pandas.
2. Train Word2Vec model on review_tokens_lemmatized using Gensim.
3. For words [cast, movie, comedy, watch, interesting], find top 10 similar words.
4. Calculate similarity between 'movie' and 'comedy'.
5. Display vector for word 'show'.
6. Generate averaged Word2Vec vectors for each review using .apply() and lambda.
7. Train LogisticRegression(max_iter=1000) for sentiment classification.
8. Predict on test set, generate classification_report and analyze results.
9. Export precomputed results to nlp_exercise_word2vec_precomputed.json.
"""

import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "imdb_reviews_preprocessed.csv"
if not CSV_PATH.exists():
    CSV_PATH = DATA_DIR / "imdb_reviews_lemmatized.csv"

MODEL_OUTPUT = OUTPUT_DIR / "imdb_word2vec.model"
JSON_OUTPUT = OUTPUT_DIR / "nlp_exercise_word2vec_precomputed.json"


def main():
    print(f"Loading data from: {CSV_PATH}")
    t0 = time.time()
    imdb_reviews = pd.read_csv(CSV_PATH)
    load_time = time.time() - t0
    print(f"Loaded {len(imdb_reviews)} rows in {load_time:.2f}s.")

    text_col = "review_tokens_lemmatized" if "review_tokens_lemmatized" in imdb_reviews.columns else "review_lemmatized"

    # Step 2: Prepare sentences and train Word2Vec
    print("Preparing tokenized sentences...")
    sentences = [str(text).split() for text in imdb_reviews[text_col].fillna("")]

    print("Training Word2Vec model (vector_size=100, window=5, min_count=3, sg=0, epochs=15)...")
    t1 = time.time()
    w2v_model = Word2Vec(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=3,
        workers=4,
        sg=0,  # CBOW
        epochs=15,
        seed=42
    )
    train_w2v_time = time.time() - t1
    vocab_size = len(w2v_model.wv)
    print(f"Word2Vec trained in {train_w2v_time:.2f}s. Vocabulary size: {vocab_size:,} words.")

    # Save trained Word2Vec model
    w2v_model.save(str(MODEL_OUTPUT))
    print(f"Model saved to: {MODEL_OUTPUT}")

    # Step 3: Top 10 similar words for [cast, movie, comedy, watch, interesting]
    target_words = ["cast", "movie", "comedy", "watch", "interesting"]
    similar_words_dict = {}
    for w in target_words:
        if w in w2v_model.wv:
            sims = w2v_model.wv.most_similar(w, topn=10)
            similar_words_dict[w] = [{"word": item[0], "similarity": round(float(item[1]), 4)} for item in sims]
        else:
            similar_words_dict[w] = []

    # Step 4: Similarity between 'movie' and 'comedy'
    sim_movie_comedy = float(w2v_model.wv.similarity("movie", "comedy"))
    print(f"Similarity between 'movie' and 'comedy': {sim_movie_comedy:.4f}")

    # Step 5: Display vector for 'show'
    vec_show = w2v_model.wv.get_vector("show")
    vec_show_sample = [round(float(v), 4) for v in vec_show[:10]]
    print(f"Vector for 'show' (first 10 dims): {vec_show_sample}")

    # Step 6: Averaged Word2Vec vector for each review using .apply() and lambda
    print("Generating averaged Word2Vec vectors for each review...")
    t2 = time.time()

    def get_review_vector(text, model):
        words = [w for w in str(text).split() if w in model.wv]
        if not words:
            return np.zeros(model.vector_size, dtype=np.float32)
        return np.mean([model.wv[w] for w in words], axis=0).astype(np.float32)

    imdb_reviews["review_vector"] = imdb_reviews[text_col].apply(
        lambda t: get_review_vector(t, w2v_model)
    )
    avg_vec_time = time.time() - t2
    print(f"Vectors generated in {avg_vec_time:.2f}s.")

    X = np.vstack(imdb_reviews["review_vector"].values)
    y = imdb_reviews["sentiment"].apply(lambda s: 1 if str(s).lower() == "positive" else 0)

    # Step 7 & 8: Train-test split (80/20) and Logistic Regression
    print("Splitting dataset and training Logistic Regression(max_iter=1000)...")
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, imdb_reviews.index, test_size=0.2, random_state=42, stratify=y
    )

    t3 = time.time()
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_time = time.time() - t3

    # Step 9: Prediction and Classification Report
    y_pred = lr.predict(X_test)
    y_prob = lr.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred).tolist()
    clf_report = classification_report(y_test, y_pred, output_dict=True)
    clf_report_text = classification_report(y_test, y_pred)
    print(clf_report_text)

    # Sample predictions
    test_df = imdb_reviews.loc[idx_test].copy()
    test_df["actual"] = y_test.values
    test_df["predicted"] = y_pred
    test_df["prob_positive"] = y_prob.round(4)

    samples = []
    for _, row in test_df.head(10).iterrows():
        samples.append({
            "text": str(row[text_col])[:200] + "...",
            "actual": "positive" if row["actual"] == 1 else "negative",
            "predicted": "positive" if row["predicted"] == 1 else "negative",
            "prob_positive": float(row["prob_positive"]),
            "is_correct": bool(row["actual"] == row["predicted"])
        })

    results = {
        "metadata": {
            "exercise": "Word2Vec - exercise",
            "input_csv": str(CSV_PATH.name),
            "total_reviews": len(imdb_reviews),
            "vector_size": 100,
            "window": 5,
            "min_count": 3,
            "sg": 0,
            "epochs": 15,
            "vocab_size": vocab_size,
            "w2v_train_time_s": round(train_w2v_time, 2),
            "lr_train_time_s": round(lr_time, 3),
            "avg_vector_time_s": round(avg_vec_time, 2)
        },
        "word_similarities": similar_words_dict,
        "movie_comedy_similarity": round(sim_movie_comedy, 4),
        "show_vector": {
            "dimension": int(vec_show.shape[0]),
            "first_10_values": vec_show_sample
        },
        "metrics": {
            "accuracy": round(float(acc), 4),
            "precision_weighted": round(float(prec), 4),
            "recall_weighted": round(float(rec), 4),
            "f1_weighted": round(float(f1), 4),
            "roc_auc": round(float(auc), 4),
            "confusion_matrix": cm,
            "classification_report": clf_report,
            "classification_report_text": clf_report_text
        },
        "samples": samples
    }

    with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Precomputed results saved to: {JSON_OUTPUT}")


if __name__ == "__main__":
    main()
