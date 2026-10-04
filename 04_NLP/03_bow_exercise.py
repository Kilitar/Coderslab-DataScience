"""
Bag of Words - Exercise (Cvičení: Reprezentace textu pomocí Bag of Words)
========================================================================
1. Load dataset imdb_reviews_lemmatized.csv via Pandas.
2. Vectorize review_lemmatized using CountVectorizer(max_features=10000).
3. Split data into train and test sets (80/20).
4. Train LogisticRegression(max_iter=1000) for sentiment classification.
5. Predict on test set.
6. Generate classification_report and analyze results.
7. Export precomputed metrics to nlp_exercise_bow_precomputed.json.
"""

import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "imdb_reviews_lemmatized.csv"
if not CSV_PATH.exists():
    CSV_PATH = DATA_DIR / "MAL_downloadable materials_session 2" / "Day 4" / "imdb_reviews_lemmatized.csv"
JSON_OUTPUT = OUTPUT_DIR / "nlp_exercise_bow_precomputed.json"


def main():
    print(f"Loading data from: {CSV_PATH}")
    t0 = time.time()
    imdb_reviews = pd.read_csv(CSV_PATH)
    load_time = time.time() - t0
    print(f"Loaded {len(imdb_reviews)} rows in {load_time:.2f}s.")

    # Step 2: Vectorization with CountVectorizer(max_features=10000)
    print("Vectorizing review_lemmatized with CountVectorizer(max_features=10000)...")
    t1 = time.time()
    vectorizer = CountVectorizer(max_features=10000)
    X = vectorizer.fit_transform(imdb_reviews["review_lemmatized"].fillna(""))
    vec_time = time.time() - t1
    feature_names = vectorizer.get_feature_names_out()
    print(f"Vectorized into shape {X.shape} in {vec_time:.2f}s. Vocab size: {len(feature_names)}")

    # Target encoding
    y = imdb_reviews["sentiment"].apply(lambda s: 1 if str(s).lower() == "positive" else 0)

    # Step 3: Train-test split (80/20)
    print("Splitting dataset into train (80%) and test (20%) sets...")
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, imdb_reviews.index, test_size=0.2, random_state=42, stratify=y
    )

    # Step 4: Model training - LogisticRegression(max_iter=1000)
    print("Training LogisticRegression(max_iter=1000)...")
    t2 = time.time()
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    train_time = time.time() - t2
    print(f"Model trained in {train_time:.2f}s.")

    # Step 5 & 6: Prediction & Classification Report
    print("Predicting on test set and evaluating...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec_weighted = precision_score(y_test, y_pred, average="weighted")
    rec_weighted = recall_score(y_test, y_pred, average="weighted")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred).tolist()
    clf_report = classification_report(y_test, y_pred, output_dict=True)
    clf_report_text = classification_report(y_test, y_pred)
    print(clf_report_text)

    # Model Interpretability: Top positive and negative coefficients
    coefs = model.coef_[0]
    top_pos_idx = coefs.argsort()[-20:][::-1]
    top_neg_idx = coefs.argsort()[:20]

    top_positive_words = [
        {"word": feature_names[i], "coefficient": round(float(coefs[i]), 4)}
        for i in top_pos_idx
    ]
    top_negative_words = [
        {"word": feature_names[i], "coefficient": round(float(coefs[i]), 4)}
        for i in top_neg_idx
    ]

    # Sample misclassified and correctly classified examples
    test_df = imdb_reviews.loc[idx_test].copy()
    test_df["actual"] = y_test.values
    test_df["predicted"] = y_pred
    test_df["prob_positive"] = y_pred_proba.round(4)
    test_df["is_correct"] = test_df["actual"] == test_df["predicted"]

    correct_samples = []
    for _, row in test_df[test_df["is_correct"]].head(5).iterrows():
        correct_samples.append({
            "text": str(row["review_lemmatized"])[:250] + "...",
            "actual": "positive" if row["actual"] == 1 else "negative",
            "predicted": "positive" if row["predicted"] == 1 else "negative",
            "prob_positive": float(row["prob_positive"])
        })

    misclassified_samples = []
    for _, row in test_df[~test_df["is_correct"]].head(5).iterrows():
        misclassified_samples.append({
            "text": str(row["review_lemmatized"])[:250] + "...",
            "actual": "positive" if row["actual"] == 1 else "negative",
            "predicted": "positive" if row["predicted"] == 1 else "negative",
            "prob_positive": float(row["prob_positive"]),
            "error_type": "False Positive (předpovězeno pozitivní, ale je negativní)" if row["predicted"] == 1 else "False Negative (předpovězeno negativní, ale je pozitivní)"
        })

    sparsity = 1.0 - (X.nnz / (X.shape[0] * X.shape[1]))

    results = {
        "metadata": {
            "exercise": "Bag of Words - exercise",
            "input_csv": str(CSV_PATH.name),
            "total_reviews": len(imdb_reviews),
            "max_features": 10000,
            "train_size": len(y_train),
            "test_size": len(y_test),
            "model": "LogisticRegression(max_iter=1000)",
            "vectorization_time_s": round(vec_time, 2),
            "training_time_s": round(train_time, 2),
            "sparsity_pct": round(sparsity * 100, 2)
        },
        "metrics": {
            "accuracy": round(float(acc), 4),
            "precision_weighted": round(float(prec_weighted), 4),
            "recall_weighted": round(float(rec_weighted), 4),
            "f1_weighted": round(float(f1_weighted), 4),
            "confusion_matrix": cm,
            "classification_report": clf_report,
            "classification_report_text": clf_report_text
        },
        "top_words": {
            "positive": top_positive_words,
            "negative": top_negative_words
        },
        "samples": {
            "correct": correct_samples,
            "misclassified": misclassified_samples
        }
    }

    with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Precomputed results saved to: {JSON_OUTPUT}")


if __name__ == "__main__":
    main()
