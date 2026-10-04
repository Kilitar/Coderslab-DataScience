"""
TF-IDF - Exercise (Cvičení: Reprezentace textu pomocí TF-IDF & Porovnání LR vs. SVM)
==================================================================================
1. Load dataset imdb_reviews_preprocessed.csv via Pandas.
2. Vectorize review_tokens_lemmatized using TfidfVectorizer(max_features=10000).
3. Split data into train and test sets (80/20).
4. Train LogisticRegression(max_iter=1000) for sentiment classification.
5. Predict on test set and generate classification_report.
6. Repeat for Support Vector Machine (LinearSVC) and compare both models.
7. Export precomputed metrics to nlp_exercise_tfidf_precomputed.json.
"""

import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
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
from sklearn.svm import LinearSVC

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "imdb_reviews_preprocessed.csv"
if not CSV_PATH.exists():
    CSV_PATH = DATA_DIR / "imdb_reviews_lemmatized.csv"

JSON_OUTPUT = OUTPUT_DIR / "nlp_exercise_tfidf_precomputed.json"


def main():
    print(f"Loading data from: {CSV_PATH}")
    t0 = time.time()
    imdb_reviews = pd.read_csv(CSV_PATH)
    load_time = time.time() - t0
    print(f"Loaded {len(imdb_reviews)} rows in {load_time:.2f}s.")

    # Determine column name
    text_col = "review_tokens_lemmatized" if "review_tokens_lemmatized" in imdb_reviews.columns else "review_lemmatized"

    # Step 2: Vectorization with TfidfVectorizer(max_features=10000)
    print(f"Vectorizing {text_col} with TfidfVectorizer(max_features=10000)...")
    t1 = time.time()
    vectorizer = TfidfVectorizer(max_features=10000)
    X = vectorizer.fit_transform(imdb_reviews[text_col].fillna(""))
    vec_time = time.time() - t1
    feature_names = vectorizer.get_feature_names_out()
    print(f"Vectorized into shape {X.shape} in {vec_time:.2f}s.")

    # Target encoding
    y = imdb_reviews["sentiment"].apply(lambda s: 1 if str(s).lower() == "positive" else 0)

    # Step 3: Train-test split (80/20)
    print("Splitting dataset into train (80%) and test (20%) sets...")
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, imdb_reviews.index, test_size=0.2, random_state=42, stratify=y
    )

    # Step 4 & 5: Model 1 - Logistic Regression
    print("Training Logistic Regression (max_iter=1000)...")
    t_lr_start = time.time()
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_train_time = time.time() - t_lr_start

    y_pred_lr = lr_model.predict(X_test)
    y_prob_lr = lr_model.predict_proba(X_test)[:, 1]

    acc_lr = accuracy_score(y_test, y_pred_lr)
    prec_lr = precision_score(y_test, y_pred_lr, average="weighted")
    rec_lr = recall_score(y_test, y_pred_lr, average="weighted")
    f1_lr = f1_score(y_test, y_pred_lr, average="weighted")
    auc_lr = roc_auc_score(y_test, y_prob_lr)
    cm_lr = confusion_matrix(y_test, y_pred_lr).tolist()
    clf_rep_lr = classification_report(y_test, y_pred_lr, output_dict=True)

    # Step 6: Model 2 - Support Vector Machine (LinearSVC)
    print("Training Support Vector Machine (LinearSVC)...")
    t_svm_start = time.time()
    svm_model = LinearSVC(random_state=42)
    svm_model.fit(X_train, y_train)
    svm_train_time = time.time() - t_svm_start

    y_pred_svm = svm_model.predict(X_test)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    prec_svm = precision_score(y_test, y_pred_svm, average="weighted")
    rec_svm = recall_score(y_test, y_pred_svm, average="weighted")
    f1_svm = f1_score(y_test, y_pred_svm, average="weighted")
    cm_svm = confusion_matrix(y_test, y_pred_svm).tolist()
    clf_rep_svm = classification_report(y_test, y_pred_svm, output_dict=True)

    print(f"Logistic Regression: Accuracy = {acc_lr*100:.2f}%, F1 = {f1_lr:.4f} (Time: {lr_train_time:.2f}s)")
    print(f"LinearSVC (SVM):      Accuracy = {acc_svm*100:.2f}%, F1 = {f1_svm:.4f} (Time: {svm_train_time:.2f}s)")

    # Model Interpretability: Top weights for LR and SVM
    lr_coefs = lr_model.coef_[0]
    svm_coefs = svm_model.coef_[0]

    top_pos_lr = lr_coefs.argsort()[-15:][::-1]
    top_neg_lr = lr_coefs.argsort()[:15]

    top_pos_svm = svm_coefs.argsort()[-15:][::-1]
    top_neg_svm = svm_coefs.argsort()[:15]

    top_features = {
        "lr_positive": [{"word": feature_names[i], "weight": round(float(lr_coefs[i]), 4)} for i in top_pos_lr],
        "lr_negative": [{"word": feature_names[i], "weight": round(float(lr_coefs[i]), 4)} for i in top_neg_lr],
        "svm_positive": [{"word": feature_names[i], "weight": round(float(svm_coefs[i]), 4)} for i in top_pos_svm],
        "svm_negative": [{"word": feature_names[i], "weight": round(float(svm_coefs[i]), 4)} for i in top_neg_svm],
    }

    # Sample comparisons on test set
    test_df = imdb_reviews.loc[idx_test].copy()
    test_df["actual"] = y_test.values
    test_df["pred_lr"] = y_pred_lr
    test_df["pred_svm"] = y_pred_svm
    test_df["prob_lr"] = y_prob_lr.round(4)

    samples = []
    for _, row in test_df.head(10).iterrows():
        samples.append({
            "text": str(row[text_col])[:200] + "...",
            "actual": "positive" if row["actual"] == 1 else "negative",
            "pred_lr": "positive" if row["pred_lr"] == 1 else "negative",
            "pred_svm": "positive" if row["pred_svm"] == 1 else "negative",
            "prob_lr": float(row["prob_lr"]),
            "models_agree": bool(row["pred_lr"] == row["pred_svm"])
        })

    sparsity = 1.0 - (X.nnz / (X.shape[0] * X.shape[1]))

    results = {
        "metadata": {
            "exercise": "TF-IDF - exercise",
            "input_csv": str(CSV_PATH.name),
            "text_column": text_col,
            "total_reviews": len(imdb_reviews),
            "max_features": 10000,
            "train_size": len(y_train),
            "test_size": len(y_test),
            "vectorization_time_s": round(vec_time, 2),
            "sparsity_pct": round(sparsity * 100, 2)
        },
        "logistic_regression": {
            "accuracy": round(float(acc_lr), 4),
            "precision_weighted": round(float(prec_lr), 4),
            "recall_weighted": round(float(rec_lr), 4),
            "f1_weighted": round(float(f1_lr), 4),
            "roc_auc": round(float(auc_lr), 4),
            "train_time_s": round(lr_train_time, 3),
            "confusion_matrix": cm_lr,
            "classification_report": clf_rep_lr
        },
        "svm": {
            "accuracy": round(float(acc_svm), 4),
            "precision_weighted": round(float(prec_svm), 4),
            "recall_weighted": round(float(rec_svm), 4),
            "f1_weighted": round(float(f1_svm), 4),
            "train_time_s": round(svm_train_time, 3),
            "confusion_matrix": cm_svm,
            "classification_report": clf_rep_svm
        },
        "top_features": top_features,
        "samples": samples
    }

    with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Precomputed results saved to: {JSON_OUTPUT}")


if __name__ == "__main__":
    main()
