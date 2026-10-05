"""
Domácí úkol (Session 2): NLP – Klasifikace recenzí McDonald's pomocí Word2Vec a SVM
====================================================================================
Dataset: mcdonalds_reviews.csv (33 396 recenzí)
Cíl: Předzpracování textu, trénování Word2Vec, vektorové průměrování vět (.apply + lambda),
     rozdělení 70:30 a trénování klasifikátoru Support Vector Machine (3 třídy sentimentu).
"""

import os
import re
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.decomposition import PCA

try:
    from gensim.models import Word2Vec
except ImportError:
    raise ImportError("Knihovna gensim není nainstalována. Spusťte ve vhodném Python prostředí s gensim.")


def ensure_nltk_corpora():
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    try:
        WordNetLemmatizer().lemmatize('test')
    except LookupError:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)


def parse_and_load_reviews(base_dir: Path) -> pd.DataFrame:
    clean_csv = base_dir / "data" / "mcdonalds_reviews.csv"
    if clean_csv.exists() and clean_csv.stat().st_size > 100000:
        print(f"Loading cleaned dataset directly from: {clean_csv}")
        df = pd.read_csv(clean_csv)
        if "review" in df.columns and "rating" in df.columns:
            return df

    raw_candidates = [
        base_dir / "data" / "MAL_downloadable materials_session 2" / "Homework" / "mcdonalds_reviews.csv",
        base_dir / "data" / "mcdonalds_reviews.csv"
    ]
    raw_path = next(p for p in raw_candidates if p.exists())
    print(f"Parsing raw CSV file from: {raw_path}...")

    with open(raw_path, 'r', encoding='latin-1', errors='replace') as f:
        text = f.read()

    lines = text.splitlines()
    records = []
    current_id = None
    current_text = []

    for line in lines[1:]:
        cleaned_line = line.rstrip(';').strip()
        if not cleaned_line:
            continue
        start_match = re.match(r'^"?(\d+),"?', cleaned_line)
        end_match = re.search(r',"?([1-5]\s*stars?)"?$', cleaned_line, re.IGNORECASE)

        if start_match:
            if current_id is not None and current_text:
                pass
            current_id = int(start_match.group(1))
            rest = cleaned_line[start_match.end():]
            if end_match:
                rating_str = end_match.group(1)
                review_part = rest[:end_match.start() - start_match.end()].strip('" ')
                records.append((current_id, review_part, rating_str))
                current_id = None
                current_text = []
            else:
                current_text = [rest.strip('" ')]
        else:
            if end_match:
                review_part = cleaned_line[:end_match.start()].strip('" ')
                current_text.append(review_part)
                rating_str = end_match.group(1)
                if current_id is not None:
                    records.append((current_id, " ".join(current_text).strip(), rating_str))
                current_id = None
                current_text = []
            else:
                current_text.append(cleaned_line.strip('" '))

    df = pd.DataFrame(records, columns=['reviewer_id', 'review', 'rating'])
    print(f"Extracted {len(df)} records from raw CSV.")

    # Save clean CSV for future fast loading
    clean_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(clean_csv, index=False, encoding='utf-8')
    print(f"Saved clean dataset to {clean_csv}")
    return df


def main():
    print("=== NLP - Exercise: McDonald's Reviews Classification with Word2Vec & SVM ===")
    t_start = time.time()
    base_dir = Path(__file__).resolve().parent.parent
    out_dir = base_dir / "04_Homework" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ensure NLTK corpora
    ensure_nltk_corpora()

    # 2. Load dataset
    df = parse_and_load_reviews(base_dir)

    # 3. Target Sentiment Mapping:
    # 1 and 2 stars -> negative, 3 stars -> neutral, 4 and 5 stars -> positive
    def map_stars_to_sentiment(r):
        r_str = str(r).lower()
        if '1' in r_str or '2' in r_str:
            return 'negative'
        elif '3' in r_str:
            return 'neutral'
        elif '4' in r_str or '5' in r_str:
            return 'positive'
        return None

    df['sentiment'] = df['rating'].apply(map_stars_to_sentiment)
    df = df.dropna(subset=['sentiment']).reset_index(drop=True)
    print(f"Valid ratings distribution:\n{df['sentiment'].value_counts()}")

    # 4. Text Cleaning & Token Lemmatization
    # - normalize case (.lower())
    # - remove all non-letters (re.sub)
    # - remove stopwords
    # - perform token lemmatization
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    print("Cleaning review text and performing lemmatization...")
    t0 = time.time()

    def clean_and_lemmatize(text):
        if not isinstance(text, str):
            return []
        letters_only = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
        tokens = [
            lemmatizer.lemmatize(word)
            for word in letters_only.split()
            if word not in stop_words and len(word) > 1
        ]
        return tokens

    df['tokens'] = df['review'].apply(clean_and_lemmatize)
    print(f"Lemmatization completed in {time.time() - t0:.2f}s.")

    # 5. Filter out reviews that have 0 tokens
    # (Hint from prompt: ensure that there are no reviews that have 0 tokens before calling the method on the data frame)
    before_filter = len(df)
    df = df[df['tokens'].apply(lambda t: len(t) > 0)].reset_index(drop=True)
    after_filter = len(df)
    print(f"Removed {before_filter - after_filter} reviews with 0 tokens. Clean dataset: {after_filter} reviews.")

    # 6. Train Word2Vec Model
    print("Training Word2Vec model on tokenized sentences...")
    t0 = time.time()
    w2v_dim = 100
    w2v_model = Word2Vec(
        sentences=df['tokens'],
        vector_size=w2v_dim,
        window=5,
        min_count=3,
        workers=4,
        epochs=10,
        seed=42
    )
    print(f"Word2Vec trained in {time.time() - t0:.2f}s! Vocabulary size: {len(w2v_model.wv):,} words.")

    # 7. For each review, generate an averaged Word2Vec vector using .apply() and lambda
    print("Generating averaged Word2Vec vector for each review via .apply() and lambda...")
    t0 = time.time()

    def get_averaged_vector(tokens, model, dim=100):
        vectors = [model.wv[w] for w in tokens if w in model.wv]
        if not vectors:
            return np.zeros(dim)
        return np.mean(vectors, axis=0)

    df['w2v_vector'] = df['tokens'].apply(lambda tokens: get_averaged_vector(tokens, w2v_model, w2v_dim))
    print(f"Averaged vectors generated in {time.time() - t0:.2f}s.")

    # 8. Divide data into train and test sets in a 70:30 ratio
    X = np.vstack(df['w2v_vector'].values)
    y = df['sentiment'].values

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.30, random_state=42, stratify=y
    )
    print(f"Train set: {X_train.shape[0]} reviews, Test set: {X_test.shape[0]} reviews (70:30 ratio)")

    # 9. Build and train Support Vector Machine (LinearSVC)
    print("Training Support Vector Machine (LinearSVC)...")
    t0 = time.time()
    svm = LinearSVC(random_state=42, max_iter=2500, C=1.0)
    svm.fit(X_train, y_train)
    print(f"SVM trained in {time.time() - t0:.2f}s.")

    # Baselines for comparison
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    p_lr = lr.predict(X_test)
    acc_lr = float(accuracy_score(y_test, p_lr))
    f1_lr = float(f1_score(y_test, p_lr, average='weighted'))

    svm_bal = LinearSVC(random_state=42, max_iter=2500, class_weight='balanced')
    svm_bal.fit(X_train, y_train)
    p_svm_bal = svm_bal.predict(X_test)
    acc_svm_bal = float(accuracy_score(y_test, p_svm_bal))
    f1_svm_bal = float(f1_score(y_test, p_svm_bal, average='weighted'))

    # 10. Make predictions on test set
    y_pred = svm.predict(X_test)

    # 11. Calculate metrics and classification_report
    acc = float(accuracy_score(y_test, y_pred))
    prec_macro = float(precision_score(y_test, y_pred, average='macro'))
    rec_macro = float(recall_score(y_test, y_pred, average='macro'))
    f1_macro = float(f1_score(y_test, y_pred, average='macro'))
    f1_weighted = float(f1_score(y_test, y_pred, average='weighted'))

    target_labels = ['negative', 'neutral', 'positive']
    cm = confusion_matrix(y_test, y_pred, labels=target_labels).tolist()
    clf_rep_dict = classification_report(y_test, y_pred, labels=target_labels, output_dict=True)
    clf_rep_text = classification_report(y_test, y_pred, labels=target_labels)

    print("\n" + "=" * 55)
    print("        CLASSIFICATION REPORT (SVM on Test Set)")
    print("=" * 55)
    print(clf_rep_text)
    print(f"Overall Accuracy : {acc * 100:.2f} %")
    print(f"Macro F1-Score   : {f1_macro:.4f}")
    print(f"Weighted F1-Score: {f1_weighted:.4f}")
    print("=" * 55)

    # 12. Sémantický prostor Word2Vec: Top 8 podobných slov pro klíčové restaurační termíny
    key_words = ['burger', 'fries', 'staff', 'manager', 'clean', 'dirty', 'cold', 'slow', 'fresh', 'delicious']
    similarity_map = {}
    for kw in key_words:
        if kw in w2v_model.wv:
            sims = w2v_model.wv.most_similar(kw, topn=6)
            similarity_map[kw] = [{"word": w, "similarity": round(float(s), 4)} for w, s in sims]

    # 2D PCA projekce pro vybraná slova
    sample_vocab = [
        'burger', 'cheeseburger', 'sandwich', 'nugget', 'fries', 'drink', 'shake', 'ice', 'coffee',
        'staff', 'employee', 'manager', 'worker', 'crew', 'cashier', 'customer', 'lady', 'girl',
        'clean', 'dirty', 'fresh', 'hot', 'cold', 'slow', 'fast', 'rude', 'friendly', 'polite',
        'horrible', 'terrible', 'worst', 'best', 'delicious', 'tasty', 'disgusting', 'gross', 'love'
    ]
    vocab_present = [w for w in sample_vocab if w in w2v_model.wv]
    vectors_present = np.array([w2v_model.wv[w] for w in vocab_present])
    pca = PCA(n_components=2, random_state=42)
    coords_2d = pca.fit_transform(vectors_present)

    pca_points = []
    for w, (x_coord, y_coord) in zip(vocab_present, coords_2d):
        category = "Jídlo & Pití" if w in ['burger', 'cheeseburger', 'sandwich', 'nugget', 'fries', 'drink', 'shake', 'ice', 'coffee'] else \
                   "Personál" if w in ['staff', 'employee', 'manager', 'worker', 'crew', 'cashier', 'customer', 'lady', 'girl'] else \
                   "Kvalita / Vlastnosti"
        pca_points.append({
            "word": w,
            "x": round(float(x_coord), 4),
            "y": round(float(y_coord), 4),
            "category": category
        })

    # 13. Ukázkové predikce z testovací sady
    test_df = df.loc[idx_test].copy()
    test_df['actual'] = y_test
    test_df['predicted'] = y_pred
    test_df['is_correct'] = test_df['actual'] == test_df['predicted']

    sample_reviews = []
    for _, r in test_df.head(30).iterrows():
        sample_reviews.append({
            "review": str(r['review'])[:200] + ("..." if len(str(r['review'])) > 200 else ""),
            "actual_stars": str(r['rating']),
            "actual": str(r['actual']),
            "predicted": str(r['predicted']),
            "is_correct": bool(r['is_correct'])
        })

    # 14. Export výsledků
    results = {
        "metadata": {
            "dataset": "mcdonalds_reviews.csv",
            "source": "Kaggle McDonald's Store Reviews",
            "total_reviews": len(df),
            "train_reviews": len(X_train),
            "test_reviews": len(X_test),
            "embedding_dim": w2v_dim,
            "vocab_size": len(w2v_model.wv),
            "classes": target_labels,
            "class_distribution": {
                "positive": int((df['sentiment'] == 'positive').sum()),
                "negative": int((df['sentiment'] == 'negative').sum()),
                "neutral": int((df['sentiment'] == 'neutral').sum())
            }
        },
        "model_comparison": {
            "svm_standard": {"accuracy": round(acc, 4), "f1_weighted": round(f1_weighted, 4), "f1_macro": round(f1_macro, 4)},
            "svm_balanced": {"accuracy": round(acc_svm_bal, 4), "f1_weighted": round(f1_svm_bal, 4)},
            "logistic_regression": {"accuracy": round(acc_lr, 4), "f1_weighted": round(f1_lr, 4)}
        },
        "test_metrics": {
            "accuracy": round(acc, 4),
            "precision_macro": round(prec_macro, 4),
            "recall_macro": round(rec_macro, 4),
            "f1_macro": round(f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
            "confusion_matrix": cm,
            "classification_report": clf_rep_dict,
            "classification_report_text": clf_rep_text
        },
        "semantic_similarities": similarity_map,
        "pca_projection": pca_points,
        "sample_predictions": sample_reviews
    }

    with open(out_dir / "mcdonalds_w2v_precomputed.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved precomputed results to {out_dir / 'mcdonalds_w2v_precomputed.json'}")

    # Save models
    joblib.dump(svm, out_dir / "mcdonalds_svm_model.joblib")
    print(f"Saved SVM model to {out_dir / 'mcdonalds_svm_model.joblib'}")

    w2v_model.save(str(out_dir / "mcdonalds_w2v.model"))
    print(f"Saved Word2Vec model to {out_dir / 'mcdonalds_w2v.model'}")

    print(f"All processing finished in {time.time() - t_start:.2f}s!")


if __name__ == "__main__":
    main()
