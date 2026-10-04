"""
Processing of textual data - Exercise 2
=========================================
1. Load data from imdb_reviews_preprocessed_1.csv into `imdb_reviews`.
2. Write a function that takes cleaned review text as an argument, and returns lemmas.
3. Add review_lemmatized column using .apply().
4. Save DataFrame to imdb_reviews_preprocessed_2.csv.
5. Generate rich precomputed stats into nlp_exercise_2_precomputed.json.
"""

from collections import Counter
import json
from pathlib import Path
import time
import pandas as pd
import spacy

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
NLP_DATA_DIR = Path(__file__).resolve().parent / "data"
NLP_DATA_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = DATA_DIR / "imdb_reviews_preprocessed_1.csv"
OUTPUT_CSV = DATA_DIR / "imdb_reviews_preprocessed_2.csv"
JSON_OUTPUT = NLP_DATA_DIR / "nlp_exercise_2_precomputed.json"


def main():
    print(f"Loading data from: {INPUT_CSV}")
    t0 = time.time()
    imdb_reviews = pd.read_csv(INPUT_CSV)
    load_time = time.time() - t0
    print(f"Loaded {len(imdb_reviews)} rows in {load_time:.2f}s.")

    # Initialize spaCy model (disabling parser and ner for high speed)
    print("Initializing spaCy model ('en_core_web_sm')...")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    # Step 2: Define lemmatization function
    def lemmatize_text(text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""
        doc = nlp(text)
        return " ".join([token.lemma_ for token in doc if not token.is_space])

    # Step 3: Add review_lemmatized column via .apply()
    print("Lemmatizing reviews using .apply(lemmatize_text)...")
    t1 = time.time()
    imdb_reviews["review_lemmatized"] = imdb_reviews["review_cleaned"].apply(lemmatize_text)
    apply_time = time.time() - t1
    print(f"Lemmatization completed in {apply_time:.2f}s ({len(imdb_reviews)/apply_time:.1f} reviews/s).")

    # Step 4: Save to CSV
    print(f"Saving to CSV: {OUTPUT_CSV}")
    imdb_reviews.to_csv(OUTPUT_CSV, index=False)
    print("Saved successfully.")

    # Step 5: Compute analytics and comparison metrics
    print("Computing metrics and vocabulary statistics...")
    cleaned_all_words = []
    lemmatized_all_words = []
    lemma_transformations = []  # pairs (original, lemma) where original != lemma

    sample_comparisons = []
    for idx in range(min(15, len(imdb_reviews))):
        c_text = str(imdb_reviews["review_cleaned"].iloc[idx])
        l_text = str(imdb_reviews["review_lemmatized"].iloc[idx])
        c_tokens = c_text.split()
        l_tokens = l_text.split()
        diffs = [(c, l) for c, l in zip(c_tokens, l_tokens) if c != l]
        sample_comparisons.append({
            "index": idx,
            "sentiment": imdb_reviews["sentiment"].iloc[idx],
            "review_cleaned_preview": c_text[:200] + ("..." if len(c_text) > 200 else ""),
            "review_lemmatized_preview": l_text[:200] + ("..." if len(l_text) > 200 else ""),
            "cleaned_word_count": len(c_tokens),
            "lemmatized_word_count": len(l_tokens),
            "transformed_tokens_count": len(diffs),
            "sample_transformations": diffs[:6]
        })

    # Sample a representative slice for global lemma transformation counts
    sample_df = imdb_reviews.sample(n=min(2000, len(imdb_reviews)), random_state=42)
    for c_text, l_text in zip(sample_df["review_cleaned"], sample_df["review_lemmatized"]):
        c_toks = str(c_text).split()
        l_toks = str(l_text).split()
        cleaned_all_words.extend(c_toks)
        lemmatized_all_words.extend(l_toks)
        for c, l in zip(c_toks, l_toks):
            if c != l:
                lemma_transformations.append(f"{c} -> {l}")

    vocab_cleaned = set(cleaned_all_words)
    vocab_lemmatized = set(lemmatized_all_words)

    counter_cleaned = Counter(cleaned_all_words)
    counter_lemmatized = Counter(lemmatized_all_words)
    counter_transformations = Counter(lemma_transformations)

    top_freq_cleaned = counter_cleaned.most_common(20)
    top_freq_lemmatized = counter_lemmatized.most_common(20)
    top_transformations = counter_transformations.most_common(25)

    stats = {
        "metadata": {
            "exercise": "Processing of textual data - exercise 2",
            "input_csv": str(INPUT_CSV.name),
            "output_csv": str(OUTPUT_CSV.name),
            "total_rows": len(imdb_reviews),
            "execution_time_seconds": round(apply_time, 2),
            "throughput_reviews_per_sec": round(len(imdb_reviews) / apply_time, 1)
        },
        "vocabulary_metrics": {
            "sample_size_for_vocab": len(sample_df),
            "sample_cleaned_total_tokens": len(cleaned_all_words),
            "sample_lemmatized_total_tokens": len(lemmatized_all_words),
            "unique_vocab_cleaned": len(vocab_cleaned),
            "unique_vocab_lemmatized": len(vocab_lemmatized),
            "vocab_reduction_count": len(vocab_cleaned) - len(vocab_lemmatized),
            "vocab_reduction_pct": round((1.0 - len(vocab_lemmatized) / len(vocab_cleaned)) * 100, 2)
        },
        "top_frequencies": {
            "cleaned": [{"word": w, "count": c} for w, c in top_freq_cleaned],
            "lemmatized": [{"word": w, "count": c} for w, c in top_freq_lemmatized]
        },
        "top_lemma_transformations": [
            {"transformation": pair, "count": c} for pair, c in top_transformations
        ],
        "samples": sample_comparisons
    }

    with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Precomputed stats saved to: {JSON_OUTPUT}")


if __name__ == "__main__":
    main()
