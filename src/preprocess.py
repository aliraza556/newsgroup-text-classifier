"""
Data loading and TF-IDF preprocessing for newsgroup text classification.
Uses a 4-class subset to keep error analysis interpretable.
"""

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


CATEGORIES = ["sci.med", "sci.space", "rec.sport.baseball", "talk.politics.guns"]

SEED = 42


def load_data():
    """Load 4-class subset and split into train/val/test (70/15/15)."""
    np.random.seed(SEED)

    raw = fetch_20newsgroups(
        subset="all",
        categories=CATEGORIES,
        remove=("headers", "footers", "quotes"),
        random_state=SEED,
    )

    texts, labels = raw.data, raw.target
    target_names = raw.target_names

    # First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.30, random_state=SEED, stratify=labels
    )

    # Second split: 50/50 of temp -> 15% val, 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Classes: {target_names}")

    return X_train, X_val, X_test, y_train, y_val, y_test, target_names


def build_tfidf(X_train, X_val, X_test, max_features=10000, ngram_max=2):
    """
    Fit TF-IDF on training set only, then transform val and test.
    We fit only on train to avoid data leakage from val/test vocabulary.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, ngram_max),
        stop_words="english",
        min_df=2,
        sublinear_tf=True,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"TF-IDF shape: {X_train_tfidf.shape} (features={max_features}, ngram=1-{ngram_max})")

    return X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer


def clean_text(texts):
    """
    Basic text cleaning: lowercase, strip whitespace, remove empty docs.
    We keep it minimal because TfidfVectorizer handles tokenization.
    Returns cleaned texts and a boolean numpy mask for label filtering.
    """
    cleaned = []
    kept_mask = []
    for t in texts:
        t = t.strip().lower()
        if len(t) > 10:
            cleaned.append(t)
            kept_mask.append(True)
        else:
            kept_mask.append(False)

    mask_array = np.array(kept_mask)
    removed_count = int((~mask_array).sum())
    if removed_count > 0:
        print(f"Removed {removed_count} near-empty documents during cleaning")

    return cleaned, mask_array
