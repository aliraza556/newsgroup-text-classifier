"""
Unit tests for data loading and preprocessing pipeline.
Verifies data integrity, split ratios, and TF-IDF output shapes.

Run: python -m pytest tests/ -v
"""

import numpy as np
from src.preprocess import load_data, build_tfidf, clean_text, CATEGORIES, SEED


def test_load_data_split_sizes():
    """Check that train/val/test split ratios are approximately 70/15/15."""
    X_train, X_val, X_test, y_train, y_val, y_test, names = load_data()
    total = len(X_train) + len(X_val) + len(X_test)
    assert 0.65 < len(X_train) / total < 0.75, "Train split should be ~70%"
    assert 0.12 < len(X_val) / total < 0.18, "Val split should be ~15%"
    assert 0.12 < len(X_test) / total < 0.18, "Test split should be ~15%"


def test_load_data_classes():
    """Ensure all 4 target categories are present."""
    _, _, _, y_train, _, _, names = load_data()
    assert len(names) == len(CATEGORIES)
    assert set(np.unique(y_train)) == {0, 1, 2, 3}


def test_tfidf_no_data_leakage():
    """TF-IDF should be fit on train only; val/test must have same feature count."""
    X_train, X_val, X_test, y_train, y_val, y_test, _ = load_data()
    Xt, Xv, Xte, vec = build_tfidf(X_train, X_val, X_test, max_features=500)
    assert Xt.shape[1] == Xv.shape[1] == Xte.shape[1]
    assert Xt.shape[1] <= 500


def test_clean_text_removes_empty():
    """Clean text should remove near-empty documents."""
    texts = ["This is a valid document with content", "", "short", "Another valid document here"]
    cleaned, mask = clean_text(texts)
    assert len(cleaned) == 2
    assert np.array_equal(mask, np.array([True, False, False, True]))


def test_reproducibility():
    """Same seed should produce identical splits."""
    X1, _, _, y1, _, _, _ = load_data()
    X2, _, _, y2, _, _, _ = load_data()
    assert X1 == X2
    assert np.array_equal(y1, y2)


if __name__ == "__main__":
    print("Running tests...")
    test_load_data_split_sizes()
    print("  test_load_data_split_sizes PASSED")
    test_load_data_classes()
    print("  test_load_data_classes PASSED")
    test_tfidf_no_data_leakage()
    print("  test_tfidf_no_data_leakage PASSED")
    test_clean_text_removes_empty()
    print("  test_clean_text_removes_empty PASSED")
    test_reproducibility()
    print("  test_reproducibility PASSED")
    print("\nAll tests passed!")
