"""
Tests for the explainability module.

Verifies that each visualisation function runs without error and produces
a valid output path.  Uses small synthetic data to keep tests fast.

Run: python -m pytest tests/test_explainability.py -v
"""

import os
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

from src.explainability import (
    plot_class_feature_importance,
    plot_tsne_embedding,
    plot_calibration_curve,
    plot_feature_overlap_heatmap,
)

FIGURES_DIR = os.path.join("outputs", "figures")
TARGET_NAMES = ["cat_a", "cat_b"]


@pytest.fixture(scope="module")
def tiny_model_and_data():
    """Train a small NB model on synthetic texts for fast testing."""
    texts = [
        "alpha bravo charlie delta echo",
        "foxtrot golf hotel india juliet",
        "alpha bravo kilo lima mike",
        "foxtrot golf november oscar papa",
    ] * 10  # repeat to give enough samples

    labels = np.array([0, 1, 0, 1] * 10)

    vectorizer = TfidfVectorizer(max_features=50)
    X = vectorizer.fit_transform(texts)

    model = MultinomialNB()
    model.fit(X, labels)

    return model, vectorizer, X, labels


class TestFeatureImportance:
    def test_produces_file(self, tiny_model_and_data, tmp_path):
        model, vec, _, _ = tiny_model_and_data
        # Temporarily redirect output to tmp_path
        original = os.path.join("outputs", "figures")
        os.makedirs(original, exist_ok=True)
        path = plot_class_feature_importance(model, vec, TARGET_NAMES, top_n=5)
        assert path is not None
        assert os.path.isfile(path)


class TestTSNE:
    def test_produces_file(self, tiny_model_and_data):
        _, _, X, y = tiny_model_and_data
        os.makedirs(FIGURES_DIR, exist_ok=True)
        path = plot_tsne_embedding(X, y, TARGET_NAMES, perplexity=5, sample_cap=40)
        assert path is not None
        assert os.path.isfile(path)


class TestCalibrationCurve:
    def test_produces_file(self, tiny_model_and_data):
        model, _, X, y = tiny_model_and_data
        os.makedirs(FIGURES_DIR, exist_ok=True)
        path = plot_calibration_curve(model, X, y, TARGET_NAMES, n_bins=3)
        assert path is not None
        assert os.path.isfile(path)

    def test_skips_for_svm(self, tiny_model_and_data):
        """LinearSVC has no predict_proba — function should return None."""
        from sklearn.svm import LinearSVC

        _, vec, X, y = tiny_model_and_data
        svm = LinearSVC(max_iter=500)
        svm.fit(X, y)
        result = plot_calibration_curve(svm, X, y, TARGET_NAMES)
        assert result is None


class TestFeatureOverlap:
    def test_produces_file(self, tiny_model_and_data):
        model, vec, _, _ = tiny_model_and_data
        os.makedirs(FIGURES_DIR, exist_ok=True)
        path = plot_feature_overlap_heatmap(model, vec, TARGET_NAMES, top_n=10)
        assert path is not None
        assert os.path.isfile(path)
