"""
Tests for the prediction module.

Run: python -m pytest tests/test_predict.py -v
"""

import os
import glob
import numpy as np
import joblib
from src.predict import predict_text, load_model

CHECKPOINT_DIR = os.path.join("outputs", "checkpoints")


def _has_checkpoint():
    return len(glob.glob(os.path.join(CHECKPOINT_DIR, "best_model_*.joblib"))) > 0


class TestPrediction:
    """Tests that require a trained model checkpoint."""

    def setup_method(self):
        if not _has_checkpoint():
            from src.preprocess import load_data, build_tfidf, clean_text
            from sklearn.naive_bayes import MultinomialNB
            X_train, X_val, X_test, y_train, y_val, y_test, names = load_data()
            X_tc, tm = clean_text(X_train)
            X_vc, vm = clean_text(X_val)
            X_tec, tem = clean_text(X_test)
            Xt, Xv, Xte, vec = build_tfidf(X_tc, X_vc, X_tec, max_features=5000)
            m = MultinomialNB()
            m.fit(Xt, y_train[tm])
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            joblib.dump({"model": m, "vectorizer": vec, "target_names": names},
                        os.path.join(CHECKPOINT_DIR, "best_model_test.joblib"))

        self.model, self.vec, self.names, _ = load_model()

    def test_space_text_classified_correctly(self):
        result = predict_text(
            "NASA launched a satellite into orbit around Mars",
            self.model, self.vec, self.names,
        )
        assert result["predicted_class"] == "sci.space"

    def test_baseball_text_classified_correctly(self):
        result = predict_text(
            "The pitcher threw a fastball and struck out the batter",
            self.model, self.vec, self.names,
        )
        assert result["predicted_class"] == "rec.sport.baseball"

    def test_short_text_returns_error(self):
        result = predict_text("hi", self.model, self.vec, self.names)
        assert "error" in result

    def test_result_has_confidence_scores(self):
        result = predict_text(
            "New medical research on heart disease treatment",
            self.model, self.vec, self.names,
        )
        if "confidence" in result:
            assert len(result["confidence"]) == len(self.names)
            total = sum(result["confidence"].values())
            assert abs(total - 1.0) < 0.01

    def test_all_categories_predictable(self):
        texts = {
            "sci.space": "The spacecraft entered lunar orbit successfully",
            "sci.med": "Clinical trials show the new vaccine is effective",
            "rec.sport.baseball": "Home run in the bottom of the ninth inning",
            "talk.politics.guns": "Gun control legislation and second amendment rights",
        }
        for expected, text in texts.items():
            result = predict_text(text, self.model, self.vec, self.names)
            assert "predicted_class" in result, f"No prediction for {expected}"
