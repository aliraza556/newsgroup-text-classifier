"""
Interactive prediction CLI for the trained newsgroup classifier.
Classifies custom text input and shows confidence scores with top contributing features.

Usage:
    python -m src.predict "Your text here"
    python -m src.predict --interactive
"""

import argparse
import glob
import os
import sys

import numpy as np
import joblib

from src.preprocess import SEED

CHECKPOINT_DIR = os.path.join("outputs", "checkpoints")


def load_model():
    """Load the best saved model checkpoint."""
    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "best_model_*.joblib"))
    if not checkpoints:
        print("No checkpoint found. Run 'python -m src.train' first.")
        sys.exit(1)

    latest = sorted(checkpoints)[-1]
    bundle = joblib.load(latest)
    return bundle["model"], bundle["vectorizer"], bundle["target_names"], latest


def predict_text(text, model, vectorizer, target_names, show_features=True):
    """Classify a single text and return prediction details."""
    cleaned = text.strip().lower()
    if len(cleaned) <= 10:
        return {"error": "Text too short (must be >10 characters)"}

    X = vectorizer.transform([cleaned])
    prediction = model.predict(X)[0]
    predicted_class = target_names[prediction]

    result = {
        "text_preview": cleaned[:120] + ("..." if len(cleaned) > 120 else ""),
        "predicted_class": predicted_class,
    }

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        result["confidence"] = {
            target_names[i]: round(float(p), 4) for i, p in enumerate(proba)
        }
        result["top_confidence"] = round(float(proba.max()), 4)

    if show_features and hasattr(model, "coef_"):
        feature_names = vectorizer.get_feature_names_out()
        nonzero = X.nonzero()[1]
        weights = []
        for feat_idx in nonzero:
            w = model.coef_[prediction, feat_idx] * X[0, feat_idx]
            weights.append((feature_names[feat_idx], float(w)))
        weights.sort(key=lambda x: -x[1])
        result["top_features"] = weights[:8]

    return result


def format_result(result):
    """Pretty-print a prediction result."""
    if "error" in result:
        return f"Error: {result['error']}"

    lines = [
        f"  Input:      {result['text_preview']}",
        f"  Prediction: {result['predicted_class']}",
    ]

    if "top_confidence" in result:
        lines.append(f"  Confidence: {result['top_confidence']:.1%}")
        lines.append("  Class probabilities:")
        for cls, prob in sorted(result["confidence"].items(), key=lambda x: -x[1]):
            bar = "#" * int(prob * 30)
            lines.append(f"    {cls:25s} {prob:.3f} {bar}")

    if "top_features" in result and result["top_features"]:
        lines.append("  Top contributing features:")
        for feat, w in result["top_features"][:5]:
            lines.append(f"    {feat:20s} {w:+.3f}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Predict newsgroup category for custom text")
    parser.add_argument("text", nargs="?", help="Text to classify")
    parser.add_argument("--interactive", action="store_true", help="Enter interactive mode")
    args = parser.parse_args()

    model, vectorizer, target_names, ckpt = load_model()
    print(f"Model loaded: {os.path.basename(ckpt)}")
    print(f"Categories: {', '.join(target_names)}\n")

    if args.interactive:
        print("Interactive mode (type 'quit' to exit)")
        print("-" * 50)
        while True:
            try:
                text = input("\nEnter text> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text.lower() in ("quit", "exit", "q"):
                break
            if not text:
                continue
            result = predict_text(text, model, vectorizer, target_names)
            print(format_result(result))
    elif args.text:
        result = predict_text(args.text, model, vectorizer, target_names)
        print(format_result(result))
    else:
        examples = [
            "NASA launched a new satellite into orbit around Mars yesterday",
            "The pitcher threw a fastball and struck out the batter in the ninth inning",
            "New research shows that this drug can reduce heart disease risk by 30 percent",
            "The second amendment protects the right to bear arms and own firearms",
        ]
        print("No text provided. Running on built-in examples:\n")
        for ex in examples:
            result = predict_text(ex, model, vectorizer, target_names)
            print(format_result(result))
            print()


if __name__ == "__main__":
    main()
