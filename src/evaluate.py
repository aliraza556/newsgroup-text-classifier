"""
Evaluation and error analysis for the trained newsgroup classifier.
Generates confusion matrix, identifies failure modes, and analyzes misclassifications.

Usage:
    python -m src.evaluate
"""

import glob
import os
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocess import load_data, clean_text, SEED

CHECKPOINT_DIR = os.path.join("outputs", "checkpoints")
FIGURES_DIR = os.path.join("outputs", "figures")


def plot_learning_curve(model, X_train, y_train, X_test, y_test, n_points=8):
    """Plot training and test accuracy as a function of training set size."""
    from sklearn.base import clone

    train_sizes = np.linspace(0.1, 1.0, n_points)
    train_scores = []
    test_scores = []

    for frac in train_sizes:
        n = max(10, int(frac * X_train.shape[0]))
        idx = np.random.choice(X_train.shape[0], n, replace=False)
        m = clone(model)
        m.fit(X_train[idx], y_train[idx])
        train_scores.append(accuracy_score(y_train[idx], m.predict(X_train[idx])))
        test_scores.append(accuracy_score(y_test, m.predict(X_test)))

    actual_sizes = [max(10, int(f * X_train.shape[0])) for f in train_sizes]

    plt.figure(figsize=(8, 5))
    plt.plot(actual_sizes, train_scores, "o-", label="Train", color="#2196F3")
    plt.plot(actual_sizes, test_scores, "s-", label="Test", color="#FF5722")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig_path = os.path.join(FIGURES_DIR, "learning_curve.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved: {fig_path}")


def error_analysis(y_true, y_pred, texts, target_names, vectorizer, model):
    """Identify and analyze misclassified examples."""
    misclassified_idx = np.where(y_true != y_pred)[0]
    print(f"\nTotal misclassified: {len(misclassified_idx)} / {len(y_true)} "
          f"({len(misclassified_idx)/len(y_true)*100:.1f}%)")

    # Group errors by (true_label -> predicted_label)
    error_pairs = {}
    for idx in misclassified_idx:
        true_label = target_names[y_true[idx]]
        pred_label = target_names[y_pred[idx]]
        key = f"{true_label} -> {pred_label}"
        if key not in error_pairs:
            error_pairs[key] = []
        error_pairs[key].append(idx)

    print("\nTop confusion pairs:")
    for pair, indices in sorted(error_pairs.items(), key=lambda x: -len(x[1]))[:5]:
        print(f"  {pair}: {len(indices)} errors")

    # Show one concrete misclassified example
    if len(misclassified_idx) > 0:
        worst_idx = misclassified_idx[0]
        true_name = target_names[y_true[worst_idx]]
        pred_name = target_names[y_pred[worst_idx]]
        text_snippet = texts[worst_idx][:300]
        print(f"\n--- Concrete failure example ---")
        print(f"True: {true_name}, Predicted: {pred_name}")
        print(f"Text: \"{text_snippet}...\"")

        # Show top features that pushed toward wrong prediction
        if hasattr(model, "coef_"):
            feature_names = vectorizer.get_feature_names_out()
            text_tfidf = vectorizer.transform([texts[worst_idx]])
            nonzero = text_tfidf.nonzero()[1]
            pred_class_idx = y_pred[worst_idx]
            feature_weights = []
            for feat_idx in nonzero:
                weight = model.coef_[pred_class_idx, feat_idx] * text_tfidf[0, feat_idx]
                feature_weights.append((feature_names[feat_idx], weight))
            feature_weights.sort(key=lambda x: -x[1])
            print(f"Top features pushing toward '{pred_name}':")
            for feat, w in feature_weights[:5]:
                print(f"  {feat}: {w:.3f}")


def main():
    np.random.seed(SEED)

    # Load data
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test, target_names = load_data()

    # Load best checkpoint
    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "best_model_*.joblib"))
    if not checkpoints:
        print("No checkpoint found. Run 'python -m src.train' first.")
        return

    latest = sorted(checkpoints)[-1]
    print(f"Loading checkpoint: {latest}")
    bundle = joblib.load(latest)
    model = bundle["model"]
    vectorizer = bundle["vectorizer"]

    # Prepare test data
    X_test_clean, test_mask = clean_text(X_test_raw)
    y_test_clean = y_test[test_mask]
    X_test_tfidf = vectorizer.transform(X_test_clean)

    # Evaluate on test set
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test_clean, y_pred)
    f1 = f1_score(y_test_clean, y_pred, average="macro")

    print(f"\n=== TEST SET RESULTS ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test_clean, y_pred, target_names=target_names))

    # Confusion matrix
    cm = confusion_matrix(y_test_clean, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names)
    plt.title("Confusion Matrix (Test Set)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig_path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
    plt.savefig(fig_path, dpi=150)
    print(f"Saved: {fig_path}")

    # Error analysis
    error_analysis(y_test_clean, y_pred, X_test_clean, target_names, vectorizer, model)

    # Check for overfitting
    X_train_clean, train_mask = clean_text(X_train_raw)
    y_train_clean = y_train[train_mask]
    X_train_tfidf = vectorizer.transform(X_train_clean)
    train_acc = accuracy_score(y_train_clean, model.predict(X_train_tfidf))

    print(f"\n=== OVERFITTING CHECK ===")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {acc:.4f}")
    gap = train_acc - acc
    print(f"Gap: {gap:.4f}")
    if gap > 0.05:
        print("Warning: possible overfitting (gap > 5%)")
    else:
        print("No significant overfitting observed.")

    # Learning curve: accuracy vs training set size
    plot_learning_curve(model, X_train_tfidf, y_train_clean, X_test_tfidf, y_test_clean)


if __name__ == "__main__":
    main()
