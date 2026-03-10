"""
Training script for newsgroup text classifier.
Trains Logistic Regression with TF-IDF features on a 4-class newsgroup subset.
Supports optional k-fold cross-validation for robust model selection.

Usage:
    python -m src.train --lr 1.0 --max_features 10000 --ngram_max 2
    python -m src.train --cross_validate --cv_folds 5
"""

import argparse
import json
import time
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score
import joblib

from src.preprocess import load_data, build_tfidf, clean_text, SEED

CHECKPOINT_DIR = os.path.join("outputs", "checkpoints")
LOG_DIR = os.path.join("outputs", "logs")


def set_all_seeds(seed=SEED):
    """Set seeds for reproducibility across numpy and sklearn."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def train_and_evaluate(model, X_train, y_train, X_val, y_val, model_name):
    """Train a model and return validation metrics."""
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="macro")

    print(f"\n--- {model_name} ---")
    print(f"  Val Accuracy: {acc:.4f}")
    print(f"  Val Macro-F1: {f1:.4f}")
    print(f"  Train time:   {train_time:.2f}s")

    return {"model_name": model_name, "accuracy": acc, "macro_f1": f1, "train_time": train_time}


def run_cross_validation(models, X_train, y_train, cv_folds, scoring="f1_macro"):
    """Run k-fold cross-validation on all models and print comparison."""
    print(f"\n{'='*55}")
    print(f"  K-FOLD CROSS-VALIDATION (k={cv_folds}, scoring={scoring})")
    print(f"{'='*55}")

    cv_results = []
    for name, model in models:
        scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=scoring)
        mean, std = scores.mean(), scores.std()
        print(f"\n  {name}:")
        print(f"    Fold scores: {[f'{s:.4f}' for s in scores]}")
        print(f"    Mean: {mean:.4f} (+/- {std:.4f})")
        cv_results.append({
            "model_name": name,
            "cv_folds": cv_folds,
            "fold_scores": scores.tolist(),
            "cv_mean": round(mean, 4),
            "cv_std": round(std, 4),
        })

    best_cv = max(cv_results, key=lambda r: r["cv_mean"])
    print(f"\n  Best by CV: {best_cv['model_name']} (mean {scoring}={best_cv['cv_mean']:.4f})")
    return cv_results


def main():
    parser = argparse.ArgumentParser(description="Train newsgroup text classifier")
    parser.add_argument("--lr", type=float, default=1.0, help="Logistic Regression C parameter (inverse regularization)")
    parser.add_argument("--max_features", type=int, default=10000, help="Max TF-IDF features")
    parser.add_argument("--ngram_max", type=int, default=2, help="Max n-gram range (1=unigrams, 2=bigrams)")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--cross_validate", action="store_true", help="Run k-fold cross-validation")
    parser.add_argument("--cv_folds", type=int, default=5, help="Number of CV folds (default: 5)")
    args = parser.parse_args()

    set_all_seeds(args.seed)
    print(f"Seed: {args.seed}")
    print(f"Hyperparameters: C={args.lr}, max_features={args.max_features}, ngram_max={args.ngram_max}")

    # Load and preprocess
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test, target_names = load_data()

    X_train_clean, train_mask = clean_text(X_train_raw)
    y_train_clean = y_train[train_mask]

    X_val_clean, val_mask = clean_text(X_val_raw)
    y_val_clean = y_val[val_mask]

    X_test_clean, test_mask = clean_text(X_test_raw)
    y_test_clean = y_test[test_mask]

    X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer = build_tfidf(
        X_train_clean, X_val_clean, X_test_clean,
        max_features=args.max_features, ngram_max=args.ngram_max,
    )

    # Compare three models
    results = []

    # 1) Logistic Regression (primary model)
    lr_model = LogisticRegression(
        C=args.lr, max_iter=1000, random_state=args.seed, solver="lbfgs"
    )
    lr_result = train_and_evaluate(lr_model, X_train_tfidf, y_train_clean, X_val_tfidf, y_val_clean, "LogisticRegression")
    results.append(lr_result)

    # 2) Multinomial Naive Bayes (baseline)
    nb_model = MultinomialNB(alpha=1.0)
    nb_result = train_and_evaluate(nb_model, X_train_tfidf, y_train_clean, X_val_tfidf, y_val_clean, "MultinomialNB")
    results.append(nb_result)

    # 3) Linear SVM (alternative)
    svm_model = LinearSVC(C=args.lr, max_iter=2000, random_state=args.seed)
    svm_result = train_and_evaluate(svm_model, X_train_tfidf, y_train_clean, X_val_tfidf, y_val_clean, "LinearSVC")
    results.append(svm_result)

    # Optional: k-fold cross-validation for robust comparison
    cv_results = None
    if args.cross_validate:
        model_list = [
            ("LogisticRegression", LogisticRegression(C=args.lr, max_iter=1000, random_state=args.seed, solver="lbfgs")),
            ("MultinomialNB", MultinomialNB(alpha=1.0)),
            ("LinearSVC", LinearSVC(C=args.lr, max_iter=2000, random_state=args.seed)),
        ]
        cv_results = run_cross_validation(model_list, X_train_tfidf, y_train_clean, args.cv_folds)

    # Select best model by macro F1
    best = max(results, key=lambda r: r["macro_f1"])
    print(f"\n=== Best model: {best['model_name']} (F1={best['macro_f1']:.4f}) ===")

    # Full classification report for best model
    if best["model_name"] == "LogisticRegression":
        best_model = lr_model
    elif best["model_name"] == "MultinomialNB":
        best_model = nb_model
    else:
        best_model = svm_model

    y_val_pred = best_model.predict(X_val_tfidf)
    print("\nClassification Report (validation set):")
    print(classification_report(y_val_clean, y_val_pred, target_names=target_names))

    # Save model and vectorizer
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_model_{best['model_name']}_f1_{best['macro_f1']:.4f}.joblib")
    joblib.dump({"model": best_model, "vectorizer": vectorizer, "target_names": target_names}, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

    # Save training log
    log = {
        "seed": args.seed,
        "C": args.lr,
        "max_features": args.max_features,
        "ngram_max": args.ngram_max,
        "results": results,
        "best_model": best["model_name"],
        "best_val_accuracy": best["accuracy"],
        "best_val_macro_f1": best["macro_f1"],
        "checkpoint": checkpoint_path,
    }
    if cv_results is not None:
        log["cross_validation"] = cv_results
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"train_log_C{args.lr}_feat{args.max_features}.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Log saved: {log_path}")

    # Final validation log line
    print(f"\n[FINAL] val_accuracy={best['accuracy']:.4f} val_macro_f1={best['macro_f1']:.4f} "
          f"model={best['model_name']} checkpoint={checkpoint_path}")


if __name__ == "__main__":
    main()
