"""
Model Interpretability & Explainability Dashboard.

Generates publication-quality visualizations that explain *why* the classifier
makes the decisions it does — going beyond raw accuracy to build trust in the model.

Visualizations produced:
    1. Per-class discriminative features (horizontal bar charts)
    2. t-SNE document embedding (2-D scatter coloured by category)
    3. Confidence calibration curve (reliability diagram)
    4. Multi-metric model comparison radar chart
    5. Feature overlap heatmap across categories

Usage:
    python -m src.explainability              # generate all plots
    python -m src.explainability --tsne_only  # just the t-SNE embedding
"""

import argparse
import glob
import os
import time

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.preprocess import load_data, build_tfidf, clean_text, SEED

CHECKPOINT_DIR = os.path.join("outputs", "checkpoints")
FIGURES_DIR = os.path.join("outputs", "figures")

# ── Colour palette (colour-blind-friendly) ────────────────────────────────────
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Per-class discriminative features
# ─────────────────────────────────────────────────────────────────────────────
def plot_class_feature_importance(model, vectorizer, target_names, top_n=15):
    """Bar chart of the most discriminative TF-IDF features per category.

    For models with ``coef_`` (Logistic Regression, Linear SVM) the raw
    weight vector is used.  For Naive Bayes the log-probability difference
    relative to the corpus mean is used instead.
    """
    feature_names = vectorizer.get_feature_names_out()

    if hasattr(model, "coef_"):
        weights = model.coef_
    elif hasattr(model, "feature_log_prob_"):
        # NB: use log-prob relative to the mean across classes
        log_prob = model.feature_log_prob_
        weights = log_prob - log_prob.mean(axis=0)
    else:
        print("  Skipping feature importance — model type not supported.")
        return None

    n_classes = len(target_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 6), sharey=False)
    if n_classes == 1:
        axes = [axes]

    for idx, (ax, name) in enumerate(zip(axes, target_names)):
        top_idx = np.argsort(weights[idx])[-top_n:]
        top_feats = feature_names[top_idx]
        top_weights = weights[idx][top_idx]

        colour = PALETTE[idx % len(PALETTE)]
        ax.barh(range(top_n), top_weights, color=colour, edgecolor="white")
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_feats, fontsize=9)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Weight")
        ax.invert_yaxis()

    fig.suptitle("Top Discriminative Features per Category", fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 2. t-SNE document embedding
# ─────────────────────────────────────────────────────────────────────────────
def plot_tsne_embedding(X_tfidf, y, target_names, perplexity=30, sample_cap=1500):
    """2-D t-SNE scatter plot of TF-IDF vectors coloured by true category.

    A random subsample is taken when the dataset is large to keep runtime
    reasonable (t-SNE is O(n²)).
    """
    n = X_tfidf.shape[0]
    if n > sample_cap:
        idx = np.random.choice(n, sample_cap, replace=False)
        X_sub = X_tfidf[idx]
        y_sub = y[idx]
    else:
        X_sub = X_tfidf
        y_sub = y

    print(f"  Running t-SNE on {X_sub.shape[0]} samples (perplexity={perplexity})…")
    start = time.time()
    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=SEED,
        init="pca",
        learning_rate="auto",
    ).fit_transform(X_sub.toarray() if hasattr(X_sub, "toarray") else X_sub)
    elapsed = time.time() - start
    print(f"  t-SNE completed in {elapsed:.1f}s")

    fig, ax = plt.subplots(figsize=(9, 7))
    for label_idx, name in enumerate(target_names):
        mask = y_sub == label_idx
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=12,
            alpha=0.65,
            label=name,
            color=PALETTE[label_idx % len(PALETTE)],
        )

    ax.set_title("t-SNE Document Embedding (TF-IDF)", fontsize=13)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.legend(fontsize=10, markerscale=3, loc="best")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    path = os.path.join(FIGURES_DIR, "tsne_embedding.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 3. Confidence calibration curve
# ─────────────────────────────────────────────────────────────────────────────
def plot_calibration_curve(model, X_test, y_test, target_names, n_bins=10):
    """Reliability diagram: predicted probability vs actual fraction of positives.

    Only works for models that support ``predict_proba``.
    """
    if not hasattr(model, "predict_proba"):
        print("  Skipping calibration curve — model has no predict_proba.")
        return None

    proba = model.predict_proba(X_test)
    n_classes = len(target_names)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfectly calibrated")

    for idx, name in enumerate(target_names):
        y_binary = (y_test == idx).astype(int)
        prob_class = proba[:, idx]
        fraction_pos, mean_pred = calibration_curve(
            y_binary, prob_class, n_bins=n_bins, strategy="uniform"
        )
        ax.plot(
            mean_pred,
            fraction_pos,
            "o-",
            label=name,
            color=PALETTE[idx % len(PALETTE)],
            markersize=5,
        )

    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Fraction of Positives", fontsize=11)
    ax.set_title("Confidence Calibration Curve", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    path = os.path.join(FIGURES_DIR, "calibration_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 4. Multi-metric radar chart (model comparison)
# ─────────────────────────────────────────────────────────────────────────────
def _make_radar(ax, angles, values, label, colour, fill_alpha=0.12):
    """Draw a single polygon on a radar/spider chart."""
    values_closed = np.concatenate([values, [values[0]]])
    ax.plot(angles, values_closed, "o-", linewidth=2, label=label, color=colour)
    ax.fill(angles, values_closed, alpha=fill_alpha, color=colour)


def plot_model_comparison_radar(X_train, y_train, X_test, y_test, target_names, C=1.0):
    """Radar chart comparing Logistic Regression, Naive Bayes, and Linear SVM
    across accuracy, macro-F1, precision, recall, and inverse training time.
    """
    models = [
        ("Logistic Regression", LogisticRegression(C=C, max_iter=1000, random_state=SEED, solver="lbfgs")),
        ("Multinomial NB", MultinomialNB(alpha=1.0)),
        ("Linear SVM", LinearSVC(C=C, max_iter=2000, random_state=SEED)),
    ]

    metrics_names = ["Accuracy", "Macro-F1", "Precision", "Recall", "Speed"]
    all_scores = []
    raw_times = []

    for name, mdl in models:
        t0 = time.time()
        mdl.fit(X_train, y_train)
        elapsed = time.time() - t0
        raw_times.append(elapsed)

        y_pred = mdl.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        prec = precision_score(y_test, y_pred, average="macro")
        rec = recall_score(y_test, y_pred, average="macro")
        all_scores.append([acc, f1, prec, rec, elapsed])

    # Normalise speed: fastest → 1.0, slowest → 0.5 (inverted, so higher = faster)
    max_time = max(t[-1] for t in all_scores)
    for s in all_scores:
        s[-1] = 1.0 - 0.5 * (s[-1] / max_time) if max_time > 0 else 1.0

    n_metrics = len(metrics_names)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    colours = ["#4C72B0", "#DD8452", "#55A868"]

    for idx, ((name, _), scores) in enumerate(zip(models, all_scores)):
        _make_radar(ax, angles, np.array(scores), name, colours[idx])

    ax.set_thetagrids(np.degrees(angles[:-1]), metrics_names, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison (Radar Chart)", fontsize=13, pad=24)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.1), fontsize=9)
    fig.tight_layout()

    path = os.path.join(FIGURES_DIR, "model_comparison_radar.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 5. Feature overlap heatmap
# ─────────────────────────────────────────────────────────────────────────────
def plot_feature_overlap_heatmap(model, vectorizer, target_names, top_n=100):
    """Heatmap showing how much the top-N features overlap between classes.

    High overlap between two categories hints at *why* they are often confused.
    """
    feature_names = vectorizer.get_feature_names_out()

    if hasattr(model, "coef_"):
        weights = model.coef_
    elif hasattr(model, "feature_log_prob_"):
        log_prob = model.feature_log_prob_
        weights = log_prob - log_prob.mean(axis=0)
    else:
        print("  Skipping overlap heatmap — model type not supported.")
        return None

    n_classes = len(target_names)
    top_sets = []
    for idx in range(n_classes):
        top_idx = np.argsort(weights[idx])[-top_n:]
        top_sets.append(set(top_idx))

    overlap_matrix = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            intersection = len(top_sets[i] & top_sets[j])
            overlap_matrix[i, j] = intersection / top_n

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        overlap_matrix,
        annot=True,
        fmt=".0%",
        cmap="YlOrRd",
        xticklabels=target_names,
        yticklabels=target_names,
        vmin=0,
        vmax=1,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(f"Top-{top_n} Feature Overlap Between Categories", fontsize=13)
    ax.set_ylabel("Category")
    ax.set_xlabel("Category")
    fig.tight_layout()

    path = os.path.join(FIGURES_DIR, "feature_overlap_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────
def generate_all_explanations(
    model, vectorizer, target_names,
    X_train_tfidf, y_train, X_test_tfidf, y_test,
    tsne_only=False,
):
    """Generate all (or selected) explainability artefacts."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    paths = {}

    if tsne_only:
        paths["tsne"] = plot_tsne_embedding(X_test_tfidf, y_test, target_names)
        return paths

    print("\n[1/5] Per-class feature importance")
    paths["feature_importance"] = plot_class_feature_importance(
        model, vectorizer, target_names
    )

    print("\n[2/5] t-SNE document embedding")
    paths["tsne"] = plot_tsne_embedding(X_test_tfidf, y_test, target_names)

    print("\n[3/5] Confidence calibration curve")
    paths["calibration"] = plot_calibration_curve(
        model, X_test_tfidf, y_test, target_names
    )

    print("\n[4/5] Multi-metric radar chart")
    paths["radar"] = plot_model_comparison_radar(
        X_train_tfidf, y_train, X_test_tfidf, y_test, target_names
    )

    print("\n[5/5] Feature overlap heatmap")
    paths["overlap"] = plot_feature_overlap_heatmap(
        model, vectorizer, target_names
    )

    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Generate model explainability & interpretability dashboard"
    )
    parser.add_argument(
        "--tsne_only",
        action="store_true",
        help="Generate only the t-SNE embedding plot",
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=10000,
        help="Max TF-IDF features (must match training config)",
    )
    parser.add_argument(
        "--ngram_max",
        type=int,
        default=2,
        help="Max n-gram range (must match training config)",
    )
    args = parser.parse_args()

    np.random.seed(SEED)

    # ── Load data ──────────────────────────────────────────────────────────
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test, target_names = load_data()

    X_train_clean, train_mask = clean_text(X_train_raw)
    y_train_clean = y_train[train_mask]

    X_test_clean, test_mask = clean_text(X_test_raw)
    y_test_clean = y_test[test_mask]

    # ── Load checkpoint or re-vectorise ────────────────────────────────────
    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "best_model_*.joblib"))
    if checkpoints:
        latest = sorted(checkpoints)[-1]
        print(f"Loading checkpoint: {latest}")
        bundle = joblib.load(latest)
        model = bundle["model"]
        vectorizer = bundle["vectorizer"]

        X_train_tfidf = vectorizer.transform(X_train_clean)
        X_test_tfidf = vectorizer.transform(X_test_clean)
    else:
        print("No checkpoint found — building TF-IDF and training a default model.")
        X_val_clean, val_mask = clean_text(X_val_raw)

        X_train_tfidf, _, X_test_tfidf, vectorizer = build_tfidf(
            X_train_clean, X_val_clean, X_test_clean,
            max_features=args.max_features, ngram_max=args.ngram_max,
        )
        model = MultinomialNB(alpha=1.0)
        model.fit(X_train_tfidf, y_train_clean)

    # ── Generate explanations ──────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  MODEL EXPLAINABILITY DASHBOARD")
    print("=" * 55)

    paths = generate_all_explanations(
        model, vectorizer, target_names,
        X_train_tfidf, y_train_clean, X_test_tfidf, y_test_clean,
        tsne_only=args.tsne_only,
    )

    print("\n" + "=" * 55)
    generated = [p for p in paths.values() if p is not None]
    print(f"  Done! {len(generated)} figure(s) saved to {FIGURES_DIR}/")
    print("=" * 55)


if __name__ == "__main__":
    main()
