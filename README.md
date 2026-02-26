# Newsgroup Text Classifier

Text classification on the **20 Newsgroups** dataset using TF-IDF features and classical ML models (Logistic Regression, Naive Bayes, Linear SVM).

## Dataset

Uses a 4-class subset of the [20 Newsgroups](http://qwone.com/~jason/20Newsgroups/) corpus (built into scikit-learn):
- `sci.med`, `sci.space`, `rec.sport.baseball`, `talk.politics.guns`
- ~3,500 samples total, split 70/15/15 (train/val/test) with stratification
- Headers, footers, and quotes removed to prevent metadata leakage

## Approach

1. **Preprocessing**: lowercase, remove near-empty docs, TF-IDF vectorization (fit on train only to avoid data leakage)
2. **Model selection**: compare Logistic Regression, Multinomial NB, Linear SVM on validation macro-F1
3. **Evaluation**: confusion matrix, per-class precision/recall, concrete error analysis with feature attribution

## Reproducibility

- All random seeds fixed (`SEED=42`, numpy, sklearn, `PYTHONHASHSEED`)
- Deterministic train/val/test split via `stratify` + `random_state`
- Experiment logs saved as JSON in `outputs/logs/`

## Usage

```bash
pip install -r requirements.txt

# Train (compares 3 models, saves best checkpoint)
python -m src.train --lr 1.0 --max_features 10000 --ngram_max 2

# Evaluate on test set (confusion matrix + error analysis)
python -m src.evaluate

# Run unit tests
python -m pytest tests/ -v
```

## Project Structure

```
newsgroup-text-classifier/
├── src/                          # Source code
│   ├── __init__.py
│   ├── preprocess.py             # Data loading, cleaning, TF-IDF vectorization
│   ├── train.py                  # Training loop with model comparison
│   └── evaluate.py               # Test evaluation, confusion matrix, error analysis
├── tests/                        # Unit tests
│   └── test_data.py              # Tests for data pipeline integrity
├── outputs/                      # Generated artifacts
│   ├── checkpoints/              # Saved model checkpoints (.joblib)
│   ├── logs/                     # Experiment logs (JSON)
│   └── figures/                  # Plots (confusion matrix, etc.)
├── requirements.txt              # Python dependencies (pip)
├── README.md
└── .gitignore
```

## Results

Best model: **Multinomial Naive Bayes** (selected by validation macro-F1)

| Metric | Validation | Test |
|--------|-----------|------|
| Accuracy | 93.56% | 93.79% |
| Macro F1 | 0.9352 | 0.9378 |

Train-test accuracy gap: 4.1% (no significant overfitting).

Top confusion pair: `sci.space -> sci.med` (9 errors) — both categories share medical/scientific vocabulary.

## Environment

- Python 3.10+
- pip (see `requirements.txt`)
- CPU only (no GPU required)
