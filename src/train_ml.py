"""
Machine Learning Model Training Pipeline (src/train_ml.py)
Trains classic ML models using BOTH Bag-of-Words (BoW) and TF-IDF representations:
- Multinomial Naive Bayes
- Logistic Regression
- Linear SVM (LinearSVC / SVC linear)
- SVM with RBF kernel (SVC rbf)
- Random Forest Classifier

Uses GridSearchCV with 5-fold Stratified Cross-Validation on training data.
Evaluates on validation split, summarizes metrics, and saves best models with joblib.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from tabulate import tabulate

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import load_dataset
from src.preprocess import Preprocessor, stratified_split_70_15_15, get_class_weights
from src.feature_extraction import ComparativeFeatureExtractor

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def get_model_search_spaces(class_weights: Dict[int, float]) -> Dict[str, Tuple[Any, Dict[str, List[Any]]]]:
    """
    Define model architectures and their respective GridSearchCV search grids.
    """
    return {
        "multinomial_nb": (
            MultinomialNB(),
            {
                "alpha": [0.01, 0.1, 0.5, 1.0, 2.0]
            }
        ),
        "logistic_regression": (
            LogisticRegression(max_iter=1000, random_state=42),
            {
                "C": [0.1, 1.0, 10.0, 50.0],
                "class_weight": ["balanced", None]
            }
        ),
        "linear_svm": (
            LinearSVC(dual=False, max_iter=2000, random_state=42),
            {
                "C": [0.01, 0.1, 1.0, 10.0],
                "class_weight": ["balanced", None]
            }
        ),
        "svm_rbf": (
            SVC(kernel="rbf", probability=True, random_state=42),
            {
                "C": [0.5, 1.0, 10.0],
                "gamma": ["scale", "auto"]
            }
        ),
        "random_forest": (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {
                "n_estimators": [100, 200],
                "max_depth": [15, 30, None],
                "min_samples_split": [2, 5]
            }
        )
    }


def evaluate_model(
    model: Any,
    X: Any,
    y: np.ndarray,
    target_names: List[str] = ["ham", "spam"]
) -> Dict[str, float]:
    """
    Evaluate fitted estimator on given feature matrix and ground-truth targets.
    """
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, average="binary", zero_division=0)
    rec = recall_score(y, preds, average="binary", zero_division=0)
    f1 = f1_score(y, preds, average="binary", zero_division=0)

    # ROC AUC calculation if probabilities or decision function available
    roc_auc = 0.5
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1]
            roc_auc = roc_auc_score(y, probs)
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X)
            roc_auc = roc_auc_score(y, scores)
    except Exception:
        roc_auc = 0.5

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
    }


def train_ml_pipeline(
    dataset_name: str = "sms",
    cv_folds: int = 5,
    quick_run: bool = False
) -> pd.DataFrame:
    """
    Complete ML training benchmark loop:
    1. Loads dataset
    2. Cleans text & extracts 70/15/15 stratified splits
    3. Fits BoW and TF-IDF vectorizers on X_train only
    4. Runs GridSearchCV with 5-fold CV for each model under both BoW and TF-IDF
    5. Saves best estimators to models/ directory
    6. Saves comparative metrics to reports/metrics_summary.csv
    """
    print(f"\n=======================================================")
    print(f"Starting ML Training & Benchmark on Dataset: '{dataset_name}'")
    print(f"=======================================================\n")

    # 1. Load Data
    raw_df = load_dataset(dataset_name)
    print(f"Dataset loaded: {len(raw_df)} total records.")

    # 2. Preprocess & Clean Text
    preprocessor = Preprocessor()
    print("Preprocessing text (lowercasing, cleaning, stopword removal, stemming)...")
    raw_df["clean_text"] = preprocessor.preprocess_series(raw_df["text"])

    # 3. Stratified 70/15/15 Split
    train_df, val_df, test_df = stratified_split_70_15_15(raw_df, target_col="target")
    print(f"Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    X_train_text = train_df["clean_text"].values
    y_train = train_df["target"].values

    X_val_text = val_df["clean_text"].values
    y_val = val_df["target"].values

    # 4. Fit Feature Extractors on Train Only
    print("\nFitting Comparative Feature Extractor on training data (BoW + TF-IDF)...")
    feature_extractor = ComparativeFeatureExtractor(
        max_features=5000,
        ngram_range=(1, 2)
    )
    X_train_bow, X_train_tfidf = feature_extractor.fit_transform_train(X_train_text)
    X_val_bow, X_val_tfidf = feature_extractor.transform_eval(X_val_text)

    # Save fitted vectorizers
    feature_extractor.save(MODELS_DIR)
    print(f"Saved vectorizers to {MODELS_DIR}")

    representations = {
        "bow": (X_train_bow, X_val_bow),
        "tfidf": (X_train_tfidf, X_val_tfidf)
    }

    class_weights = get_class_weights(y_train)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    all_results = []

    # 5. Train Each Model on Both Representations
    for rep_name, (X_tr, X_va) in representations.items():
        print(f"\n" + "#" * 60)
        print(f"   REPRESENTATION: {rep_name.upper()}")
        print("#" * 60)

        model_spaces = get_model_search_spaces(class_weights)

        # In quick_run mode or for heavy estimators like svm_rbf, constrain params if needed
        if quick_run:
            model_spaces["svm_rbf"] = (SVC(kernel="rbf", probability=True, random_state=42), {"C": [1.0]})
            model_spaces["random_forest"] = (RandomForestClassifier(n_estimators=100, random_state=42), {"max_depth": [30]})

        for model_name, (estimator, param_grid) in model_spaces.items():
            print(f"\n--> Training {model_name.upper()} with {rep_name.upper()} via GridSearchCV (5-fold CV)...")
            start_time = time.time()

            grid_search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                cv=cv,
                scoring="f1",
                n_jobs=-1,
                refit=True
            )
            grid_search.fit(X_tr, y_train)
            train_duration = time.time() - start_time

            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_cv_f1 = grid_search.best_score_

            # Evaluate on Validation set
            val_metrics = evaluate_model(best_model, X_va, y_val)

            print(f"    Done in {train_duration:.2f}s | Best CV F1: {best_cv_f1:.4f} | Best Params: {best_params}")
            print(f"    Validation => Accuracy: {val_metrics['accuracy']:.4f} | Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f} | F1: {val_metrics['f1_score']:.4f}")

            # Save model with joblib
            model_filename = f"{rep_name}_{model_name}.joblib"
            model_path = MODELS_DIR / model_filename
            joblib.dump(best_model, model_path)

            all_results.append({
                "representation": rep_name.upper(),
                "model": model_name,
                "best_params": json.dumps(best_params),
                "cv_f1_score": round(float(best_cv_f1), 4),
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1_score": val_metrics["f1_score"],
                "val_roc_auc": val_metrics["roc_auc"],
                "train_time_sec": round(train_duration, 2),
                "saved_model": model_filename
            })

    results_df = pd.DataFrame(all_results)
    
    # Save benchmark table
    metrics_path = REPORTS_DIR / "metrics_summary.csv"
    results_df.to_csv(metrics_path, index=False)
    print(f"\nSaved metrics summary to: {metrics_path}")

    # Print summary table
    print("\n" + "=" * 90)
    print("                    FINAL BENCHMARK LEADERBOARD (VALIDATION SET)")
    print("=" * 90)
    summary_table = results_df[["representation", "model", "val_accuracy", "val_precision", "val_recall", "val_f1_score", "val_roc_auc"]].sort_values(
        by="val_f1_score", ascending=False
    )
    print(tabulate(summary_table, headers="keys", tablefmt="fancy_grid", showindex=False))

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML models on BoW and TF-IDF with GridSearchCV.")
    parser.add_argument("--dataset", type=str, default="sms", help="Dataset name: sms, email, phishing, urls")
    parser.add_argument("--cv", type=int, default=5, help="Number of cross-validation folds (default: 5)")
    parser.add_argument("--quick", action="store_true", help="Quick run mode with reduced search grid")
    args = parser.parse_args()

    train_ml_pipeline(dataset_name=args.dataset, cv_folds=args.cv, quick_run=args.quick)

