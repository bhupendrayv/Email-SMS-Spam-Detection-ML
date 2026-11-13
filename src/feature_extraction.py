"""
Feature extraction module: Comparative Bag-of-Words (BoW) and TF-IDF pipelines.
Core Innovation:
- Bag-of-Words (CountVectorizer) with unigram + bigram (1, 2)
- TF-IDF (TfidfVectorizer) with unigram + bigram (1, 2)
- Systematic hyperparameter grid tuning for max_features and ngram_range
- Strict fit on training data only to prevent data leakage
- Vocabulary inspection and diagnostic comparisons
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score


class ComparativeFeatureExtractor:
    """
    Manages and compares Bag-of-Words (CountVectorizer) and TF-IDF (TfidfVectorizer).
    Enforces strict fitting on training data only.
    """

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        sublinear_tf: bool = True,
        binary_bow: bool = False
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        self.binary_bow = binary_bow

        # 1. Bag-of-Words Vectorizer (CountVectorizer: unigram + bigram)
        self.bow_vectorizer = CountVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            binary=self.binary_bow
        )

        # 2. TF-IDF Vectorizer (TfidfVectorizer: unigram + bigram)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
            norm="l2"
        )

        self.is_fitted = False

    def fit(self, X_train: Union[List[str], np.ndarray]) -> "ComparativeFeatureExtractor":
        """
        Fit both BoW and TF-IDF vectorizers strictly on training data.
        """
        self.bow_vectorizer.fit(X_train)
        self.tfidf_vectorizer.fit(X_train)
        self.is_fitted = True
        return self

    def transform_bow(self, texts: Union[List[str], np.ndarray]) -> sp.csr_matrix:
        """Transform texts using Bag-of-Words."""
        if not self.is_fitted:
            raise RuntimeError("Vectorizer is not fitted. Call fit(X_train) first.")
        return self.bow_vectorizer.transform(texts)

    def transform_tfidf(self, texts: Union[List[str], np.ndarray]) -> sp.csr_matrix:
        """Transform texts using TF-IDF."""
        if not self.is_fitted:
            raise RuntimeError("Vectorizer is not fitted. Call fit(X_train) first.")
        return self.tfidf_vectorizer.transform(texts)

    def fit_transform_train(
        self, X_train: Union[List[str], np.ndarray]
    ) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
        """Fit on training data and return both transformed matrices."""
        self.fit(X_train)
        return self.transform_bow(X_train), self.transform_tfidf(X_train)

    def transform_eval(
        self, X_eval: Union[List[str], np.ndarray]
    ) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
        """Transform validation or test data without refitting."""
        return self.transform_bow(X_eval), self.transform_tfidf(X_eval)

    def get_vocabulary_diagnostics(self) -> Dict[str, Any]:
        """Inspect and compare extracted feature spaces."""
        if not self.is_fitted:
            return {"status": "not_fitted"}

        bow_vocab = self.bow_vectorizer.vocabulary_
        tfidf_vocab = self.tfidf_vectorizer.vocabulary_

        # Sample top unigrams and bigrams
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        unigrams = [f for f in feature_names if " " not in f][:15]
        bigrams = [f for f in feature_names if " " in f][:15]

        return {
            "is_fitted": self.is_fitted,
            "ngram_range": self.ngram_range,
            "max_features_cap": self.max_features,
            "bow_vocab_size": len(bow_vocab),
            "tfidf_vocab_size": len(tfidf_vocab),
            "sample_unigrams": unigrams,
            "sample_bigrams": bigrams,
        }

    def save(self, output_dir: Union[str, Path]):
        """Serialize both vectorizers."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.bow_vectorizer, output_dir / "bow_vectorizer.joblib")
        joblib.dump(self.tfidf_vectorizer, output_dir / "tfidf_vectorizer.joblib")

    @classmethod
    def load(cls, output_dir: Union[str, Path]) -> "ComparativeFeatureExtractor":
        """Load serialized vectorizers from disk."""
        output_dir = Path(output_dir)
        extractor = cls()
        extractor.bow_vectorizer = joblib.load(output_dir / "bow_vectorizer.joblib")
        extractor.tfidf_vectorizer = joblib.load(output_dir / "tfidf_vectorizer.joblib")
        extractor.is_fitted = True
        return extractor


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter Tuning for Vectorizers (ngram_range & max_features)
# ─────────────────────────────────────────────────────────────────────────────

def tune_vectorizer_hyperparameters(
    X_train: Union[List[str], np.ndarray],
    y_train: Union[List[int], np.ndarray],
    ngram_candidates: List[Tuple[int, int]] = [(1, 1), (1, 2), (1, 3)],
    max_features_candidates: List[int] = [1000, 3000, 5000, 10000],
    representation: str = "tfidf",
    n_splits: int = 3,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Tune max_features and ngram_range using Stratified Cross-Validation strictly on X_train.
    Uses MultinomialNB as a fast, discriminative evaluator.
    Returns best configuration, best score, and complete grid results.
    """
    X_arr = np.array(X_train)
    y_arr = np.array(y_train)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    grid_results = []
    best_score = -1.0
    best_params = {}

    for max_feat in max_features_candidates:
        for ngram in ngram_candidates:
            fold_f1_scores = []

            for train_idx, val_idx in skf.split(X_arr, y_arr):
                X_tr, X_va = X_arr[train_idx], X_arr[val_idx]
                y_tr, y_va = y_arr[train_idx], y_arr[val_idx]

                if representation == "tfidf":
                    vec = TfidfVectorizer(max_features=max_feat, ngram_range=ngram, min_df=2, sublinear_tf=True)
                else:
                    vec = CountVectorizer(max_features=max_feat, ngram_range=ngram, min_df=2)

                # Fit ONLY on fold train
                X_tr_trans = vec.fit_transform(X_tr)
                X_va_trans = vec.transform(X_va)

                clf = MultinomialNB()
                clf.fit(X_tr_trans, y_tr)
                preds = clf.predict(X_va_trans)

                fold_f1 = f1_score(y_va, preds, average="weighted")
                fold_f1_scores.append(fold_f1)

            mean_score = float(np.mean(fold_f1_scores))
            res = {
                "max_features": max_feat,
                "ngram_range": ngram,
                "representation": representation,
                "mean_cv_f1": round(mean_score, 4),
            }
            grid_results.append(res)

            if mean_score > best_score:
                best_score = mean_score
                best_params = {"max_features": max_feat, "ngram_range": ngram}

    return {
        "best_params": best_params,
        "best_cv_f1": round(best_score, 4),
        "all_results": sorted(grid_results, key=lambda x: x["mean_cv_f1"], reverse=True)
    }
