"""
Unified Inference Engine for Spam, Email, Phishing, and URL Threat Detection.
Provides token attribution, probability confidence scores, and explainability.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

from src.preprocess import Preprocessor
from src.advanced_detectors import PhishingDetector, MaliciousURLDetector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


class SpamPredictor:
    """
    Unified detector supporting:
    1. SMS spam classification
    2. Email spam classification (Subject + Body)
    3. Phishing detection with social engineering indicators
    4. Malicious URL inspection (Character n-gram TF-IDF + Lexical features)
    5. Token explainability and probability confidence
    """

    def __init__(
        self,
        model_name: str = "svm_rbf",
        representation: str = "tfidf",
        models_dir: Union[str, Path] = MODELS_DIR
    ):
        self.representation = representation.lower()
        self.model_name = model_name.lower().replace(" ", "_")
        self.models_dir = Path(models_dir)
        self.preprocessor = Preprocessor()

        # Load vectorizer
        vec_file = "tfidf_vectorizer.joblib" if self.representation == "tfidf" else "bow_vectorizer.joblib"
        vec_path = self.models_dir / vec_file
        if not vec_path.exists():
            raise FileNotFoundError(f"Vectorizer not found at {vec_path}. Please train models first.")
        self.vectorizer = joblib.load(vec_path)

        # Load classifier model
        model_filename = f"{self.representation}_{self.model_name}.joblib"
        model_path = self.models_dir / model_filename
        if not model_path.exists():
            # Fallback to available model
            avail = [f for f in os.listdir(self.models_dir) if f.startswith(self.representation) and f.endswith(".joblib") and "vectorizer" not in f]
            if avail:
                model_filename = avail[0]
                model_path = self.models_dir / model_filename
            else:
                raise FileNotFoundError(f"No trained models found in {self.models_dir}")
        self.model = joblib.load(model_path)

        # Sub-detectors
        self.phishing_detector = PhishingDetector(spam_detector=self)
        self.url_detector = MaliciousURLDetector()

    def get_token_attributions(self, clean_text: str) -> List[Dict[str, Any]]:
        """
        Calculates token-level influence on the spam classification verdict.
        """
        words = clean_text.split()
        if not words:
            return []

        vocab = getattr(self.vectorizer, "vocabulary_", {})
        attributions = []

        has_coef = hasattr(self.model, "coef_")
        has_log_prob = hasattr(self.model, "feature_log_prob_")

        for word in set(words):
            if word in vocab:
                idx = vocab[word]
                score = 0.0
                if has_log_prob:
                    # Naive Bayes Log-Likelihood Ratio
                    log_spam = self.model.feature_log_prob_[1, idx]
                    log_ham = self.model.feature_log_prob_[0, idx]
                    score = float(log_spam - log_ham)
                elif has_coef:
                    score = float(self.model.coef_[0, idx])
                else:
                    # Generic heuristic weight
                    score = 1.2 if word in {"free", "claim", "win", "prize", "cash", "call", "urgent", "credit", "card", "stop"} else 0.2

                attributions.append({
                    "word": word,
                    "score": round(score, 3),
                    "sentiment": "Spam Trigger" if score > 0 else "Ham Indicator"
                })

        attributions.sort(key=lambda x: abs(x["score"]), reverse=True)
        return attributions[:10]

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict SMS or message text.
        Returns label, confidence %, spam probability %, clean text, and top influencing words.
        """
        clean = self.preprocessor.clean_text(text)
        X_vec = self.vectorizer.transform([clean])

        pred_class = int(self.model.predict(X_vec)[0])

        spam_prob = 0.5
        if hasattr(self.model, "predict_proba"):
            try:
                probs = self.model.predict_proba(X_vec)[0]
                spam_prob = float(probs[1])
            except Exception:
                spam_prob = 1.0 if pred_class == 1 else 0.0
        elif hasattr(self.model, "decision_function"):
            try:
                dec = float(self.model.decision_function(X_vec)[0])
                # Sigmoid calibration
                spam_prob = 1.0 / (1.0 + np.exp(-dec))
            except Exception:
                spam_prob = 1.0 if pred_class == 1 else 0.0
        else:
            spam_prob = 1.0 if pred_class == 1 else 0.0

        ham_prob = 1.0 - spam_prob
        is_spam = bool(pred_class == 1) or (spam_prob >= 0.5)
        confidence = (spam_prob if is_spam else ham_prob) * 100.0

        triggers = self.get_token_attributions(clean)

        return {
            "prediction": "Spam" if is_spam else "Legitimate (Ham)",
            "is_spam": is_spam,
            "spam_probability_pct": round(spam_prob * 100.0, 1),
            "ham_probability_pct": round(ham_prob * 100.0, 1),
            "confidence_pct": round(confidence, 1),
            "representation": self.representation.upper(),
            "model_name": self.model_name,
            "cleaned_text": clean,
            "top_influencing_words": triggers
        }

    def predict_email(self, subject: str, body: str) -> Dict[str, Any]:
        """Classify email taking subject and body into account."""
        combined_text = f"Subject: {subject}\n\n{body}" if subject.strip() else body
        res = self.predict(combined_text)
        res["subject"] = subject
        res["body"] = body
        return res

    def predict_phishing(self, subject: str, body: str) -> Dict[str, Any]:
        """Detect phishing attacks using combined NLP and heuristic indicators."""
        return self.phishing_detector.predict(subject, body)

    def predict_url(self, url: str) -> Dict[str, Any]:
        """Analyze URL maliciousness."""
        return self.url_detector.predict(url)
