"""
Advanced Threat Detectors:
- Phishing Email Detector (Text + Heuristic Threat Indicators: urgency, link mismatches, credential harvesting)
- Malicious URL Detector (Character n-gram TF-IDF + Lexical URL features: length, dots, @, IP address, HTTPS, suspicious tokens)
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Phishing Email Detector
# ─────────────────────────────────────────────────────────────────────────────

URGENCY_WORDS = [
    "urgent", "immediately", "action required", "account suspended", "suspended",
    "unauthorized", "verify your account", "security alert", "compromised",
    "within 24 hours", "locked", "expire", "immediate attention", "deactivated",
    "billing problem", "failed payment", "confirm immediately"
]

CREDENTIAL_WORDS = [
    "password", "pin", "ssn", "social security", "credit card", "cvv",
    "login", "sign in", "verify password", "credentials", "banking password",
    "security question", "account details", "identity verification"
]

SUSPICIOUS_EMAIL_CALLS_TO_ACTION = [
    "click here", "click below", "update now", "verify now", "claim prize",
    "reset now", "download attachment", "open document", "review invoice"
]


class PhishingDetector:
    """
    Analyzes emails for social-engineering and phishing attacks using
    combined NLP text classification and deep heuristic threat indicators:
    - Urgency cues
    - Credential harvesting keywords
    - Mismatched or suspicious URLs
    - Suspicious calls to action
    """

    def __init__(self, spam_detector=None):
        self.spam_detector = spam_detector

    def analyze_indicators(self, text: str) -> Dict[str, Any]:
        text_lower = text.lower()

        # 1. Urgency indicators
        found_urgency = [w for w in URGENCY_WORDS if w in text_lower]

        # 2. Credential harvesting cues
        found_credentials = [w for w in CREDENTIAL_WORDS if w in text_lower]

        # 3. Call to actions
        found_cta = [w for w in SUSPICIOUS_EMAIL_CALLS_TO_ACTION if w in text_lower]

        # 4. Link mismatch & IP link detection
        url_pattern = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')
        urls = url_pattern.findall(text)
        ip_link_detected = any(re.search(r'https?://(?:\d{1,3}\.){3}\d{1,3}', u) for u in urls)
        mismatched_links = []
        for u in urls:
            if "@" in u or "-" in urlparse(u).netloc or u.count(".") > 3 or ip_link_detected:
                mismatched_links.append(u)

        # Composite Phishing Risk Score (0 - 100)
        risk_score = 0
        risk_score += min(len(found_urgency) * 20, 40)
        risk_score += min(len(found_credentials) * 25, 40)
        risk_score += min(len(found_cta) * 15, 30)
        if ip_link_detected:
            risk_score += 35
        if mismatched_links:
            risk_score += 20

        risk_score = min(risk_score, 100)

        is_phishing = risk_score >= 45 or (len(found_urgency) > 0 and len(found_credentials) > 0)

        return {
            "is_phishing": is_phishing,
            "phishing_risk_score": risk_score,
            "urgency_triggers": found_urgency,
            "credential_triggers": found_credentials,
            "calls_to_action": found_cta,
            "detected_urls": urls,
            "mismatched_suspicious_urls": mismatched_links,
            "ip_link_detected": ip_link_detected,
        }

    def predict(self, subject: str, body: str) -> Dict[str, Any]:
        combined_text = f"Subject: {subject}\n\n{body}"
        indicator_results = self.analyze_indicators(combined_text)

        # Combine with base spam prediction if available
        base_spam_prob = 0.5
        if self.spam_detector:
            try:
                base_pred = self.spam_detector.predict(combined_text)
                base_spam_prob = base_pred["spam_probability"]
            except Exception:
                pass

        # Final blended phishing probability
        phishing_prob = (0.5 * (indicator_results["phishing_risk_score"] / 100.0)) + (0.5 * base_spam_prob)
        is_threat = phishing_prob >= 0.5 or indicator_results["is_phishing"]

        return {
            "is_phishing": bool(is_threat),
            "verdict": "🚨 Phishing Threat Detected" if is_threat else "✅ Safe / Legitimate Email",
            "phishing_probability": round(float(phishing_prob) * 100.0, 1),
            "threat_risk_score": indicator_results["phishing_risk_score"],
            "indicators": indicator_results
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Malicious URL Detector
# ─────────────────────────────────────────────────────────────────────────────

SUSPICIOUS_URL_KEYWORDS = [
    "login", "verify", "update", "account", "secure", "banking", "free",
    "paypal", "apple", "netflix", "microsoft", "signin", "support", "bonus",
    "gift", "claim", "wallet", "binance", "metamask", "service"
]


class MaliciousURLDetector:
    """
    Evaluates URL security using character n-gram TF-IDF and Lexical Features:
    - URL length
    - Number of dots (subdomain fragmentation)
    - Presence of '@' (obfuscated redirection)
    - Raw IP address as hostname
    - HTTPS presence / absence
    - Number of hyphens / suspicious path depth
    - Suspicious brand / spoof keywords
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.char_vectorizer = None
        self.classifier = None
        self._init_or_load_model(model_path)

    def extract_lexical_features(self, url: str) -> Dict[str, Any]:
        """Extract quantitative structural and security features from URL string."""
        url_clean = str(url).strip()
        parsed = urlparse(url_clean if "://" in url_clean else f"http://{url_clean}")
        hostname = parsed.netloc or parsed.path.split("/")[0]

        length = len(url_clean)
        num_dots = url_clean.count(".")
        has_at_symbol = "@" in url_clean
        num_hyphens = url_clean.count("-")
        num_slashes = url_clean.count("/")
        num_digits = sum(c.isdigit() for c in url_clean)
        digit_ratio = num_digits / max(length, 1)

        # Check if hostname is an IP address
        ip_pattern = re.compile(r'^(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?$')
        has_ip = bool(ip_pattern.match(hostname))

        # Check HTTPS
        has_https = url_clean.lower().startswith("https://")

        # Suspicious keywords
        found_keywords = [kw for kw in SUSPICIOUS_URL_KEYWORDS if kw in url_clean.lower()]

        # Heuristic Risk Calculation
        risk_points = 0
        if length > 75:
            risk_points += 20
        if num_dots > 3:
            risk_points += 20
        if has_at_symbol:
            risk_points += 30
        if has_ip:
            risk_points += 35
        if not has_https:
            risk_points += 15
        if num_hyphens > 2:
            risk_points += 15
        if found_keywords:
            risk_points += min(len(found_keywords) * 15, 30)

        risk_score = min(risk_points, 100)

        return {
            "url": url_clean,
            "hostname": hostname,
            "url_length": length,
            "num_dots": num_dots,
            "has_at_symbol": has_at_symbol,
            "has_ip_address": has_ip,
            "has_https": has_https,
            "num_hyphens": num_hyphens,
            "digit_ratio": round(digit_ratio, 3),
            "suspicious_keywords": found_keywords,
            "heuristic_risk_score": risk_score
        }

    def _init_or_load_model(self, model_path: Optional[Path]):
        vec_path = MODELS_DIR / "url_char_tfidf.joblib"
        clf_path = MODELS_DIR / "url_classifier.joblib"

        if vec_path.exists() and clf_path.exists():
            try:
                self.char_vectorizer = joblib.load(vec_path)
                self.classifier = joblib.load(clf_path)
                return
            except Exception:
                pass

        # Train a robust baseline model on representative URL patterns
        self._train_baseline_model(vec_path, clf_path)

    def _train_baseline_model(self, vec_path: Path, clf_path: Path):
        # Sample training URLs (benign vs malicious patterns)
        benign_samples = [
            "https://www.google.com/search?q=machine+learning",
            "https://github.com/scikit-learn/scikit-learn",
            "https://en.wikipedia.org/wiki/Spam_filtering",
            "https://stackoverflow.com/questions/tagged/python",
            "https://docs.python.org/3/library/urllib.parse.html",
            "https://www.nytimes.com/section/technology",
            "https://www.kaggle.com/datasets",
            "https://streamlit.io/components",
            "https://hub.docker.com/_/python",
            "https://arxiv.org/abs/2301.00000"
        ]
        malicious_samples = [
            "http://192.168.1.1/login-verify-account.php",
            "http://paypal-security-update.com@account-check.xyz/signin",
            "http://apple-support-verify-icloud.tk/reset",
            "http://free-crypto-bonus-claim.online/metamask-login",
            "http://netflix-billing-failed-update.top/auth",
            "http://bankofamerica.com-secure-verify.info/login.html",
            "http://account-microsoft-suspended-alert.net/auth/verify",
            "http://secure-login-portal-0918.xyz/wallet",
            "http://104.28.19.12/malware/download.exe",
            "http://www-facebook-security-check.biz/login.php"
        ]

        X_text = benign_samples + malicious_samples
        y = [0] * len(benign_samples) + [1] * len(malicious_samples)

        # Character 3-to-5-gram TF-IDF
        self.char_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=1000)
        X_vec = self.char_vectorizer.fit_transform(X_text)

        self.classifier = LogisticRegression(C=2.0)
        self.classifier.fit(X_vec, y)

        try:
            joblib.dump(self.char_vectorizer, vec_path)
            joblib.dump(self.classifier, clf_path)
        except Exception:
            pass

    def predict(self, url: str) -> Dict[str, Any]:
        lexical = self.extract_lexical_features(url)
        url_clean = lexical["url"]

        ml_prob = 0.5
        if self.char_vectorizer and self.classifier:
            try:
                vec = self.char_vectorizer.transform([url_clean])
                probs = self.classifier.predict_proba(vec)[0]
                ml_prob = float(probs[1])
            except Exception:
                ml_prob = lexical["heuristic_risk_score"] / 100.0

        # Weighted combination: 50% lexical heuristics + 50% char n-gram TF-IDF
        threat_score = (0.5 * ml_prob) + (0.5 * (lexical["heuristic_risk_score"] / 100.0))
        is_malicious = threat_score >= 0.45 or lexical["has_ip_address"] or (lexical["has_at_symbol"] and not lexical["has_https"])

        verdict = "🚨 Malicious / Phishing URL" if is_malicious else "✅ Benign / Safe URL"

        return {
            "url": url_clean,
            "is_malicious": bool(is_malicious),
            "verdict": verdict,
            "threat_probability": round(float(threat_score) * 100.0, 1),
            "lexical_features": lexical
        }

