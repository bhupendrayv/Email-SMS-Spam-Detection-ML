"""
Comprehensive Automated Test Suite:
- Unit tests for text preprocessor, feature engineering, and stratified splitting
- Integration tests for all FastAPI REST endpoints
"""

from __future__ import annotations

import unittest
from fastapi.testclient import TestClient
from src.preprocess import Preprocessor, stratified_split_70_15_15
from src.api import app, SMSRequest, EmailRequest, PhishingRequest, URLRequest
from src.predict import SpamPredictor
from src.advanced_detectors import PhishingDetector, MaliciousURLDetector
from src.api import ml_resources


class TestPreprocessingPipeline(unittest.TestCase):
    def setUp(self):
        self.preprocessor = Preprocessor()

    def test_clean_text_numbers_and_punctuation(self):
        text = "Hello! Win $1,000 cash prizes today 12345???"
        cleaned = self.preprocessor.clean_text(text)
        self.assertNotIn("12345", cleaned)
        self.assertNotIn("?", cleaned)
        self.assertNotIn("!", cleaned)
        self.assertTrue(cleaned.islower())

    def test_extra_features_extraction(self):
        text = "URGENT! Call +18005550199 or email us at support@claim-prize.com immediately! Visit http://scam.org"
        feats = self.preprocessor.extract_extra_features(text)
        self.assertGreaterEqual(feats["url_count"], 1)
        self.assertGreaterEqual(feats["email_count"], 1)
        self.assertGreaterEqual(feats["phone_count"], 1)
        self.assertGreater(feats["uppercase_ratio"], 0.0)
        self.assertGreater(feats["message_length"], 20)

    def test_stratified_split(self):
        import pandas as pd
        df = pd.DataFrame({
            "text": [f"message {i}" for i in range(100)],
            "target": [0] * 80 + [1] * 20
        })
        train_df, val_df, test_df = stratified_split_70_15_15(df, target_col="target")
        self.assertEqual(len(train_df), 70)
        self.assertEqual(len(val_df), 15)
        self.assertEqual(len(test_df), 15)
        self.assertEqual(train_df["target"].sum(), 14)
        self.assertEqual(val_df["target"].sum(), 3)
        self.assertEqual(test_df["target"].sum(), 3)


class TestAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pred = SpamPredictor(model_name="svm_rbf", representation="tfidf")
        ml_resources["predictor"] = pred
        ml_resources["phishing_detector"] = PhishingDetector(spam_detector=pred)
        ml_resources["url_detector"] = MaliciousURLDetector()

    def test_api_sms_endpoint(self):
        from src.api import predict_sms
        req = SMSRequest(text="WINNER! Claim your £1000 cash prize! Call 09061701461 to claim")
        res = predict_sms(req)
        self.assertEqual(res.label, "Spam")
        self.assertGreaterEqual(res.spam_probability, 0.5)
        self.assertIn("svm_rbf", res.model_used)

    def test_api_email_endpoint(self):
        from src.api import predict_email
        req = EmailRequest(subject="Team lunch", body="Let's grab pizza tomorrow at 1pm")
        res = predict_email(req)
        self.assertEqual(res.label, "Legitimate (Ham)")
        self.assertLess(res.spam_probability, 0.5)

    def test_api_phishing_endpoint(self):
        from src.api import predict_phishing
        req = PhishingRequest(
            subject="Urgent Security Notice",
            body="Your PayPal account has been suspended! Enter password immediately: http://192.168.1.1/login"
        )
        res = predict_phishing(req)
        self.assertEqual(res.label, "Spam")
        self.assertGreaterEqual(res.spam_probability, 0.5)
        self.assertIn("indicators", res.details)

    def test_api_url_endpoint(self):
        from src.api import predict_url
        req = URLRequest(url="http://paypal-security-update.com@account-check.xyz/login.php")
        res = predict_url(req)
        self.assertEqual(res.label, "Spam")
        self.assertGreaterEqual(res.spam_probability, 0.5)

    def test_api_stats_and_feedback(self):
        from src.api import get_stats, get_history, submit_feedback, FeedbackRequest
        stats = get_stats()
        self.assertIn("total_checked", stats)
        self.assertIn("spam_percentage", stats)

        history = get_history(limit=5)
        self.assertIn("history", history)

        fb_req = FeedbackRequest(
            message_content="Sample text",
            predicted_label="Spam",
            actual_label="Spam",
            comments="Verified"
        )
        fb_res = submit_feedback(fb_req)
        self.assertEqual(fb_res["status"], "success")


if __name__ == "__main__":
    unittest.main()

