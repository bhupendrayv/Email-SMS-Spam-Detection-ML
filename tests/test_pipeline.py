"""
Unit and integration tests for Spam Detection ML pipeline.
"""

import unittest
from src.preprocessor import TextPreprocessor


class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = TextPreprocessor()

    def test_url_normalization(self):
        text = "Click here https://spam-link.biz/claim to get money!"
        clean = self.preprocessor.clean_text(text)
        self.assertIn("urltag", clean)
        self.assertNotIn("https", clean)

    def test_phone_normalization(self):
        text = "Call 08000930705 now for free entry!"
        clean = self.preprocessor.clean_text(text)
        self.assertIn("phonetag", clean)

    def test_currency_normalization(self):
        text = "You won £1000 or $500!"
        clean = self.preprocessor.clean_text(text)
        self.assertIn("currencytag", clean)

    def test_sms_slang_expansion(self):
        text = "r great thx b4 lunch"
        clean = self.preprocessor.clean_text(text)
        self.assertIn("great", clean)
        self.assertIn("thank", clean)  # stemmed from thanks


if __name__ == "__main__":
    unittest.main()

