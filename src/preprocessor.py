"""
Text preprocessing and cleaning pipeline for Email & SMS Spam Detection.
"""

import re
import string
import unicodedata
from typing import List, Optional

# Standard English stopwords fallback if nltk data is not yet downloaded
DEFAULT_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
    'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
    'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll',
    'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
    'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
    "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

# Preserve key spam indicators from being removed as stopwords
SPAM_SENSITIVE_WORDS = {'free', 'win', 'cash', 'prize', 'urgent', 'claim', 'call', 'text', 'now'}
FILTERED_STOPWORDS = DEFAULT_STOPWORDS - SPAM_SENSITIVE_WORDS

# Common SMS / Chat abbreviations expansion
SMS_ABBREVIATIONS = {
    r"\bu\b": "you",
    r"\bur\b": "your",
    r"\br\b": "are",
    r"\b2\b": "two",
    r"\b4\b": "for",
    r"\btxt\b": "text",
    r"\bmsg\b": "message",
    r"\bplz\b": "please",
    r"\bpls\b": "please",
    r"\bthx\b": "thanks",
    r"\bgr8\b": "great",
    r"\bl8r\b": "later",
    r"\bw/\b": "with",
    r"\bw/o\b": "without",
    r"\bc\b": "see",
    r"\bb4\b": "before"
}


class TextPreprocessor:
    """
    Robust text cleaner for spam classification.
    Supports lowercasing, entity normalization (URL, Phone, Email, Currency),
    SMS slang expansion, stopword filtering, and stemming.
    """

    def __init__(
        self,
        normalize_entities: bool = True,
        expand_slang: bool = True,
        remove_stopwords: bool = True,
        use_stemming: bool = True,
        custom_stopwords: Optional[set] = None
    ):
        self.normalize_entities = normalize_entities
        self.expand_slang = expand_slang
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming
        self.stopwords = custom_stopwords if custom_stopwords is not None else FILTERED_STOPWORDS
        
        self.stemmer = None
        if self.use_stemming:
            try:
                from nltk.stem.porter import PorterStemmer
                self.stemmer = PorterStemmer()
            except ImportError:
                self.stemmer = None

    def clean_text(self, text: str) -> str:
        """Full cleaning pipeline returning normalized string."""
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)

        # Convert to lower case
        text = text.lower()

        if self.expand_slang:
            for pattern, replacement in SMS_ABBREVIATIONS.items():
                text = re.sub(pattern, replacement, text)

        if self.normalize_entities:
            # URLs
            text = re.sub(r'https?://\S+|www\.\S+', ' urltag ', text)
            # Email addresses
            text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' emailtag ', text)
            # Currency signs ($ £ € etc followed by numbers or standalone)
            text = re.sub(r'[\$£€¥]\s*\d+(?:\.\d+)?|\b\d+\s*[\$£€¥]', ' currencytag ', text)
            text = re.sub(r'[\$£€¥]', ' currencytag ', text)
            # Phone / numeric strings with 7+ digits (typical call numbers / shortcodes)
            text = re.sub(r'\b\+?\d{1,4}?[-.\s]?\(?\d{1,3}?\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b', ' phonetag ', text)
            # General standalone numbers
            text = re.sub(r'\b\d+\b', ' numbertag ', text)

        # Remove remaining punctuation except tag brackets if any
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

        # Tokenize by whitespace
        tokens = text.split()

        # Stopword filtering
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]

        # Stemming
        if self.use_stemming and self.stemmer is not None:
            tokens = [self.stemmer.stem(t) for t in tokens]

        return " ".join(tokens)

    def __call__(self, text: str) -> str:
        return self.clean_text(text)

