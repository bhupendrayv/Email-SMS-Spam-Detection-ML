"""
Preprocessing and Exploratory Data Analysis (EDA) Pipeline for Spam Detection.
Supports:
- Text cleaning: lowercasing, punctuation/number removal, whitespace normalization
- Stopword removal and stemming (or lemmatization) via NLTK
- Engineered extra features extraction:
    * URL count
    * Email count
    * Phone number count
    * Message length (character and word length)
    * Uppercase ratio
- Class imbalance handling:
    * Compute class weights
    * SMOTE oversampling (fit and applied strictly to training split only)
- Stratified 70/15/15 train/validation/test split
- Automated EDA generation:
    * Class distribution plots
    * Message length distribution plots
    * Word clouds (Spam vs Ham)
    * Top spam keywords bar charts
    * Saved to reports/ directory
"""

from __future__ import annotations

import os
import re
import string
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# NLTK resources setup
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
except ImportError:
    nltk = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Default Stopwords fallback
FALLBACK_STOPWORDS = {
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


class Preprocessor:
    """
    Comprehensive text cleaner and feature engineer for Spam Detection.
    """

    def __init__(
        self,
        use_stemming: bool = True,
        use_lemmatization: bool = False,
        remove_stopwords: bool = True,
        custom_stopwords: Optional[set] = None
    ):
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.remove_stopwords = remove_stopwords

        # Set up stopwords
        if custom_stopwords is not None:
            self.stopwords = custom_stopwords
        else:
            try:
                from nltk.corpus import stopwords
                self.stopwords = set(stopwords.words("english"))
            except Exception:
                self.stopwords = FALLBACK_STOPWORDS

        # Set up stemmer / lemmatizer
        self.stemmer = None
        self.lemmatizer = None

        if self.use_lemmatization:
            try:
                self.lemmatizer = WordNetLemmatizer()
            except Exception:
                self.lemmatizer = None

        if self.use_stemming:
            try:
                self.stemmer = PorterStemmer()
            except Exception:
                self.stemmer = None

        # Regex patterns for feature extraction
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'\b(?:\+?\d{1,4}[-.\s]?)?\(?\d{3,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b')

    def extract_extra_features(self, text: str) -> Dict[str, Union[int, float]]:
        """
        Extract numeric engineered features from raw message before text cleaning:
        - url_count
        - email_count
        - phone_count
        - message_length (character count)
        - word_count
        - uppercase_ratio
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        url_count = len(self.url_pattern.findall(text))
        email_count = len(self.email_pattern.findall(text))
        phone_count = len(self.phone_pattern.findall(text))
        message_length = len(text)
        words = text.split()
        word_count = len(words)

        # Calculate uppercase ratio
        alpha_chars = [c for c in text if c.isalpha()]
        if len(alpha_chars) > 0:
            uppercase_count = sum(1 for c in alpha_chars if c.isupper())
            uppercase_ratio = uppercase_count / len(alpha_chars)
        else:
            uppercase_ratio = 0.0

        return {
            "url_count": url_count,
            "email_count": email_count,
            "phone_count": phone_count,
            "message_length": message_length,
            "word_count": word_count,
            "uppercase_ratio": round(uppercase_ratio, 4),
        }

    def clean_text(self, text: str) -> str:
        """
        Standard text cleaning:
        - Lowercase
        - Remove URLs and emails before punctuation removal to avoid fragment noise
        - Remove numbers
        - Remove punctuation
        - Remove extra whitespace
        - Remove stopwords
        - Apply stemming or lemmatization (NLTK)
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        # Normalize unicode
        text = unicodedata.normalize("NFKD", text)

        # Lowercase
        text = text.lower()

        # Remove URLs and Email patterns
        text = self.url_pattern.sub(" ", text)
        text = self.email_pattern.sub(" ", text)

        # Remove numbers
        text = re.sub(r'\d+', ' ', text)

        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Normalize whitespace
        tokens = text.split()

        # Remove stopwords
        if self.remove_stopwords:
            tokens = [w for w in tokens if w not in self.stopwords and len(w) > 1]

        # Apply Stemming or Lemmatization
        if self.use_lemmatization and self.lemmatizer is not None:
            tokens = [self.lemmatizer.lemmatize(w) for w in tokens]
        elif self.use_stemming and self.stemmer is not None:
            tokens = [self.stemmer.stem(w) for w in tokens]

        return " ".join(tokens)

    def preprocess_series(self, series: pd.Series) -> pd.Series:
        """Apply clean_text across a pandas Series."""
        return series.astype(str).apply(self.clean_text)

    def extract_features_df(self, series: pd.Series) -> pd.DataFrame:
        """Extract extra engineered numeric features dataframe from a raw text Series."""
        features_list = [self.extract_extra_features(text) for text in series]
        return pd.DataFrame(features_list)


# ─────────────────────────────────────────────────────────────────────────────
# Stratified 70/15/15 Split
# ─────────────────────────────────────────────────────────────────────────────

def stratified_split_70_15_15(
    df: pd.DataFrame,
    text_col: str = "text",
    target_col: str = "target",
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform a strict stratified 70% Train, 15% Validation, 15% Test split.
    First split 70% Train and 30% Temp (stratified by target).
    Second split Temp into 50% Val and 50% Test (15% each of original, stratified).
    """
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=random_state,
        stratify=df[target_col]
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=random_state,
        stratify=temp_df[target_col]
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Class Imbalance Handling
# ─────────────────────────────────────────────────────────────────────────────

def get_class_weights(y: Union[np.ndarray, pd.Series]) -> Dict[int, float]:
    """
    Compute balanced class weights for scikit-learn or PyTorch models.
    """
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=np.array(y))
    return {cls: float(weight) for cls, weight in zip(classes, weights)}


def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE oversampling strictly to training data only to avoid data leakage.
    Falls back gracefully if imbalanced-learn is not installed.
    """
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        return X_resampled, y_resampled
    except ImportError:
        print("[WARNING] imbalanced-learn (imblearn) is not installed. Returning original training data without SMOTE.")
        return X_train, y_train


# ─────────────────────────────────────────────────────────────────────────────
# EDA Visualizations (Reports)
# ─────────────────────────────────────────────────────────────────────────────

def generate_eda_reports(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    target_col: str = "target",
    output_dir: Union[str, Path] = REPORTS_DIR,
    save_format: str = "png"
) -> Dict[str, str]:
    """
    Generate and save EDA plots:
    1. Class distribution (Count & Percentage)
    2. Message length distribution by class (Characters & Words)
    3. Word clouds (Separate for Spam and Ham)
    4. Top 20 spam keywords bar chart
    Saves all figures into reports/ and returns mapping of artifact names to paths.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from collections import Counter

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_plots = {}

    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. Class Distribution Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    class_counts = df[label_col].value_counts()
    palette = ["#2ecc71" if str(k).lower() in ("ham", "safe", "0", "benign") else "#e74c3c" for k in class_counts.index]
    bars = sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax, palette=palette)
    total = len(df)
    for bar in bars.patches:
        height = bar.get_height()
        ax.annotate(f"{int(height)} ({height/total*100:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    ax.set_title("Dataset Class Distribution", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Class Label", fontweight='bold')
    ax.set_ylabel("Count", fontweight='bold')
    plt.tight_layout()
    dist_path = output_dir / f"eda_class_distribution.{save_format}"
    fig.savefig(dist_path, dpi=300)
    plt.close(fig)
    generated_plots["class_distribution"] = str(dist_path)

    # 2. Message Length Distribution
    df_copy = df.copy()
    df_copy["char_length"] = df_copy[text_col].astype(str).apply(len)
    df_copy["word_length"] = df_copy[text_col].astype(str).apply(lambda s: len(s.split()))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Character length KDE / histogram (capped at 99th percentile for clean display)
    q99_char = df_copy["char_length"].quantile(0.99)
    filtered_char = df_copy[df_copy["char_length"] <= q99_char]
    sns.histplot(data=filtered_char, x="char_length", hue=label_col, kde=True,
                 bins=40, ax=axes[0], palette={"ham": "#2ecc71", "spam": "#e74c3c"} if "ham" in df_copy[label_col].values else "viridis")
    axes[0].set_title("Message Character Length by Class (<=99th %tile)", fontweight='bold')
    axes[0].set_xlabel("Number of Characters")
    axes[0].set_ylabel("Frequency")

    # Word count KDE / histogram
    q99_word = df_copy["word_length"].quantile(0.99)
    filtered_word = df_copy[df_copy["word_length"] <= q99_word]
    sns.histplot(data=filtered_word, x="word_length", hue=label_col, kde=True,
                 bins=30, ax=axes[1], palette={"ham": "#2ecc71", "spam": "#e74c3c"} if "ham" in df_copy[label_col].values else "viridis")
    axes[1].set_title("Message Word Count by Class (<=99th %tile)", fontweight='bold')
    axes[1].set_xlabel("Number of Words")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    len_path = output_dir / f"eda_message_length_distribution.{save_format}"
    fig.savefig(len_path, dpi=300)
    plt.close(fig)
    generated_plots["length_distribution"] = str(len_path)

    # Clean text helper for tokens
    preprocessor = Preprocessor()
    cleaned_texts = df_copy[text_col].astype(str).apply(preprocessor.clean_text)

    # 3. Word Clouds
    try:
        from wordcloud import WordCloud

        # Spam Word Cloud
        spam_mask = (df_copy[target_col] == 1) if target_col in df_copy.columns else (df_copy[label_col].astype(str).str.lower().isin(["spam", "phishing", "1"]))
        ham_mask = ~spam_mask

        spam_text = " ".join(cleaned_texts[spam_mask])
        ham_text = " ".join(cleaned_texts[ham_mask])

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        if spam_text.strip():
            wc_spam = WordCloud(width=800, height=450, background_color="white", colormap="Reds", max_words=150).generate(spam_text)
            axes[0].imshow(wc_spam, interpolation="bilinear")
            axes[0].set_title("Word Cloud: SPAM Messages", fontsize=15, fontweight="bold", color="#c0392b")
        axes[0].axis("off")

        if ham_text.strip():
            wc_ham = WordCloud(width=800, height=450, background_color="white", colormap="Greens", max_words=150).generate(ham_text)
            axes[1].imshow(wc_ham, interpolation="bilinear")
            axes[1].set_title("Word Cloud: HAM (Legitimate) Messages", fontsize=15, fontweight="bold", color="#27ae60")
        axes[1].axis("off")

        plt.tight_layout()
        wc_path = output_dir / f"eda_wordclouds.{save_format}"
        fig.savefig(wc_path, dpi=300)
        plt.close(fig)
        generated_plots["wordclouds"] = str(wc_path)
    except Exception as e:
        print(f"[WARNING] WordCloud generation failed or wordcloud library unavailable: {e}")

    # 4. Top 20 Spam Words Bar Chart
    try:
        spam_mask = (df_copy[target_col] == 1) if target_col in df_copy.columns else (df_copy[label_col].astype(str).str.lower().isin(["spam", "phishing", "1"]))
        spam_tokens = " ".join(cleaned_texts[spam_mask]).split()
        top_spam = Counter(spam_tokens).most_common(20)

        if top_spam:
            words, counts = zip(*top_spam)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=list(counts), y=list(words), ax=ax, palette="rocket")
            ax.set_title("Top 20 Most Frequent Spam Keywords", fontsize=14, fontweight="bold", pad=12)
            ax.set_xlabel("Word Frequency", fontweight="bold")
            ax.set_ylabel("Spam Token", fontweight="bold")
            plt.tight_layout()
            top_words_path = output_dir / f"eda_top_spam_words.{save_format}"
            fig.savefig(top_words_path, dpi=300)
            plt.close(fig)
            generated_plots["top_spam_words"] = str(top_words_path)
    except Exception as e:
        print(f"[WARNING] Top spam keywords plot generation failed: {e}")

    return generated_plots


# ─────────────────────────────────────────────────────────────────────────────
# CLI runner for preprocessing & EDA verification
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.dataset import load_dataset

    print("=" * 60)
    print("Running Preprocessing & EDA Pipeline on SMS dataset...")
    print("=" * 60)

    df = load_dataset("sms")
    print(f"Loaded dataset: {df.shape[0]} samples.")

    # Feature extraction demo
    prep = Preprocessor()
    extra_feats = prep.extract_features_df(df["text"].iloc[:5])
    print("\nExtra features sample for first 5 rows:")
    print(extra_feats)

    # Stratified 70/15/15 split
    train_df, val_df, test_df = stratified_split_70_15_15(df)
    print(f"\nStratified 70/15/15 Split:")
    print(f"  Train : {train_df.shape[0]} rows ({train_df.shape[0]/len(df)*100:.1f}%) | Spam ratio: {train_df['target'].mean()*100:.2f}%")
    print(f"  Val   : {val_df.shape[0]} rows ({val_df.shape[0]/len(df)*100:.1f}%) | Spam ratio: {val_df['target'].mean()*100:.2f}%")
    print(f"  Test  : {test_df.shape[0]} rows ({test_df.shape[0]/len(df)*100:.1f}%) | Spam ratio: {test_df['target'].mean()*100:.2f}%")

    # Class weights
    weights = get_class_weights(train_df["target"])
    print(f"\nCalculated Class Weights: {weights}")

    # EDA Visualizations
    print("\nGenerating EDA reports in reports/ ...")
    plots = generate_eda_reports(df)
    for name, p in plots.items():
        print(f"  [Created] {name} -> {p}")

    print("\nPreprocessing & EDA pipeline executed successfully!")

