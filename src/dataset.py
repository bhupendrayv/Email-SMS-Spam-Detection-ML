"""
Dataset management: unified multi-dataset loader and train/test stratified splitter.

Supports four datasets:
  1. sms       — UCI SMS Spam Collection  (ham / spam)
  2. email     — SpamAssassin Email Corpus (ham / spam)
  3. phishing  — Phishing Email Dataset    (safe / phishing)
  4. urls      — Malicious URLs Dataset    (benign / phishing / malware / defacement)

Each dataset is loaded through the download_data module which handles
downloading, Kaggle CLI integration, and offline synthetic fallback generation.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


# ──────────────────────────────────────────────────────────────────────
# Path constants
# ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Ensure project root is importable (for download_data at top-level)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ──────────────────────────────────────────────────────────────────────
# Dataset metadata registry
# ──────────────────────────────────────────────────────────────────────

DATASET_INFO: Dict[str, Dict] = {
    "sms": {
        "name": "SMS Spam Collection (UCI)",
        "csv_path": RAW_DATA_DIR / "sms_spam" / "sms_spam.csv",
        "label_map": {"ham": 0, "spam": 1},
        "text_column": "text",
        "num_classes": 2,
        "description": (
            "5,574 SMS messages collected for spam research, tagged as "
            "ham (legitimate) or spam."
        ),
    },
    "email": {
        "name": "SpamAssassin Email Corpus",
        "csv_path": RAW_DATA_DIR / "email_spam" / "email_spam.csv",
        "label_map": {"ham": 0, "spam": 1},
        "text_column": "text",
        "num_classes": 2,
        "description": (
            "~6,000 raw emails from the Apache SpamAssassin project, "
            "labelled as ham or spam."
        ),
    },
    "phishing": {
        "name": "Phishing Email Dataset (Kaggle)",
        "csv_path": RAW_DATA_DIR / "phishing_email" / "phishing_email.csv",
        "label_map": {"safe": 0, "phishing": 1},
        "text_column": "text",
        "num_classes": 2,
        "description": (
            "Emails labelled as phishing or safe.  Useful for training "
            "classifiers that protect against social-engineering attacks."
        ),
    },
    "urls": {
        "name": "Malicious URLs Dataset (Kaggle)",
        "csv_path": RAW_DATA_DIR / "malicious_urls" / "malicious_urls.csv",
        "label_map": {"benign": 0, "phishing": 1, "malware": 2, "defacement": 3},
        "text_column": "text",  # URL string treated as text
        "num_classes": 4,
        "description": (
            "URLs labelled as benign, phishing, malware, or defacement.  "
            "The URL string itself is the input feature."
        ),
    },
}


def available_datasets() -> List[str]:
    """Return the list of valid dataset keys."""
    return list(DATASET_INFO.keys())


def dataset_summary() -> pd.DataFrame:
    """Return a summary table of all registered datasets."""
    rows = []
    for key, info in DATASET_INFO.items():
        exists = info["csv_path"].exists()
        n_rows = "—"
        if exists:
            try:
                n_rows = f"{len(pd.read_csv(info['csv_path'])):,}"
            except Exception:
                n_rows = "err"
        rows.append({
            "Key": key,
            "Name": info["name"],
            "Classes": info["num_classes"],
            "Labels": ", ".join(info["label_map"].keys()),
            "Rows": n_rows,
            "Downloaded": "✓" if exists else "✗",
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# Core loader
# ──────────────────────────────────────────────────────────────────────

def load_dataset(
    dataset_key: str = "sms",
    data_path: Optional[str] = None,
    auto_download: bool = True,
) -> pd.DataFrame:
    """
    Load a project dataset by key and return a normalised DataFrame.

    Parameters
    ----------
    dataset_key : str
        One of ``"sms"``, ``"email"``, ``"phishing"``, ``"urls"``.
    data_path : str, optional
        Override path to a custom CSV/TSV file.  When given, ``dataset_key``
        is still used to look up the label map.
    auto_download : bool
        If the CSV is missing and this is True, attempt download via
        ``download_data.py`` (falls back to synthetic generation offline).

    Returns
    -------
    pd.DataFrame
        Columns: ``label`` (str), ``text`` (str), ``target`` (int).
        The URL dataset also keeps a ``url`` column.
    """
    if dataset_key not in DATASET_INFO:
        raise ValueError(
            f"Unknown dataset key '{dataset_key}'.  "
            f"Choose from: {available_datasets()}"
        )

    info = DATASET_INFO[dataset_key]

    # Resolve the CSV path
    if data_path is not None:
        csv_path = Path(data_path)
    else:
        csv_path = info["csv_path"]

    # Auto-download / generate if missing
    if not csv_path.exists() and auto_download:
        try:
            from download_data import load_dataset as dl_load
            _log(f"Dataset '{dataset_key}' not found locally — attempting download …")
            return dl_load(dataset_key, auto_download=True)
        except ImportError:
            _log("download_data module not found.  Attempting inline fallback …")
            _inline_fallback(dataset_key, csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset '{dataset_key}' not found at {csv_path}.\n"
            f"Run:  python download_data.py --{dataset_key}"
        )

    # Read and normalise
    df = _read_csv_flexible(csv_path)
    df = _normalise_columns(df, info)
    return df


def _log(msg: str) -> None:
    print(f"  [dataset] {msg}")


def _read_csv_flexible(path: Path) -> pd.DataFrame:
    """Read a CSV/TSV with multiple fallback strategies."""
    for sep in [",", "\t"]:
        for enc in ["utf-8", "latin-1"]:
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc, on_bad_lines="skip")
                if len(df.columns) >= 2 and len(df) > 0:
                    return df
            except Exception:
                continue
    # Last resort: headerless tab-separated
    try:
        return pd.read_csv(path, sep="\t", header=None,
                           names=["label", "text"], encoding="latin-1")
    except Exception as exc:
        raise RuntimeError(f"Cannot parse {path}: {exc}") from exc


def _normalise_columns(df: pd.DataFrame, info: Dict) -> pd.DataFrame:
    """Standardise column names and create integer target."""
    # Lowercase all column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Common renames
    rename_map = {}
    if "v1" in df.columns and "v2" in df.columns:
        rename_map = {"v1": "label", "v2": "text"}
    if "message" in df.columns and "text" not in df.columns:
        rename_map["message"] = "text"
    if "email text" in df.columns:
        rename_map["email text"] = "text"
    if "email type" in df.columns:
        rename_map["email type"] = "label"
    if "type" in df.columns and "label" not in df.columns:
        rename_map["type"] = "label"
    if rename_map:
        df = df.rename(columns=rename_map)

    # Ensure required columns
    if "label" not in df.columns or "text" not in df.columns:
        # Try to infer
        cols = list(df.columns)
        if len(cols) >= 2:
            text_col = next((c for c in cols if any(
                k in c for k in ["text", "body", "content", "email", "url", "message"])),
                cols[-1])
            label_col = next((c for c in cols if any(
                k in c for k in ["label", "type", "class", "category"])),
                cols[0])
            df = df.rename(columns={text_col: "text", label_col: "label"})

    df = df[["label", "text"] + [c for c in df.columns if c not in ("label", "text")]].copy()
    df = df.dropna(subset=["label", "text"])
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["text"] = df["text"].astype(str)

    # Map labels to integers
    df["target"] = df["label"].map(info["label_map"])
    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)
    return df


def _inline_fallback(dataset_key: str, csv_path: Path) -> None:
    """Generate a minimal synthetic dataset when download_data is unavailable."""
    import random
    random.seed(42)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if dataset_key == "sms":
        rows = ([{"label": "ham", "text": f"Normal message {i}."} for i in range(200)] +
                [{"label": "spam", "text": f"FREE PRIZE! Call {i:05d} NOW!"} for i in range(50)])
    elif dataset_key == "email":
        rows = ([{"label": "ham", "text": f"Meeting notes for project #{i}."} for i in range(200)] +
                [{"label": "spam", "text": f"URGENT: Claim £{i*100} at scam.com."} for i in range(50)])
    elif dataset_key == "phishing":
        rows = ([{"label": "safe", "text": f"Hi team, review doc #{i}."} for i in range(200)] +
                [{"label": "phishing", "text": f"Verify your account at fake.com/id={i}."} for i in range(100)])
    elif dataset_key == "urls":
        rows = ([{"label": "benign", "text": f"https://example.com/page{i}"} for i in range(150)] +
                [{"label": "phishing", "text": f"http://phish{i}.xyz/login"} for i in range(50)] +
                [{"label": "malware", "text": f"http://malware{i}.tk/payload"} for i in range(30)] +
                [{"label": "defacement", "text": f"http://hacked{i}.ml/deface"} for i in range(20)])
    else:
        return

    random.shuffle(rows)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    _log(f"Generated inline fallback ({len(rows)} rows) → {csv_path}")


# ──────────────────────────────────────────────────────────────────────
# Train / test split
# ──────────────────────────────────────────────────────────────────────

def get_train_test_data(
    dataset_key: str = "sms",
    data_path: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Load a dataset and return stratified (X_train, X_test, y_train, y_test).

    Parameters
    ----------
    dataset_key : str
        ``"sms"``, ``"email"``, ``"phishing"``, or ``"urls"``.
    data_path : str, optional
        Override CSV path.
    test_size : float
        Fraction held out for testing.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    X_train, X_test : pd.Series[str]
        Raw message text.
    y_train, y_test : pd.Series[int]
        Integer-encoded labels.
    """
    df = load_dataset(dataset_key, data_path=data_path)
    X = df["text"]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def get_combined_dataset(
    keys: Optional[List[str]] = None,
    binary: bool = True,
) -> pd.DataFrame:
    """
    Load and concatenate multiple datasets into one DataFrame.

    When ``binary=True`` (default), all labels are collapsed to 0 (safe/ham/benign)
    and 1 (spam/phishing/malware/defacement).  Useful for unified model training.
    """
    if keys is None:
        keys = ["sms", "email"]

    frames = []
    for key in keys:
        try:
            df = load_dataset(key)
            df["source_dataset"] = key
            frames.append(df)
        except FileNotFoundError:
            _log(f"Skipping '{key}' (not downloaded)")
            continue

    if not frames:
        raise RuntimeError("No datasets could be loaded.")

    combined = pd.concat(frames, ignore_index=True)

    if binary:
        # Collapse to binary: 0 = benign/ham/safe, 1 = everything else
        combined["target"] = (combined["target"] > 0).astype(int)
        combined["label"] = combined["target"].map({0: "ham", 1: "spam"})

    return combined
