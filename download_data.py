#!/usr/bin/env python3
"""
download_data.py — Automated Dataset Downloader & Validator
============================================================

Downloads, extracts, validates and normalises all four project datasets into
data/raw/<dataset_name>/.  Each dataset is converted to a canonical CSV with
at minimum the columns ``label`` and ``text`` (plus any dataset-specific extras).

Datasets
--------
1. **SMS Spam Collection** (UCI) — 5 574 SMS messages labelled ham / spam
2. **SpamAssassin Email Corpus** — ~6 000 emails labelled ham / spam
3. **Phishing Email Dataset** (Kaggle) — emails labelled phishing / safe
4. **Malicious URLs Dataset** (Kaggle) — URLs labelled benign / phishing / malware / defacement

Usage
-----
    # Download everything (uses network)
    python download_data.py --all

    # Download a single dataset
    python download_data.py --sms
    python download_data.py --email
    python download_data.py --phishing
    python download_data.py --urls

    # Generate offline synthetic fallbacks (no network required)
    python download_data.py --synthetic

    # Verify existing downloads
    python download_data.py --verify

Prerequisites
-------------
* Kaggle datasets require the ``kaggle`` CLI to be installed and authenticated
  (``pip install kaggle`` + place ``kaggle.json`` in ``~/.kaggle/``).
  If the CLI is unavailable the script prints manual-download instructions
  and falls back to synthetic data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import textwrap
import urllib.request
import zipfile
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas is required.  Install with:  pip install pandas")


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "data"
RAW_DIR      = DATA_DIR / "raw"

# Per-dataset output directories
SMS_DIR      = RAW_DIR / "sms_spam"
EMAIL_DIR    = RAW_DIR / "email_spam"
PHISHING_DIR = RAW_DIR / "phishing_email"
URL_DIR      = RAW_DIR / "malicious_urls"

TIMEOUT = 60  # seconds for HTTP requests

HEADERS = {"User-Agent": "Mozilla/5.0 (SpamDetectionProject/1.0)"}


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _log(msg: str, level: str = "INFO") -> None:
    colours = {"INFO": "\033[94m", "OK": "\033[92m", "WARN": "\033[93m",
               "ERR": "\033[91m", "END": "\033[0m"}
    prefix = colours.get(level, "") + f"[{level}]" + colours.get("END", "")
    print(f"  {prefix} {msg}")


def _download(url: str, dest: Path, desc: str = "") -> bool:
    """Download *url* to *dest*.  Returns True on success."""
    desc = desc or url
    _log(f"Downloading {desc} …")
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as fp:
                shutil.copyfileobj(resp, fp)
        _log(f"Saved → {dest}", "OK")
        return True
    except Exception as exc:
        _log(f"Download failed ({exc})", "ERR")
        return False


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _kaggle_available() -> bool:
    """Check whether the ``kaggle`` CLI is installed and authenticated."""
    try:
        r = subprocess.run(
            [sys.executable, "-m", "kaggle", "--version"],
            capture_output=True, text=True, timeout=15,
        )
        return r.returncode == 0
    except Exception:
        return False


def _kaggle_download(dataset_slug: str, dest_dir: Path) -> bool:
    """Use ``kaggle datasets download`` to fetch a Kaggle dataset."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Kaggle download: {dataset_slug} → {dest_dir}")
    try:
        r = subprocess.run(
            [sys.executable, "-m", "kaggle", "datasets", "download",
             "-d", dataset_slug, "-p", str(dest_dir), "--unzip"],
            capture_output=True, text=True, timeout=300,
        )
        if r.returncode == 0:
            _log("Kaggle download succeeded", "OK")
            return True
        else:
            _log(f"Kaggle CLI error: {r.stderr.strip()}", "ERR")
            return False
    except Exception as exc:
        _log(f"Kaggle CLI error: {exc}", "ERR")
        return False


def _print_manual_instructions(dataset_name: str, kaggle_slug: str,
                                target_dir: Path, url_hint: str) -> None:
    """Print clear manual-download instructions for a Kaggle dataset."""
    _log(f"Automatic download unavailable for {dataset_name}.", "WARN")
    print(textwrap.dedent(f"""\
        ┌─────────────────────────────────────────────────────────────────┐
        │  MANUAL DOWNLOAD INSTRUCTIONS — {dataset_name:<30s} │
        ├─────────────────────────────────────────────────────────────────┤
        │  1. Go to: {url_hint:<51s} │
        │  2. Click  "Download"  (you need a free Kaggle account).       │
        │  3. Extract the ZIP into:                                      │
        │     {str(target_dir):<59s} │
        │                                                                │
        │  Alternatively, install & configure the Kaggle CLI:            │
        │     pip install kaggle                                         │
        │     # Place your API key at ~/.kaggle/kaggle.json              │
        │     kaggle datasets download -d {kaggle_slug:<28s} │
        └─────────────────────────────────────────────────────────────────┘
    """))


# ══════════════════════════════════════════════════════════════════════
#  1.  SMS SPAM COLLECTION  (UCI)
# ══════════════════════════════════════════════════════════════════════

def download_sms_spam() -> Optional[Path]:
    """
    Download the UCI SMS Spam Collection (5 574 messages).

    Tries three sources in order:
      1. UCI Machine Learning Repository ZIP
      2. GitHub mirror (raw TSV)
      3. Kaggle CLI (if available)
    """
    _log("=" * 60)
    _log("Dataset 1/4 — SMS Spam Collection (UCI)")
    _log("=" * 60)

    out_csv = SMS_DIR / "sms_spam.csv"
    if out_csv.exists():
        df = pd.read_csv(out_csv)
        _log(f"Already exists ({len(df)} rows).  Skipping download.", "OK")
        return out_csv

    SMS_DIR.mkdir(parents=True, exist_ok=True)

    # Source 1 — UCI archive ZIP
    uci_url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = SMS_DIR / "sms_spam_collection.zip"
    if _download(uci_url, zip_path, "UCI SMS Spam Collection ZIP"):
        try:
            with zipfile.ZipFile(zip_path) as zf:
                # The collection file inside is called SMSSpamCollection
                for name in zf.namelist():
                    if "SMSSpamCollection" in name and not name.endswith("/"):
                        with zf.open(name) as f:
                            content = f.read().decode("utf-8", errors="replace")
                        rows = []
                        for line in content.strip().split("\n"):
                            parts = line.strip().split("\t", 1)
                            if len(parts) == 2:
                                rows.append({"label": parts[0].strip().lower(),
                                              "text": parts[1].strip()})
                        df = pd.DataFrame(rows)
                        df.to_csv(out_csv, index=False)
                        _log(f"Extracted {len(df)} messages → {out_csv}", "OK")
                        return out_csv
        except Exception as exc:
            _log(f"ZIP extraction failed: {exc}", "ERR")

    # Source 2 — GitHub mirror (justmarkham)
    gh_url = ("https://raw.githubusercontent.com/justmarkham/"
              "pycon-2016-tutorial/master/data/sms.tsv")
    tsv_path = SMS_DIR / "sms.tsv"
    if _download(gh_url, tsv_path, "GitHub mirror TSV"):
        try:
            df = pd.read_csv(tsv_path, sep="\t", names=["label", "text"], header=0)
            df["label"] = df["label"].str.strip().str.lower()
            df.to_csv(out_csv, index=False)
            _log(f"Converted {len(df)} messages → {out_csv}", "OK")
            return out_csv
        except Exception as exc:
            _log(f"TSV parse failed: {exc}", "ERR")

    # Source 3 — Kaggle CLI
    if _kaggle_available():
        if _kaggle_download("uciml/sms-spam-collection-dataset", SMS_DIR):
            # Kaggle usually provides spam.csv with columns v1, v2
            for candidate in SMS_DIR.glob("*.csv"):
                try:
                    df = pd.read_csv(candidate, encoding="latin-1")
                    cols = [c.lower() for c in df.columns]
                    if "v1" in cols and "v2" in cols:
                        df = df.rename(columns={"v1": "label", "v2": "text"})
                    elif "label" not in cols:
                        continue
                    df = df[["label", "text"]].dropna()
                    df["label"] = df["label"].str.strip().str.lower()
                    df.to_csv(out_csv, index=False)
                    _log(f"Kaggle → {len(df)} messages → {out_csv}", "OK")
                    return out_csv
                except Exception:
                    continue

    _log("All download sources failed for SMS Spam Collection.", "ERR")
    return None


# ══════════════════════════════════════════════════════════════════════
#  2.  SPAMASSASSIN EMAIL CORPUS
# ══════════════════════════════════════════════════════════════════════

def _parse_email_file(file_path: Path) -> Optional[str]:
    """Extract the text body from a raw email file."""
    try:
        with open(file_path, "rb") as fp:
            msg = BytesParser(policy=policy.default).parse(fp)
        body = msg.get_body(preferencelist=("plain",))
        if body:
            text = body.get_content()
            if isinstance(text, str):
                return text.strip()
        # Fallback: just read the file and strip headers
        with open(file_path, "r", encoding="utf-8", errors="replace") as fp:
            content = fp.read()
        # Crude header/body split on first blank line
        parts = content.split("\n\n", 1)
        return parts[1].strip() if len(parts) > 1 else content.strip()
    except Exception:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as fp:
                return fp.read().strip()
        except Exception:
            return None


def _extract_tar_emails(tar_url: str, dest_dir: Path, label: str,
                        tar_path: Path) -> List[dict]:
    """Download a tar.bz2, extract emails, return list of dicts."""
    rows = []
    if _download(tar_url, tar_path, f"SpamAssassin {label} corpus"):
        try:
            with tarfile.open(tar_path, "r:bz2") as tf:
                tf.extractall(path=dest_dir, filter="data")
            # Walk extracted directories for email files
            for root, _, files in os.walk(dest_dir):
                for fname in files:
                    fpath = Path(root) / fname
                    # Skip hidden/system files and non-text
                    if fname.startswith(".") or fname == "cmds":
                        continue
                    text = _parse_email_file(fpath)
                    if text and len(text) > 10:
                        rows.append({"label": label, "text": text[:10000]})
            _log(f"  Parsed {len(rows)} {label} emails", "OK")
        except Exception as exc:
            _log(f"  Extraction error: {exc}", "ERR")
    return rows


def download_email_spam() -> Optional[Path]:
    """
    Download the Apache SpamAssassin public corpus.

    Three archives are fetched:
      - easy_ham    (~2 500 legitimate emails)
      - hard_ham    (~250 tricky legitimate emails)
      - spam        (~500 spam emails)
      - spam_2      (~1 400 additional spam emails)
    """
    _log("=" * 60)
    _log("Dataset 2/4 — SpamAssassin Email Corpus")
    _log("=" * 60)

    out_csv = EMAIL_DIR / "email_spam.csv"
    if out_csv.exists():
        df = pd.read_csv(out_csv)
        _log(f"Already exists ({len(df)} rows).  Skipping download.", "OK")
        return out_csv

    EMAIL_DIR.mkdir(parents=True, exist_ok=True)
    extract_dir = EMAIL_DIR / "_extracted"

    base_url = "https://spamassassin.apache.org/old/publiccorpus"
    archives = [
        (f"{base_url}/20030228_easy_ham.tar.bz2",   "ham",  "easy_ham.tar.bz2"),
        (f"{base_url}/20030228_hard_ham.tar.bz2",   "ham",  "hard_ham.tar.bz2"),
        (f"{base_url}/20030228_spam.tar.bz2",       "spam", "spam.tar.bz2"),
        (f"{base_url}/20030228_spam_2.tar.bz2",     "spam", "spam_2.tar.bz2"),
    ]

    all_rows: List[dict] = []
    for url, label, fname in archives:
        tar_path = EMAIL_DIR / fname
        rows = _extract_tar_emails(url, extract_dir, label, tar_path)
        all_rows.extend(rows)

    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(out_csv, index=False)
        _log(f"Saved {len(df)} emails → {out_csv}", "OK")
        _log(f"  ham: {len(df[df['label']=='ham'])},  spam: {len(df[df['label']=='spam'])}")
        # Clean up extracted directory to save disk space
        if extract_dir.exists():
            shutil.rmtree(extract_dir, ignore_errors=True)
        return out_csv

    _log("SpamAssassin download failed.", "ERR")
    return None


# ══════════════════════════════════════════════════════════════════════
#  3.  PHISHING EMAIL DATASET  (Kaggle)
# ══════════════════════════════════════════════════════════════════════

def download_phishing_email() -> Optional[Path]:
    """
    Download the Kaggle "Phishing Email Dataset".

    Kaggle slug: ``subhajournal/phishingemails``
    Columns expected after download: ``Email Text``, ``Email Type``
    """
    _log("=" * 60)
    _log("Dataset 3/4 — Phishing Email Dataset (Kaggle)")
    _log("=" * 60)

    out_csv = PHISHING_DIR / "phishing_email.csv"
    if out_csv.exists():
        df = pd.read_csv(out_csv)
        _log(f"Already exists ({len(df)} rows).  Skipping download.", "OK")
        return out_csv

    PHISHING_DIR.mkdir(parents=True, exist_ok=True)
    kaggle_slug = "subhajournal/phishingemails"
    kaggle_url  = f"https://www.kaggle.com/datasets/{kaggle_slug}"

    if _kaggle_available():
        if _kaggle_download(kaggle_slug, PHISHING_DIR):
            # Find and normalise the CSV
            for candidate in PHISHING_DIR.glob("*.csv"):
                try:
                    df = pd.read_csv(candidate, encoding="utf-8", on_bad_lines="skip")
                    cols_lower = {c: c.lower().strip() for c in df.columns}
                    df = df.rename(columns=cols_lower)
                    # Map to canonical columns
                    text_col = None
                    label_col = None
                    for c in df.columns:
                        cl = c.lower()
                        if "text" in cl or "body" in cl or "content" in cl or "email" in cl:
                            text_col = c
                        if "type" in cl or "label" in cl or "class" in cl:
                            label_col = c
                    if text_col and label_col:
                        out = df[[label_col, text_col]].dropna()
                        out.columns = ["label", "text"]
                        out["label"] = out["label"].str.strip().str.lower()
                        # Normalise labels → phishing / safe
                        label_map = {
                            "phishing email": "phishing", "phishing": "phishing",
                            "safe email": "safe", "safe": "safe",
                            "spam": "phishing", "ham": "safe",
                            "1": "phishing", "0": "safe",
                        }
                        out["label"] = out["label"].map(
                            lambda x: label_map.get(x, x))
                        out.to_csv(out_csv, index=False)
                        _log(f"Kaggle → {len(out)} emails → {out_csv}", "OK")
                        return out_csv
                except Exception:
                    continue

    # Manual download instructions
    _print_manual_instructions(
        "Phishing Email Dataset", kaggle_slug, PHISHING_DIR, kaggle_url)

    # Offer to place a manual file
    for candidate in PHISHING_DIR.glob("*.csv"):
        if candidate.name != out_csv.name:
            try:
                df = pd.read_csv(candidate, encoding="utf-8", on_bad_lines="skip")
                if len(df) > 50:
                    _log(f"Found manually placed CSV: {candidate.name} ({len(df)} rows)", "OK")
                    # Best-effort column mapping
                    cols = list(df.columns)
                    text_col = next((c for c in cols if any(
                        k in c.lower() for k in ["text", "body", "content", "email"])), cols[-1])
                    label_col = next((c for c in cols if any(
                        k in c.lower() for k in ["type", "label", "class"])), cols[0])
                    out = df[[label_col, text_col]].dropna()
                    out.columns = ["label", "text"]
                    out["label"] = out["label"].str.strip().str.lower()
                    out.to_csv(out_csv, index=False)
                    return out_csv
            except Exception:
                continue

    _log("Phishing email dataset not available — will use synthetic fallback.", "WARN")
    return None


# ══════════════════════════════════════════════════════════════════════
#  4.  MALICIOUS URLs DATASET  (Kaggle)
# ══════════════════════════════════════════════════════════════════════

def download_malicious_urls() -> Optional[Path]:
    """
    Download the Kaggle "Malicious URLs Dataset".

    Kaggle slug: ``sid321axn/malicious-urls-dataset``
    Columns expected: ``url``, ``type`` (benign / phishing / malware / defacement)
    """
    _log("=" * 60)
    _log("Dataset 4/4 — Malicious URLs Dataset (Kaggle)")
    _log("=" * 60)

    out_csv = URL_DIR / "malicious_urls.csv"
    if out_csv.exists():
        df = pd.read_csv(out_csv)
        _log(f"Already exists ({len(df)} rows).  Skipping download.", "OK")
        return out_csv

    URL_DIR.mkdir(parents=True, exist_ok=True)
    kaggle_slug = "sid321axn/malicious-urls-dataset"
    kaggle_url  = f"https://www.kaggle.com/datasets/{kaggle_slug}"

    if _kaggle_available():
        if _kaggle_download(kaggle_slug, URL_DIR):
            for candidate in URL_DIR.glob("*.csv"):
                try:
                    df = pd.read_csv(candidate, encoding="utf-8", on_bad_lines="skip")
                    cols_lower = {c: c.lower().strip() for c in df.columns}
                    df = df.rename(columns=cols_lower)
                    url_col = next((c for c in df.columns if "url" in c.lower()), None)
                    type_col = next((c for c in df.columns if "type" in c.lower()
                                     or "label" in c.lower() or "class" in c.lower()), None)
                    if url_col and type_col:
                        out = df[[url_col, type_col]].dropna()
                        out.columns = ["url", "label"]
                        out["label"] = out["label"].str.strip().str.lower()
                        # Also store the URL as 'text' for unified pipeline compat
                        out["text"] = out["url"]
                        out.to_csv(out_csv, index=False)
                        _log(f"Kaggle → {len(out)} URLs → {out_csv}", "OK")
                        return out_csv
                except Exception:
                    continue

    _print_manual_instructions(
        "Malicious URLs Dataset", kaggle_slug, URL_DIR, kaggle_url)

    # Check for manually placed files
    for candidate in URL_DIR.glob("*.csv"):
        if candidate.name != out_csv.name:
            try:
                df = pd.read_csv(candidate, encoding="utf-8", on_bad_lines="skip")
                if len(df) > 50:
                    _log(f"Found manually placed CSV: {candidate.name} ({len(df)} rows)", "OK")
                    cols = list(df.columns)
                    url_col = next((c for c in cols if "url" in c.lower()), cols[0])
                    type_col = next((c for c in cols if any(
                        k in c.lower() for k in ["type", "label", "class"])), cols[-1])
                    out = df[[url_col, type_col]].dropna()
                    out.columns = ["url", "label"]
                    out["label"] = out["label"].str.strip().str.lower()
                    out["text"] = out["url"]
                    out.to_csv(out_csv, index=False)
                    return out_csv
            except Exception:
                continue

    _log("Malicious URLs dataset not available — will use synthetic fallback.", "WARN")
    return None


# ══════════════════════════════════════════════════════════════════════
#  SYNTHETIC FALLBACKS  (offline-safe)
# ══════════════════════════════════════════════════════════════════════

def generate_synthetic_sms(target: Path, n_ham: int = 4400, n_spam: int = 740) -> Path:
    """Generate a realistic synthetic SMS corpus for offline use."""
    _log(f"Generating synthetic SMS dataset ({n_ham + n_spam} messages)…")
    random.seed(42)

    ham_templates = [
        "Hey {name}, are we still meeting for lunch today at {place}?",
        "Can you send me the {topic} report when you get a chance?",
        "Thanks for the update, will review it by {day}.",
        "Don't forget mom's birthday dinner this {day} at 7pm.",
        "Just arrived at the station, see you in {n} mins.",
        "Please find attached the slides from the {topic} presentation.",
        "Where are you? We are waiting at {place}.",
        "Got your voicemail, I'll call you back after my {day} meeting.",
        "Ok sounds good to me. Let's touch base on {day}.",
        "Can you pick up milk and bread on your way home?",
        "Great job on the demo yesterday, the client loved the {topic} feature.",
        "I'm running {n} minutes late due to traffic.",
        "Have you seen my keys? I think I left them on the kitchen counter.",
        "Let's reschedule our 1-on-1 to next {day} if that works.",
        "Thanks for helping with the bug fix in the {topic} module.",
        "Movie at {n}? I heard the new one is good.",
        "I'm at {place} now, do you want me to order for you?",
        "Happy birthday! Hope you have a wonderful day 🎂",
        "Flight lands at {n}:30, can you pick me up from the airport?",
        "Reminder: dentist appointment tomorrow at {n} am.",
    ]
    spam_templates = [
        "CONGRATULATIONS! You won £{amount} cash prize! Call {phone} NOW to claim!",
        "URGENT: Your account suspended. Verify at {url} immediately.",
        "FREE entry in our £{amount} weekly draw! Text WIN to {phone}.",
        "Exclusive offer! Get {n}% OFF luxury watches today! Visit {url}",
        "You have 1 new message from {name} dating service. Reply CHAT to {phone}.",
        "ALERT: Refund of £{amount} pending on your tax. Click {url} to receive.",
        "Claim your complimentary voucher today! Call {phone} to confirm delivery.",
        "Guaranteed loan up to £{amount}! No credit check! Apply at {url}.",
        "Your parcel delivery failed. Pay £{n}.99 at {url} to reschedule.",
        "HOT investment: Earn £{amount}/day from home! Register {url} now.",
        "WINNER!! As a valued customer you've been selected for a £{amount} prize!",
        "Ur mobile has won a £{amount} prize! Claim by calling {phone}. T&Cs apply.",
        "Congrats! {n} people selected for free iPhone! Visit {url} in 24 hrs.",
        "IMPORTANT: Click {url} to confirm your identity or account will be closed.",
        "Free ringtone! Reply to this msg. Charges: £{n}/week. Text STOP to end.",
    ]

    names   = ["Alex", "Sarah", "John", "David", "Emma", "Michael", "Lisa",
               "Chris", "Jessica", "James", "Priya", "Carlos", "Aisha"]
    places  = ["downtown", "Starbucks", "the office", "Main Street", "the cafe",
               "Central Park", "the library", "Gate B12"]
    topics  = ["Q3 sales", "marketing", "backend", "UX research", "finance",
               "analytics", "deployment", "security audit"]
    days    = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    amounts = ["1000", "5000", "250", "750", "10000", "500", "2000", "900"]
    urls    = ["http://bit.ly/claimprize", "http://secure-alert.info/verify",
               "http://fast-cash.net/apply", "http://track-parcel.com/pay",
               "http://lottery-uk.win/entry", "http://tax-refund.co/claim"]
    phones  = ["08000930705", "09061743811", "08712460324", "07786200117",
               "08452810075", "09064015307", "08718726971"]

    data = []
    for _ in range(n_ham):
        t = random.choice(ham_templates)
        msg = t.format(name=random.choice(names), place=random.choice(places),
                       topic=random.choice(topics), day=random.choice(days),
                       n=random.randint(2, 45))
        data.append({"label": "ham", "text": msg})
    for _ in range(n_spam):
        t = random.choice(spam_templates)
        msg = t.format(name=random.choice(names), phone=random.choice(phones),
                       url=random.choice(urls), amount=random.choice(amounts),
                       n=random.randint(2, 90))
        data.append({"label": "spam", "text": msg})
    random.shuffle(data)

    df = pd.DataFrame(data)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    _log(f"Synthetic SMS: {len(df)} messages → {target}", "OK")
    return target


def generate_synthetic_email(target: Path, n_ham: int = 2800, n_spam: int = 700) -> Path:
    """Generate a synthetic email spam corpus."""
    _log(f"Generating synthetic email dataset ({n_ham + n_spam} emails)…")
    random.seed(43)

    ham_subjects = [
        "Re: Meeting agenda for {day}",
        "Project {topic} — status update",
        "Invoice #{n} attached",
        "Lunch plans?",
        "Team standup notes — {day}",
        "FYI: Server maintenance tonight",
        "Happy holidays from the {topic} team",
        "Quick question about {topic}",
        "Out of office: back on {day}",
    ]
    spam_subjects = [
        "You WON!!! Claim your £{amount} prize NOW",
        "URGENT: verify your account immediately",
        "FREE {topic} - limited time offer",
        "Lose weight FAST — guaranteed results",
        "Make \${amount} per day working from home",
        "Your order #{n} has shipped — confirm delivery",
        "Important security alert — action required",
        "Exclusive deal just for you — {n}% OFF",
    ]

    names  = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace"]
    topics = ["marketing", "Q4 budget", "cloud migration", "hiring plan",
              "product launch", "Viagra", "Rolex watches", "weight loss pills"]
    days   = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    amounts = ["1000", "5000", "250", "50000"]

    data = []
    for _ in range(n_ham):
        subj = random.choice(ham_subjects).format(
            day=random.choice(days), topic=random.choice(topics[:5]),
            n=random.randint(1000, 9999), amount=random.choice(amounts))
        body_lines = random.randint(2, 6)
        body = f"Subject: {subj}\n\n" + "\n".join(
            [f"Paragraph {i+1} of a normal work email about {random.choice(topics[:5])}."
             for i in range(body_lines)])
        data.append({"label": "ham", "text": body})
    for _ in range(n_spam):
        subj = random.choice(spam_subjects).format(
            topic=random.choice(topics[5:]), n=random.randint(10, 95),
            amount=random.choice(amounts))
        body = (f"Subject: {subj}\n\n"
                f"Dear valued customer,\n"
                f"Act now to claim your prize of £{random.choice(amounts)}! "
                f"Visit http://scam-{random.randint(100,999)}.example.com to claim. "
                f"This offer expires in {random.randint(1,24)} hours!\n"
                f"Call {random.randint(8000000000, 9999999999)} immediately.")
        data.append({"label": "spam", "text": body})
    random.shuffle(data)

    df = pd.DataFrame(data)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    _log(f"Synthetic email: {len(df)} emails → {target}", "OK")
    return target


def generate_synthetic_phishing(target: Path, n_safe: int = 3000,
                                 n_phishing: int = 1500) -> Path:
    """Generate a synthetic phishing email corpus."""
    _log(f"Generating synthetic phishing dataset ({n_safe + n_phishing} emails)…")
    random.seed(44)

    safe_templates = [
        "Hi {name}, here are the meeting notes from {day}'s standup.",
        "Please review the attached {topic} document at your earliest convenience.",
        "Reminder: the {topic} deadline is next {day}.",
        "Thank you for your purchase. Your order #{n} has been confirmed.",
        "Your monthly statement for account ending in {n} is now available.",
        "Team outing scheduled for {day} — please RSVP by end of week.",
        "The {topic} release is scheduled for next {day}. Please test your changes.",
        "Hi {name}, I'd like to schedule a 1:1 to discuss your {topic} progress.",
    ]
    phishing_templates = [
        "URGENT: Your {bank} account has been compromised. Click {url} to secure it now.",
        "Your {bank} password expires today. Reset it immediately at {url}.",
        "We detected unusual sign-in activity on your {bank} account from {country}. Verify: {url}",
        "Your package from {carrier} could not be delivered. Confirm address: {url}",
        "Congratulations {name}! You've been selected for a ${amount} reward. Claim: {url}",
        "ACTION REQUIRED: Your {bank} account will be suspended unless you verify at {url}",
        "Dear customer, your recent {bank} transaction of ${amount} needs verification: {url}",
        "IT Department: Your mailbox is full. Click {url} to upgrade storage immediately.",
        "HR Notice: Review and sign your updated employment agreement at {url}",
        "Invoice #{n} overdue — payment required within 24 hours. Pay at {url}",
    ]

    names    = ["John", "Sarah", "David", "Emma", "Michael", "Lisa", "Carlos"]
    topics   = ["Q3 review", "marketing plan", "server migration", "hiring"]
    days     = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    banks    = ["PayPal", "Bank of America", "Chase", "Wells Fargo", "HSBC", "Apple ID"]
    carriers = ["FedEx", "UPS", "DHL", "Royal Mail", "USPS"]
    countries = ["Russia", "China", "Nigeria", "Unknown location", "Brazil"]
    urls     = ["http://secure-verify.example.com", "http://account-update.phish.com",
                "http://login-check.scam.net", "http://delivery-confirm.fake.org"]
    amounts  = ["500", "1000", "2500", "50", "10000"]

    data = []
    for _ in range(n_safe):
        t = random.choice(safe_templates)
        msg = t.format(name=random.choice(names), topic=random.choice(topics),
                       day=random.choice(days), n=random.randint(1000, 99999))
        data.append({"label": "safe", "text": msg})
    for _ in range(n_phishing):
        t = random.choice(phishing_templates)
        msg = t.format(name=random.choice(names), bank=random.choice(banks),
                       carrier=random.choice(carriers), country=random.choice(countries),
                       url=random.choice(urls), amount=random.choice(amounts),
                       n=random.randint(10000, 99999))
        data.append({"label": "phishing", "text": msg})
    random.shuffle(data)

    df = pd.DataFrame(data)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    _log(f"Synthetic phishing: {len(df)} emails → {target}", "OK")
    return target


def generate_synthetic_urls(target: Path, n: int = 5000) -> Path:
    """Generate a synthetic malicious URLs dataset."""
    _log(f"Generating synthetic URL dataset ({n} URLs)…")
    random.seed(45)

    tlds_safe = [".com", ".org", ".edu", ".gov", ".net", ".io", ".co.uk"]
    tlds_bad  = [".xyz", ".top", ".tk", ".ml", ".ga", ".cf", ".info", ".biz"]
    domains_safe  = ["google", "github", "stackoverflow", "wikipedia",
                     "microsoft", "amazon", "apple", "bbc", "nytimes",
                     "linkedin", "python", "numpy", "ubuntu"]
    domains_bad   = ["free-prize-win", "secure-login-verify", "paypai-update",
                     "account-alert", "bankofamerica-secure",
                     "delivery-track-confirm", "lottery-winner-claim",
                     "cheap-meds-online", "hack-tool-free"]
    paths_safe = ["/", "/about", "/docs", "/blog", "/products", "/help",
                  "/api/v2", "/en/latest/", "/search?q=python"]
    paths_bad  = ["/login.php?id=", "/verify?token=", "/wp-admin/update.php",
                  "/download/free.exe", "/claim?ref=", "/.env",
                  "/admin/shell.php", "/redirect?to="]

    types = {"benign": 0.5, "phishing": 0.2, "malware": 0.15, "defacement": 0.15}
    data = []
    for label, frac in types.items():
        count = int(n * frac)
        for _ in range(count):
            if label == "benign":
                d = random.choice(domains_safe)
                t = random.choice(tlds_safe)
                p = random.choice(paths_safe)
                scheme = random.choice(["https://", "https://www."])
                url = f"{scheme}{d}{t}{p}"
            else:
                d = random.choice(domains_bad)
                t = random.choice(tlds_bad)
                p = random.choice(paths_bad) + str(random.randint(1000, 99999))
                scheme = random.choice(["http://", "https://", "http://www."])
                url = f"{scheme}{d}{t}{p}"
            data.append({"url": url, "label": label, "text": url})
    random.shuffle(data)

    df = pd.DataFrame(data)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    _log(f"Synthetic URLs: {len(df)} URLs → {target}", "OK")
    return target


# ══════════════════════════════════════════════════════════════════════
#  VERIFICATION
# ══════════════════════════════════════════════════════════════════════

def verify_datasets() -> None:
    """Print a summary of all downloaded datasets and their shapes."""
    _log("=" * 60)
    _log("Dataset Verification Report")
    _log("=" * 60)

    datasets = [
        ("SMS Spam Collection", SMS_DIR / "sms_spam.csv"),
        ("SpamAssassin Email",  EMAIL_DIR / "email_spam.csv"),
        ("Phishing Email",      PHISHING_DIR / "phishing_email.csv"),
        ("Malicious URLs",      URL_DIR / "malicious_urls.csv"),
    ]

    for name, path in datasets:
        if path.exists():
            try:
                df = pd.read_csv(path, nrows=None)
                labels = df["label"].value_counts().to_dict()
                labels_str = ", ".join(f"{k}: {v}" for k, v in labels.items())
                size_mb = path.stat().st_size / (1024 * 1024)
                _log(f"✓  {name:<25s}  {len(df):>7,} rows  ({size_mb:.1f} MB)  [{labels_str}]", "OK")
            except Exception as exc:
                _log(f"✗  {name:<25s}  file exists but unreadable ({exc})", "ERR")
        else:
            _log(f"✗  {name:<25s}  NOT FOUND at {path}", "WARN")


# ══════════════════════════════════════════════════════════════════════
#  UNIFIED LOADER (used by the rest of the project)
# ══════════════════════════════════════════════════════════════════════

DATASET_REGISTRY: Dict[str, Dict] = {
    "sms": {
        "name": "SMS Spam Collection",
        "path": SMS_DIR / "sms_spam.csv",
        "downloader": download_sms_spam,
        "fallback": lambda: generate_synthetic_sms(SMS_DIR / "sms_spam.csv"),
        "label_map": {"ham": 0, "spam": 1},
        "description": "5,574 SMS messages labelled ham/spam (UCI)",
    },
    "email": {
        "name": "SpamAssassin Email Corpus",
        "path": EMAIL_DIR / "email_spam.csv",
        "downloader": download_email_spam,
        "fallback": lambda: generate_synthetic_email(EMAIL_DIR / "email_spam.csv"),
        "label_map": {"ham": 0, "spam": 1},
        "description": "~6,000 emails labelled ham/spam (Apache SpamAssassin)",
    },
    "phishing": {
        "name": "Phishing Email Dataset",
        "path": PHISHING_DIR / "phishing_email.csv",
        "downloader": download_phishing_email,
        "fallback": lambda: generate_synthetic_phishing(PHISHING_DIR / "phishing_email.csv"),
        "label_map": {"safe": 0, "phishing": 1},
        "description": "Emails labelled phishing/safe (Kaggle)",
    },
    "urls": {
        "name": "Malicious URLs Dataset",
        "path": URL_DIR / "malicious_urls.csv",
        "downloader": download_malicious_urls,
        "fallback": lambda: generate_synthetic_urls(URL_DIR / "malicious_urls.csv"),
        "label_map": {"benign": 0, "phishing": 1, "malware": 2, "defacement": 3},
        "description": "URLs labelled benign/phishing/malware/defacement (Kaggle)",
    },
}


def load_dataset(dataset_key: str, auto_download: bool = True) -> pd.DataFrame:
    """
    Load one of the four project datasets by key.

    Parameters
    ----------
    dataset_key : str
        One of: ``"sms"``, ``"email"``, ``"phishing"``, ``"urls"``
    auto_download : bool
        If True and the file is missing, attempt to download it (then fallback
        to synthetic).

    Returns
    -------
    pd.DataFrame
        Columns: ``label``, ``text``, ``target`` (int-encoded label).
        For the URL dataset there is also a ``url`` column.
    """
    if dataset_key not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_key}'.  "
            f"Choose from: {list(DATASET_REGISTRY.keys())}")

    info = DATASET_REGISTRY[dataset_key]
    csv_path = info["path"]

    if not csv_path.exists() and auto_download:
        result = info["downloader"]()
        if result is None:
            info["fallback"]()

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset '{dataset_key}' not found at {csv_path}.\n"
            f"Run:  python download_data.py --{dataset_key}")

    df = pd.read_csv(csv_path)
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    if "text" not in df.columns and "message" in df.columns:
        df = df.rename(columns={"message": "text"})
    df["target"] = df["label"].map(info["label_map"])
    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)
    return df


# ══════════════════════════════════════════════════════════════════════
#  CLI ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download & manage all project datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
          python download_data.py --all          # Download everything
          python download_data.py --sms --email  # Download only SMS and Email
          python download_data.py --synthetic    # Generate offline fallbacks
          python download_data.py --verify       # Check what's downloaded
        """),
    )
    parser.add_argument("--all",       action="store_true", help="Download all four datasets")
    parser.add_argument("--sms",       action="store_true", help="SMS Spam Collection (UCI)")
    parser.add_argument("--email",     action="store_true", help="SpamAssassin Email Corpus")
    parser.add_argument("--phishing",  action="store_true", help="Phishing Email Dataset (Kaggle)")
    parser.add_argument("--urls",      action="store_true", help="Malicious URLs Dataset (Kaggle)")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic fallbacks for all")
    parser.add_argument("--verify",    action="store_true", help="Verify existing datasets")

    args = parser.parse_args()

    # Default to --all if nothing specified
    if not any([args.all, args.sms, args.email, args.phishing, args.urls,
                args.synthetic, args.verify]):
        args.all = True

    print()
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║     EMAIL & SMS SPAM DETECTION — DATASET MANAGER             ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()

    if args.synthetic:
        generate_synthetic_sms(SMS_DIR / "sms_spam.csv")
        generate_synthetic_email(EMAIL_DIR / "email_spam.csv")
        generate_synthetic_phishing(PHISHING_DIR / "phishing_email.csv")
        generate_synthetic_urls(URL_DIR / "malicious_urls.csv")
        print()
        verify_datasets()
        return

    if args.all or args.sms:
        result = download_sms_spam()
        if result is None:
            generate_synthetic_sms(SMS_DIR / "sms_spam.csv")

    if args.all or args.email:
        result = download_email_spam()
        if result is None:
            generate_synthetic_email(EMAIL_DIR / "email_spam.csv")

    if args.all or args.phishing:
        result = download_phishing_email()
        if result is None:
            generate_synthetic_phishing(PHISHING_DIR / "phishing_email.csv")

    if args.all or args.urls:
        result = download_malicious_urls()
        if result is None:
            generate_synthetic_urls(URL_DIR / "malicious_urls.csv")

    print()
    verify_datasets()

    if args.verify and not any([args.all, args.sms, args.email, args.phishing, args.urls]):
        verify_datasets()


if __name__ == "__main__":
    main()

