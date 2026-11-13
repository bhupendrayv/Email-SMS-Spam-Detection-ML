"""
Quick environment and setup check.
"""

import sys

def check():
    print(f"Python version: {sys.version}")
    packages = ["numpy", "pandas", "sklearn", "joblib", "nltk", "streamlit", "plotly"]
    for pkg in packages:
        try:
            m = __import__(pkg)
            print(f"✓ {pkg}: {getattr(m, '__version__', 'installed')}")
        except ImportError:
            print(f"✗ {pkg}: NOT installed")

if __name__ == "__main__":
    check()

