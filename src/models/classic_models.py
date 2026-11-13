"""
Classic Machine Learning Models for Spam Detection.
Provides standard configurations and scikit-learn compatible wrappers.
"""

from typing import Dict, Any
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def get_classic_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Returns a dictionary of classic ML models configured for sparse text features.
    CalibratedClassifierCV is used for LinearSVC to ensure reliable probability estimates.
    """
    return {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.2),
        "Logistic Regression": LogisticRegression(
            C=1.5,
            max_iter=1000,
            class_weight='balanced',
            random_state=random_state
        ),
        "Support Vector Machine (LinearSVC)": CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight='balanced', random_state=random_state),
            cv=3
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=30,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=random_state
        )
    }

