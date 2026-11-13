"""
Models package for classic and deep learning spam classifiers.
"""

from .classic_models import get_classic_models
from .deep_models import get_deep_learning_models

__all__ = ["get_classic_models", "get_deep_learning_models"]

