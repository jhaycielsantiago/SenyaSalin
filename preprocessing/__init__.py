"""
Preprocessing module.

Converts raw landmark JSON records into normalized feature vectors
suitable for training classifiers. Handles:
- Translation invariance (wrist-relative normalization)
- Scale invariance (hand-size normalization)
- Missing hand padding
- Frame aggregation (per-recording → single feature vector)
"""

from .normalizer import LandmarkNormalizer
from .feature_extractor import FeatureExtractor
from .data_loader import DataLoader

__all__ = ["LandmarkNormalizer", "FeatureExtractor", "DataLoader"]
