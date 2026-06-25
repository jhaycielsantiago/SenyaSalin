"""
senyasalin — core inference package.

This package is the public API for running gesture recognition in applications.
It wraps the extraction, preprocessing, and model inference into a single
Pipeline class that accepts a webcam frame and returns a recognition result.

Usage:
    from senyasalin import GestureRecognizer

    recognizer = GestureRecognizer.load("models/saved/knn.pkl")
    result = recognizer.recognize(bgr_frame)
    print(result.label, result.phrase, result.confidence)
"""

from .pipeline import GestureRecognizer, RecognitionResult
from .mapper import IntentMapper

__all__ = ["GestureRecognizer", "RecognitionResult", "IntentMapper"]
