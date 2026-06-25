"""
Gesture recognition pipeline — the core inference path.

Accepts a BGR webcam frame, extracts landmarks, normalizes them,
classifies the gesture, and maps it to an intent and phrase.

This is the only module that application code needs to import.
Everything else (extraction, preprocessing, training) is infrastructure
that feeds into this pipeline.

The pipeline is intentionally stateless across frames — it classifies
each frame independently. Temporal smoothing (e.g., voting across N frames)
is the caller's responsibility and can be implemented in the app layer.

Future contributors: add temporal smoothing, confidence thresholding, and
multi-gesture sequencing at the app layer without modifying this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import json
import numpy as np

from extraction.landmark_extractor import LandmarkExtractor, FrameLandmarks
from preprocessing.normalizer import LandmarkNormalizer
from .mapper import IntentMapper


@dataclass
class RecognitionResult:
    """The output of a single gesture recognition call."""
    label: Optional[str]        # Gesture class: "HELP", "WATER", etc. (None if below threshold)
    intent: Optional[str]       # Intent: "request_assistance" (None if label is None)
    phrase: Optional[str]       # Natural language: "Kailangan ko po ng tulong."
    confidence: float           # Classifier confidence [0.0, 1.0]
    tts_lang_code: str = "tl"  # BCP-47 code for TTS
    below_threshold: bool = False  # True if confidence < min_confidence


class GestureRecognizer:
    """
    End-to-end gesture recognition pipeline.

    Usage:
        recognizer = GestureRecognizer.load("models/saved/knn.pkl")
        result = recognizer.recognize(bgr_frame)

    Args:
        model: A trained sklearn-compatible classifier.
        class_names: List of gesture labels (index = encoded class).
        normalizer: LandmarkNormalizer (must match training normalization).
        mapper: IntentMapper for label → phrase mapping.
        min_confidence: Minimum confidence to return a label (else below_threshold=True).
    """

    def __init__(
        self,
        model,
        class_names: list[str],
        normalizer: LandmarkNormalizer | None = None,
        mapper: IntentMapper | None = None,
        min_confidence: float = 0.70,
    ) -> None:
        self._model = model
        self._class_names = class_names
        self._normalizer = normalizer or LandmarkNormalizer()
        self._mapper = mapper or IntentMapper()
        self._min_confidence = min_confidence
        self._extractor = LandmarkExtractor()

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        gestures_path: str | Path = "schemas/gestures.json",
        mappings_path: str | Path = "schemas/mappings.json",
        language: str = "taglish",
        min_confidence: float = 0.70,
    ) -> "GestureRecognizer":
        """
        Load a trained model from disk.

        Args:
            model_path: Path to a .pkl model file (e.g. models/saved/knn.pkl)
            gestures_path: Path to schemas/gestures.json
            mappings_path: Path to schemas/mappings.json
            language: Output language for phrases
            min_confidence: Minimum confidence threshold
        """
        model_path = Path(model_path)
        model = joblib.load(model_path)

        meta_path = model_path.with_suffix(".meta.json")
        class_names: list[str] = []
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            class_names = meta.get("class_names", [])

        return cls(
            model=model,
            class_names=class_names,
            normalizer=LandmarkNormalizer(),
            mapper=IntentMapper(gestures_path, mappings_path, language),
            min_confidence=min_confidence,
        )

    def recognize(self, bgr_frame: np.ndarray) -> RecognitionResult:
        """
        Recognize a gesture from a single BGR webcam frame.

        Steps:
        1. Extract landmarks (MediaPipe Hands)
        2. Normalize left + right hand landmarks
        3. Classify with the loaded model
        4. Map class index → label → intent → phrase

        Returns RecognitionResult. If no hand is detected or confidence
        is below threshold, label/intent/phrase will be None.
        """
        # --- Extract ---
        frame_lm = self._extractor.extract_frame(bgr_frame)

        if not frame_lm.left_hand.detected and not frame_lm.right_hand.detected:
            return RecognitionResult(
                label=None, intent=None, phrase=None,
                confidence=0.0, below_threshold=True,
            )

        # --- Normalize ---
        feature_vec = self._normalizer.normalize_frame(
            left_hand_landmarks=frame_lm.left_hand.landmarks if frame_lm.left_hand.detected else [],
            right_hand_landmarks=frame_lm.right_hand.landmarks if frame_lm.right_hand.detected else [],
        )

        # --- Classify ---
        feature_2d = feature_vec.reshape(1, -1)
        class_idx = int(self._model.predict(feature_2d)[0])
        confidence = 0.0

        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba(feature_2d)[0]
            confidence = float(proba[class_idx])

        if class_idx >= len(self._class_names):
            return RecognitionResult(
                label=None, intent=None, phrase=None,
                confidence=confidence, below_threshold=True,
            )

        label = self._class_names[class_idx]

        if confidence < self._min_confidence:
            return RecognitionResult(
                label=label, intent=None, phrase=None,
                confidence=confidence, below_threshold=True,
            )

        # --- Map ---
        intent = self._mapper.get_intent(label)
        phrase = self._mapper.get_phrase(label)
        tts_code = self._mapper.get_tts_lang_code(label)

        return RecognitionResult(
            label=label,
            intent=intent,
            phrase=phrase,
            confidence=confidence,
            tts_lang_code=tts_code,
            below_threshold=False,
        )

    def recognize_from_landmarks(self, frame_lm: FrameLandmarks) -> RecognitionResult:
        """
        Classify from pre-extracted FrameLandmarks (useful for testing/notebooks).
        """
        feature_vec = self._normalizer.normalize_frame(
            left_hand_landmarks=frame_lm.left_hand.landmarks if frame_lm.left_hand.detected else [],
            right_hand_landmarks=frame_lm.right_hand.landmarks if frame_lm.right_hand.detected else [],
        )

        feature_2d = feature_vec.reshape(1, -1)
        class_idx = int(self._model.predict(feature_2d)[0])
        confidence = 0.0

        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba(feature_2d)[0]
            confidence = float(proba[class_idx])

        label = self._class_names[class_idx] if class_idx < len(self._class_names) else None

        if not label or confidence < self._min_confidence:
            return RecognitionResult(
                label=label, intent=None, phrase=None,
                confidence=confidence, below_threshold=True,
            )

        return RecognitionResult(
            label=label,
            intent=self._mapper.get_intent(label),
            phrase=self._mapper.get_phrase(label),
            confidence=confidence,
            tts_lang_code=self._mapper.get_tts_lang_code(label),
            below_threshold=False,
        )

    def close(self) -> None:
        self._extractor.close()

    def __enter__(self) -> "GestureRecognizer":
        return self

    def __exit__(self, *args) -> None:
        self.close()
