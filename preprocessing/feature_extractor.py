"""
Feature extraction — converts multi-frame landmark records into fixed-size
feature vectors for training.

Strategy: aggregate across frames by computing statistics (mean, std, max)
over all frames in a recording. This collapses variable-length sequences into
a fixed-length vector without requiring sequence models, enabling the KNN/RF/MLP
baselines to run without padding/masking logic.

Future contributors: For sequence-aware models (LSTM, Transformer), skip
aggregation and return the raw frame sequence instead. The DataLoader supports
both via the `aggregate` parameter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from .normalizer import LandmarkNormalizer
from extraction.landmark_extractor import GestureLandmarkRecord


AggregationStrategy = Literal["mean", "mean_std", "mean_std_max"]


class FeatureExtractor:
    """
    Converts a GestureLandmarkRecord into a fixed-size feature vector.

    Args:
        normalizer: LandmarkNormalizer instance.
        aggregation: How to collapse frame-level vectors.
            - "mean": Mean across frames → shape (126,)
            - "mean_std": Mean + std → shape (252,)
            - "mean_std_max": Mean + std + max → shape (378,)
    """

    FEATURE_DIM = {
        "mean": 126,
        "mean_std": 252,
        "mean_std_max": 378,
    }

    def __init__(
        self,
        normalizer: LandmarkNormalizer | None = None,
        aggregation: AggregationStrategy = "mean",
    ) -> None:
        self.normalizer = normalizer or LandmarkNormalizer()
        self.aggregation = aggregation

    @property
    def feature_dim(self) -> int:
        return self.FEATURE_DIM[self.aggregation]

    def extract_from_record(self, record: GestureLandmarkRecord) -> np.ndarray:
        """
        Extract a single feature vector from a GestureLandmarkRecord.

        Returns shape (feature_dim,).
        """
        frame_vectors = []
        for frame in record.frames:
            vec = self.normalizer.normalize_frame(
                left_hand_landmarks=frame.left_hand.landmarks if frame.left_hand.detected else [],
                right_hand_landmarks=frame.right_hand.landmarks if frame.right_hand.detected else [],
            )
            frame_vectors.append(vec)

        if not frame_vectors:
            return np.zeros(self.feature_dim, dtype=np.float32)

        frames_arr = np.stack(frame_vectors, axis=0)  # (T, 126)
        return self._aggregate(frames_arr)

    def extract_from_file(self, path: Path) -> tuple[str, np.ndarray]:
        """
        Load a landmark JSON file and extract (label, feature_vector).
        """
        record = GestureLandmarkRecord.load(path)
        return record.label, self.extract_from_record(record)

    def _aggregate(self, frames: np.ndarray) -> np.ndarray:
        """Collapse (T, 126) → (feature_dim,)"""
        mean = frames.mean(axis=0)
        if self.aggregation == "mean":
            return mean

        std = frames.std(axis=0)
        if self.aggregation == "mean_std":
            return np.concatenate([mean, std])

        # mean_std_max
        max_ = frames.max(axis=0)
        return np.concatenate([mean, std, max_])
