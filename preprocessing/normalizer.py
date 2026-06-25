"""
Landmark normalization.

Two normalization strategies are implemented:
1. wrist_relative: Translates all landmarks so the wrist (landmark 0) is at origin.
2. unit_scale: After wrist-relative, scales so the max hand dimension = 1.0.

Both produce translation-invariant and scale-invariant feature vectors,
which is critical for gesture classifiers to generalize across users,
screen distances, and hand sizes.

Future contributors: add new strategies by subclassing NormalizerBase
or implementing the normalize_landmarks() protocol.
"""

from __future__ import annotations

import numpy as np


def normalize_landmarks(
    landmarks: list[dict],
    strategy: str = "wrist_relative_unit_scale",
) -> np.ndarray:
    """
    Normalize a list of 21 MediaPipe hand landmarks.

    Args:
        landmarks: List of {"id": int, "x": float, "y": float, "z": float}
        strategy: One of:
            - "raw": No normalization (useful for debugging)
            - "wrist_relative": Subtract wrist position
            - "wrist_relative_unit_scale": Wrist-relative + scale by max distance

    Returns:
        Flat numpy array of shape (63,): [x0,y0,z0, ..., x20,y20,z20]
    """
    if not landmarks:
        return np.zeros(63, dtype=np.float32)

    pts = np.array([[lm["x"], lm["y"], lm["z"]] for lm in landmarks], dtype=np.float32)

    if strategy == "raw":
        return pts.flatten()

    # Wrist (landmark 0) becomes origin
    wrist = pts[0].copy()
    pts -= wrist

    if strategy == "wrist_relative":
        return pts.flatten()

    if strategy == "wrist_relative_unit_scale":
        # Scale by the max distance from the wrist across all landmarks
        max_dist = np.max(np.linalg.norm(pts, axis=1))
        if max_dist > 1e-6:
            pts /= max_dist
        return pts.flatten()

    raise ValueError(f"Unknown normalization strategy: {strategy!r}")


def normalize_frame_vector(
    left_hand_landmarks: list[dict],
    right_hand_landmarks: list[dict],
    strategy: str = "wrist_relative_unit_scale",
) -> np.ndarray:
    """
    Normalize both hands from a single frame into a combined feature vector.

    Returns shape (126,): [left_hand(63), right_hand(63)]
    If a hand is missing (empty list), its 63 values are all zeros.
    """
    left_vec = normalize_landmarks(left_hand_landmarks, strategy)
    right_vec = normalize_landmarks(right_hand_landmarks, strategy)
    return np.concatenate([left_vec, right_vec], dtype=np.float32)


class LandmarkNormalizer:
    """
    Stateless normalizer that applies a named strategy.

    Provides a consistent interface for use in both training and inference.
    """

    SUPPORTED_STRATEGIES = {
        "raw",
        "wrist_relative",
        "wrist_relative_unit_scale",
    }

    def __init__(self, strategy: str = "wrist_relative_unit_scale") -> None:
        if strategy not in self.SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unknown strategy {strategy!r}. Choose from {self.SUPPORTED_STRATEGIES}"
            )
        self.strategy = strategy

    def normalize_hand(self, landmarks: list[dict]) -> np.ndarray:
        return normalize_landmarks(landmarks, self.strategy)

    def normalize_frame(
        self,
        left_hand_landmarks: list[dict],
        right_hand_landmarks: list[dict],
    ) -> np.ndarray:
        return normalize_frame_vector(
            left_hand_landmarks, right_hand_landmarks, self.strategy
        )
