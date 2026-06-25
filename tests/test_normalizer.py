"""
Tests for LandmarkNormalizer.

Verifies translation invariance and scale invariance properties
that the gesture classifier depends on.
"""

import numpy as np
import pytest
from preprocessing.normalizer import LandmarkNormalizer, normalize_landmarks


def make_landmarks(pts: list[tuple[float, float, float]]) -> list[dict]:
    return [{"id": i, "x": x, "y": y, "z": z} for i, (x, y, z) in enumerate(pts)]


def make_hand_21(offset=(0.0, 0.0, 0.0), scale=1.0) -> list[dict]:
    """21 landmarks at predictable positions."""
    ox, oy, oz = offset
    pts = [(ox + scale * i * 0.01, oy + scale * i * 0.01, oz) for i in range(21)]
    return make_landmarks(pts)


class TestNormalizerTranslationInvariance:
    def test_wrist_becomes_origin(self):
        lm = make_hand_21(offset=(0.3, 0.4, 0.0))
        vec = normalize_landmarks(lm, "wrist_relative")
        # Landmark 0 (wrist) should be at origin
        assert vec[0] == pytest.approx(0.0)
        assert vec[1] == pytest.approx(0.0)
        assert vec[2] == pytest.approx(0.0)

    def test_translation_invariance(self):
        lm_a = make_hand_21(offset=(0.1, 0.2, 0.0))
        lm_b = make_hand_21(offset=(0.8, 0.9, 0.0))
        vec_a = normalize_landmarks(lm_a, "wrist_relative")
        vec_b = normalize_landmarks(lm_b, "wrist_relative")
        np.testing.assert_allclose(vec_a, vec_b, atol=1e-6)


class TestNormalizerScaleInvariance:
    def test_scale_invariance(self):
        lm_small = make_hand_21(scale=0.5)
        lm_large = make_hand_21(scale=2.0)
        vec_small = normalize_landmarks(lm_small, "wrist_relative_unit_scale")
        vec_large = normalize_landmarks(lm_large, "wrist_relative_unit_scale")
        np.testing.assert_allclose(vec_small, vec_large, atol=1e-5)


class TestNormalizerFrame:
    def test_frame_vector_shape(self):
        norm = LandmarkNormalizer()
        lm = make_hand_21()
        vec = norm.normalize_frame(lm, lm)
        assert vec.shape == (126,)

    def test_missing_hand_is_zeros(self):
        norm = LandmarkNormalizer()
        vec = norm.normalize_frame([], make_hand_21())
        assert vec[:63].sum() == pytest.approx(0.0)
        assert vec[63:].sum() != 0.0

    def test_empty_landmarks_returns_zeros(self):
        vec = normalize_landmarks([], "wrist_relative_unit_scale")
        assert vec.shape == (63,)
        assert vec.sum() == pytest.approx(0.0)
