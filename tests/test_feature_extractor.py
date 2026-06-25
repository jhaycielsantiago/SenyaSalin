"""
Tests for FeatureExtractor.

Validates output shapes and aggregation strategies.
"""

import numpy as np
import pytest
from extraction.landmark_extractor import (
    GestureLandmarkRecord,
    FrameLandmarks,
    HandLandmarks,
)
from preprocessing.feature_extractor import FeatureExtractor


def make_fake_hand(detected=True) -> HandLandmarks:
    lm = [{"id": i, "x": float(i) * 0.01, "y": float(i) * 0.02, "z": 0.0} for i in range(21)]
    return HandLandmarks(detected=detected, landmarks=lm if detected else [])


def make_fake_record(num_frames: int = 10, label: str = "HELP") -> GestureLandmarkRecord:
    frames = [
        FrameLandmarks(
            frame_index=i,
            timestamp_ms=float(i * 33),
            left_hand=make_fake_hand(),
            right_hand=make_fake_hand(),
        )
        for i in range(num_frames)
    ]
    return GestureLandmarkRecord(
        label=label,
        signer_id="test",
        session_id="test_session",
        dataset_version="0.1",
        frames=frames,
    )


class TestFeatureExtractor:
    def test_mean_aggregation_shape(self):
        record = make_fake_record(num_frames=30)
        extractor = FeatureExtractor(aggregation="mean")
        vec = extractor.extract_from_record(record)
        assert vec.shape == (126,)

    def test_mean_std_aggregation_shape(self):
        record = make_fake_record(num_frames=30)
        extractor = FeatureExtractor(aggregation="mean_std")
        vec = extractor.extract_from_record(record)
        assert vec.shape == (252,)

    def test_mean_std_max_aggregation_shape(self):
        record = make_fake_record(num_frames=30)
        extractor = FeatureExtractor(aggregation="mean_std_max")
        vec = extractor.extract_from_record(record)
        assert vec.shape == (378,)

    def test_empty_record_returns_zeros(self):
        record = make_fake_record(num_frames=0)
        extractor = FeatureExtractor(aggregation="mean")
        vec = extractor.extract_from_record(record)
        assert vec.shape == (126,)
        assert vec.sum() == pytest.approx(0.0)

    def test_feature_dim_property(self):
        for agg, expected in [("mean", 126), ("mean_std", 252), ("mean_std_max", 378)]:
            extractor = FeatureExtractor(aggregation=agg)
            assert extractor.feature_dim == expected
