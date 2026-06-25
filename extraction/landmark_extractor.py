"""
MediaPipe Hands landmark extractor.

Wraps MediaPipe Hands to extract 21-point 3D landmarks per hand per frame.
Output conforms to schemas/landmark_schema.json.

Future contributors: to add pose landmarks (MediaPipe Pose), extend the
`extract_frame` method to return a `pose` key alongside `left_hand`/`right_hand`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
import mediapipe as mp
import numpy as np


@dataclass
class HandLandmarks:
    detected: bool
    landmarks: list[dict]  # [{"id": int, "x": float, "y": float, "z": float}]
    handedness_score: float = 0.0

    def to_flat_vector(self) -> np.ndarray:
        """Returns a flat (63,) vector: [x0,y0,z0, x1,y1,z1, ..., x20,y20,z20].
        If not detected, returns zeros. Used as input to classifiers."""
        if not self.detected:
            return np.zeros(63, dtype=np.float32)
        return np.array(
            [[lm["x"], lm["y"], lm["z"]] for lm in self.landmarks],
            dtype=np.float32,
        ).flatten()


@dataclass
class FrameLandmarks:
    frame_index: int
    timestamp_ms: float
    left_hand: HandLandmarks
    right_hand: HandLandmarks

    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "timestamp_ms": self.timestamp_ms,
            "left_hand": {
                "detected": self.left_hand.detected,
                "landmarks": self.left_hand.landmarks,
                "handedness_score": self.left_hand.handedness_score,
            },
            "right_hand": {
                "detected": self.right_hand.detected,
                "landmarks": self.right_hand.landmarks,
                "handedness_score": self.right_hand.handedness_score,
            },
        }


@dataclass
class GestureLandmarkRecord:
    """A complete landmark recording for one gesture instance."""
    label: str
    signer_id: str
    session_id: str
    dataset_version: str
    frames: list[FrameLandmarks] = field(default_factory=list)
    fps: float = 30.0
    source: str = "webcam"

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "signer_id": self.signer_id,
            "session_id": self.session_id,
            "dataset_version": self.dataset_version,
            "num_frames": self.num_frames,
            "fps": self.fps,
            "source": self.source,
            "frames": [f.to_dict() for f in self.frames],
        }

    def save(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "GestureLandmarkRecord":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        frames = []
        for fd in data["frames"]:
            frames.append(
                FrameLandmarks(
                    frame_index=fd["frame_index"],
                    timestamp_ms=fd["timestamp_ms"],
                    left_hand=HandLandmarks(
                        detected=fd["left_hand"]["detected"],
                        landmarks=fd["left_hand"]["landmarks"],
                        handedness_score=fd["left_hand"].get("handedness_score", 0.0),
                    ),
                    right_hand=HandLandmarks(
                        detected=fd["right_hand"]["detected"],
                        landmarks=fd["right_hand"]["landmarks"],
                        handedness_score=fd["right_hand"].get("handedness_score", 0.0),
                    ),
                )
            )

        return cls(
            label=data["label"],
            signer_id=data["signer_id"],
            session_id=data["session_id"],
            dataset_version=data["dataset_version"],
            frames=frames,
            fps=data.get("fps", 30.0),
            source=data.get("source", "unknown"),
        )


_EMPTY_HAND = HandLandmarks(detected=False, landmarks=[], handedness_score=0.0)


class LandmarkExtractor:
    """
    Stateful MediaPipe Hands wrapper.

    Usage:
        extractor = LandmarkExtractor()
        frame_lm = extractor.extract_frame(bgr_frame, frame_index=0, timestamp_ms=0)
        extractor.close()

    Or use as a context manager:
        with LandmarkExtractor() as extractor:
            ...
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        max_num_hands: int = 2,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def extract_frame(
        self,
        bgr_frame: np.ndarray,
        frame_index: int = 0,
        timestamp_ms: float = 0.0,
    ) -> FrameLandmarks:
        """
        Extract hand landmarks from a single BGR frame (OpenCV format).

        Returns a FrameLandmarks with left and right hand data.
        If a hand is not detected, that hand's .detected = False and landmarks = [].
        """
        import cv2
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

        left_hand = _EMPTY_HAND
        right_hand = _EMPTY_HAND

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_lms, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                label = handedness.classification[0].label  # "Left" or "Right"
                score = handedness.classification[0].score

                landmarks = [
                    {
                        "id": i,
                        "x": round(lm.x, 6),
                        "y": round(lm.y, 6),
                        "z": round(lm.z, 6),
                    }
                    for i, lm in enumerate(hand_lms.landmark)
                ]

                hand = HandLandmarks(
                    detected=True,
                    landmarks=landmarks,
                    handedness_score=round(score, 4),
                )

                # MediaPipe reports from the camera's perspective (mirrored).
                # "Left" in MediaPipe = user's right hand in a selfie/webcam view.
                if label == "Left":
                    right_hand = hand
                else:
                    left_hand = hand

        return FrameLandmarks(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            left_hand=left_hand,
            right_hand=right_hand,
        )

    def close(self) -> None:
        self._hands.close()

    def __enter__(self) -> "LandmarkExtractor":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
