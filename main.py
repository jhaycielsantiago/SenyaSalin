"""
SenyaSalin — webcam demo entry point.

Runs real-time FSL gesture recognition using a standard webcam.
Press Q to quit. Requires a trained model in models/saved/.

If no model is found, run:
    python -m extraction.video_processor --raw-dir data/raw --output-dir data/processed
    python -m training.train --export-splits
Then re-launch this script.
"""

from __future__ import annotations

import sys
import time
import tempfile
import os
from pathlib import Path

import cv2
import numpy as np

import config
from senyasalin import GestureRecognizer
from senyasalin.pipeline import RecognitionResult


def _draw_overlay(frame: np.ndarray, result: RecognitionResult) -> None:
    h, w = frame.shape[:2]

    if result.label is None:
        cv2.putText(
            frame, "No hand detected",
            (20, 40), cv2.FONT_HERSHEY_DUPLEX,
            config.OVERLAY_FONT_SCALE, (180, 180, 180), config.OVERLAY_THICKNESS,
        )
        return

    conf_pct = int(result.confidence * 100)
    color = (0, 220, 80) if not result.below_threshold else (0, 165, 255)

    cv2.putText(
        frame, f"{result.label}  {conf_pct}%",
        (20, 40), cv2.FONT_HERSHEY_DUPLEX,
        config.OVERLAY_FONT_SCALE, color, config.OVERLAY_THICKNESS,
    )

    if result.phrase:
        cv2.putText(
            frame, result.phrase,
            (20, 75), cv2.FONT_HERSHEY_SIMPLEX,
            config.OVERLAY_FONT_SCALE * 0.85, (255, 255, 255), config.OVERLAY_THICKNESS,
        )
    elif result.below_threshold:
        cv2.putText(
            frame, f"Low confidence — hold sign steady ({conf_pct}% < {int(config.CONFIDENCE_THRESHOLD * 100)}%)",
            (20, 75), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (0, 165, 255), 1,
        )

    cv2.putText(
        frame, "Press Q to quit",
        (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
        0.45, (160, 160, 160), 1,
    )


def _speak(phrase: str, lang_code: str) -> None:
    try:
        from gtts import gTTS
        import pygame

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        gTTS(text=phrase, lang=lang_code).save(tmp_path)

        pygame.mixer.init()
        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.05)

        os.unlink(tmp_path)
    except Exception as e:
        print(f"[TTS] Could not speak phrase: {e}")


def main() -> None:
    model_path = Path(config.MODEL_PATH)
    if not model_path.exists():
        print(f"[ERROR] No model found at {model_path}.")
        print("  Train first:  python -m training.train")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    recognizer = GestureRecognizer.load(
        model_path=model_path,
        gestures_path=config.GESTURES_PATH,
        mappings_path=config.MAPPINGS_PATH,
        language=config.OUTPUT_LANGUAGE,
        min_confidence=config.CONFIDENCE_THRESHOLD,
    )

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {config.CAMERA_INDEX}.")
        print("  Try: python main.py  (then edit config.py CAMERA_INDEX)")
        sys.exit(1)

    last_phrase: str | None = None
    last_spoken_at: float = 0.0

    print(f"Camera open. Performing {config.OUTPUT_LANGUAGE.upper()} recognition. Press Q to quit.\n")

    with recognizer:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame capture failed — camera disconnected?")
                break

            result = recognizer.recognize(frame)
            _draw_overlay(frame, result)

            now = time.monotonic()
            if (
                config.TTS_ENABLED
                and result.phrase is not None
                and not result.below_threshold
                and (
                    result.phrase != last_phrase
                    or now - last_spoken_at > config.PHRASE_COOLDOWN_SECONDS
                )
            ):
                last_phrase = result.phrase
                last_spoken_at = now
                _speak(result.phrase, result.tts_lang_code)

            cv2.imshow(config.WINDOW_TITLE, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
