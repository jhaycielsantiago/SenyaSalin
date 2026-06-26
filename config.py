"""
SenyaSalin runtime configuration.

Modify values here to change camera index, model selection,
confidence threshold, or output language. All downstream
modules read from this file rather than hardcoding values.
"""

# ─── Camera ───────────────────────────────────────────────────────────────────

CAMERA_INDEX: int = 0
CAMERA_WIDTH: int = 960
CAMERA_HEIGHT: int = 540

# ─── Model ────────────────────────────────────────────────────────────────────

MODEL_PATH: str = "models/saved/knn.pkl"
GESTURES_PATH: str = "schemas/gestures.json"
MAPPINGS_PATH: str = "schemas/mappings.json"

# ─── Recognition ──────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD: float = 0.70
OUTPUT_LANGUAGE: str = "taglish"   # "tagalog" | "taglish" | "english"

# ─── Audio / TTS ──────────────────────────────────────────────────────────────

TTS_ENABLED: bool = True
PHRASE_COOLDOWN_SECONDS: float = 2.5  # min seconds between repeated TTS outputs

# ─── Display ──────────────────────────────────────────────────────────────────

WINDOW_TITLE: str = "SenyaSalin — FSL Gesture Recognition"
OVERLAY_FONT_SCALE: float = 0.7
OVERLAY_THICKNESS: int = 2
