"""
Intent mapper.

Connects a recognized gesture label → intent → natural language phrase.
Driven entirely by schemas/gestures.json and schemas/mappings.json.

This design ensures that phrase outputs can be updated, corrected, or
extended without retraining any model — just edit the JSON files.

Future contributors: to support a new language (e.g. Cebuano, Ilocano),
add a new key to each intent entry in schemas/mappings.json and rebuild
— no code changes needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


class IntentMapper:
    """
    Maps gesture labels to intents and natural language phrases.

    Args:
        gestures_path: Path to schemas/gestures.json
        mappings_path: Path to schemas/mappings.json
        language: Default output language ("tagalog", "taglish", "english")
    """

    def __init__(
        self,
        gestures_path: str | Path = "schemas/gestures.json",
        mappings_path: str | Path = "schemas/mappings.json",
        language: str = "taglish",
    ) -> None:
        self.language = language
        self._gestures = self._load_gestures(Path(gestures_path))
        self._mappings = self._load_mappings(Path(mappings_path))

    def get_intent(self, label: str) -> Optional[str]:
        """Returns the intent string for a gesture label, or None if not found."""
        gesture = self._gestures.get(label)
        if gesture is None:
            return None
        return gesture.get("intent")

    def get_phrase(self, label: str, language: Optional[str] = None) -> Optional[str]:
        """
        Returns the natural language phrase for a gesture label.

        Args:
            label: Gesture class label (e.g. "HELP")
            language: Override default language.
        """
        intent = self.get_intent(label)
        if intent is None:
            return None

        lang = language or self.language
        intent_data = self._mappings.get(intent)
        if intent_data is None:
            return None

        return intent_data.get(lang)

    def get_tts_lang_code(self, label: str, language: Optional[str] = None) -> str:
        """Returns the BCP-47 language code for TTS synthesis."""
        intent = self.get_intent(label)
        if intent is None:
            return "tl"

        lang = language or self.language
        intent_data = self._mappings.get(intent, {})
        return intent_data.get("tts_lang_code", {}).get(lang, "tl")

    def list_labels(self) -> list[str]:
        return list(self._gestures.keys())

    @staticmethod
    def _load_gestures(path: Path) -> dict:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("gestures", {})

    @staticmethod
    def _load_mappings(path: Path) -> dict:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("intents", {})
