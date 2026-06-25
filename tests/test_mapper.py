"""
Tests for IntentMapper.

Validates that the schema files are well-formed and that all
gesture labels map to valid intents and phrases.
"""

import pytest
from senyasalin.mapper import IntentMapper


@pytest.fixture
def mapper():
    return IntentMapper(
        gestures_path="schemas/gestures.json",
        mappings_path="schemas/mappings.json",
        language="taglish",
    )


class TestIntentMapper:
    def test_help_intent(self, mapper):
        assert mapper.get_intent("HELP") == "request_assistance"

    def test_help_phrase_taglish(self, mapper):
        phrase = mapper.get_phrase("HELP", language="taglish")
        assert phrase is not None
        assert len(phrase) > 0

    def test_help_phrase_english(self, mapper):
        phrase = mapper.get_phrase("HELP", language="english")
        assert phrase == "I need help."

    def test_unknown_label_returns_none(self, mapper):
        assert mapper.get_intent("NONEXISTENT") is None
        assert mapper.get_phrase("NONEXISTENT") is None

    def test_all_labels_have_intents(self, mapper):
        for label in mapper.list_labels():
            intent = mapper.get_intent(label)
            assert intent is not None, f"Missing intent for {label}"

    def test_all_labels_have_taglish_phrases(self, mapper):
        for label in mapper.list_labels():
            phrase = mapper.get_phrase(label, language="taglish")
            assert phrase is not None and len(phrase) > 0, f"Missing taglish phrase for {label}"

    def test_all_labels_have_english_phrases(self, mapper):
        for label in mapper.list_labels():
            phrase = mapper.get_phrase(label, language="english")
            assert phrase is not None and len(phrase) > 0, f"Missing english phrase for {label}"

    def test_tts_lang_code_taglish(self, mapper):
        code = mapper.get_tts_lang_code("HELP", language="taglish")
        assert code == "tl"

    def test_list_labels_not_empty(self, mapper):
        labels = mapper.list_labels()
        assert len(labels) >= 15
