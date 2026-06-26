# SenyaSalin Open Gesture Schema — Specification v0.1

The gesture schema (`schemas/gestures.json`) is the authoritative source of truth
for all gesture labels, their descriptions, and their intent mappings.
It is versioned independently of the pipeline and released under **CC BY 4.0**.

---

## Schema Files

| File | Purpose |
|---|---|
| `schemas/gestures.json` | Per-sign descriptive metadata and intent keys |
| `schemas/mappings.json` | Intent → natural language phrase mapping (multi-language) |
| `schemas/benchmark_config.json` | Benchmark parameters tied to a dataset version |
| `schemas/landmark_schema.json` | Structure of `.npy` landmark files produced by extraction |

---

## Gesture Entry Fields (`gestures.json`)

```json
{
  "sign_id": "FSL_HELP_001",
  "label": "HELP",
  "description": "Dominant hand in 'A' shape placed on top of non-dominant flat hand, lifted upward",
  "handshape": "A",
  "movement": "upward_lift",
  "location": "neutral_space",
  "dominant_hand": "right",
  "requires_both_hands": true,
  "intent": "request_assistance",
  "scenario_tags": ["emergency", "health", "public_service"],
  "validation_status": "team_reviewed",
  "validation_notes": "Based on standard FSL reference. Awaiting Deaf community validation.",
  "version": "0.1",
  "added_by": "team_explicit",
  "added_date": "2025-01-01"
}
```

### Field Reference

| Field | Type | Required | Description |
|---|---|---|---|
| `sign_id` | string | yes | Globally unique ID. Format: `FSL_{LABEL}_{NNN}`. Never reuse or recycle IDs. |
| `label` | string | yes | Short uppercase identifier used as the classifier class name. Must match the key in the `gestures` object. |
| `description` | string | yes | Human-readable, observer-perspective description of the gesture. See [annotation guide](annotation_guide.md). |
| `handshape` | string | yes | Handshape code: A, B, C, D, F, H, L, N, O, S, T, W, Y, 1, 5, 8, flat_B, flat_O |
| `movement` | string | yes | Movement descriptor: e.g. `upward_lift`, `double_tap_chin`, `wrist_shake`, `circular_on_palm` |
| `location` | string | yes | Sign location: `neutral_space`, `chin`, `mouth`, `forehead`, `ear`, `palm`, `wrist` |
| `dominant_hand` | string | yes | `"right"`, `"left"`, or `"both"` |
| `requires_both_hands` | boolean | yes | Whether the sign requires both hands simultaneously |
| `intent` | string | yes | Key linking to `mappings.json`. Must have a matching entry. |
| `scenario_tags` | string[] | yes | Usage contexts: `emergency`, `health`, `daily_need`, `safety`, `communication`, `social`, `public_service` |
| `validation_status` | string | yes | `proposed`, `team_reviewed`, or `community_validated` |
| `validation_notes` | string | no | Free-text notes on validation source or caveats |
| `version` | string | yes | Schema version string when this entry was added |
| `added_by` | string | yes | Contributor handle or `team_explicit` |
| `added_date` | string | yes | ISO 8601 date (`YYYY-MM-DD`) |

---

## Intent Mapping Fields (`mappings.json`)

```json
"request_assistance": {
  "tagalog": "Kailangan ko po ng tulong.",
  "taglish": "Kailangan ko po ng tulong.",
  "english": "I need help.",
  "tts_lang_code": {
    "tagalog": "tl",
    "taglish": "tl",
    "english": "en"
  }
}
```

**Adding a new language:** Add a new key alongside `tagalog`/`taglish`/`english` under every intent.
No code changes are needed — the pipeline reads output language from config.

---

## Landmark File Format (`landmark_schema.json`)

Each `.npy` file in `data/processed/` is a NumPy array with shape `(N_frames, 126)`.

The 126 features are:
- Left hand: 21 landmarks × 3 coordinates (x, y, z) = 63 values
- Right hand: 21 landmarks × 3 coordinates (x, y, z) = 63 values
- Total: 126 values per frame

Coordinates are **wrist-relative and unit-scale normalized** (see `preprocessing/normalizer.py`).
If a hand is not detected in a frame, its 63 values are set to zero.

---

## Versioning Policy

- The schema `version` field increments on any **breaking change** (new required field, changed field semantics, removed intent).
- Adding new gesture entries does **not** increment the schema version.
- Adding a new language to `mappings.json` does **not** increment the schema version.
- Whenever `num_classes` changes in `benchmark_config.json`, retrain and re-run the benchmark before merging.

---

## License

All schema files (`schemas/`) are released under **CC BY 4.0**.
You may use, share, and adapt them for any purpose — research, commercial, or community — with attribution:

> *SenyaSalin Open Gesture Schema v0.1, Team EXPLICIT, FEU ACM TechSprint 2025*
