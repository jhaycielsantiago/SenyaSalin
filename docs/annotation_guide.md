# Annotation Guide — Contributing New Signs to SenyaSalin

This guide explains how to add a new FSL sign to the dataset and pipeline.
Adding a sign requires four steps: schema entry, video collection, extraction, and training.

---

## Step 1 — Add the Sign to the Schema

Open `schemas/gestures.json` and add a new entry under `"gestures"`.
Use the existing entries as templates.

```json
"DOCTOR": {
  "sign_id": "FSL_DOCTOR_001",
  "label": "DOCTOR",
  "description": "Dominant hand in 'D' shape, two fingers tapped on inside of opposite wrist",
  "handshape": "D",
  "movement": "double_tap_wrist",
  "location": "wrist",
  "dominant_hand": "right",
  "requires_both_hands": true,
  "intent": "request_doctor",
  "scenario_tags": ["health", "emergency"],
  "validation_status": "proposed",
  "validation_notes": "Needs community validation before marking as team_reviewed.",
  "version": "0.2",
  "added_by": "your_github_handle",
  "added_date": "2025-03-01"
}
```

Then open `schemas/mappings.json` and add the corresponding intent:

```json
"request_doctor": {
  "tagalog": "Kailangan ko po ng doktor.",
  "taglish": "Kailangan ko po ng doktor.",
  "english": "I need a doctor.",
  "tts_lang_code": {
    "tagalog": "tl",
    "taglish": "tl",
    "english": "en"
  }
}
```

Finally update `schemas/benchmark_config.json`:
- Increment `num_classes` by 1
- Add `"DOCTOR"` to the `gestures` array

---

## Step 2 — Collect Video Data

Create a directory for the new sign:

```bash
mkdir -p data/raw/DOCTOR
```

Record **at least 30 short video clips** (2–5 seconds each) of the sign being performed.
Aim for variety:
- Multiple contributors (at least 3 different people)
- Slightly different camera angles, distances, and lighting
- Both left- and right-hand dominant performers if possible

Save clips as `.mp4` or `.avi` files directly in `data/raw/DOCTOR/`.

**Naming convention:** `{contributor_id}_{take_number}.mp4`
Example: `kgarcia_01.mp4`, `kgarcia_02.mp4`, `cabiera_01.mp4`

---

## Step 3 — Extract Landmarks

Run the extraction pipeline on the new sign directory:

```bash
python -m extraction.video_processor \
  --raw-dir data/raw/DOCTOR \
  --output-dir data/processed/DOCTOR
```

This runs MediaPipe Hands on each frame and saves per-frame landmark arrays as `.npy` files.
Check the extraction log — frames where no hand was detected are skipped automatically.

---

## Step 4 — Retrain and Benchmark

Retrain all baseline models with the expanded dataset:

```bash
python -m training.train --export-splits --seed 42
```

Then run the benchmark to verify the new sign improves or does not regress performance:

```bash
python -m benchmark.run_benchmark
```

Compare results against the v0.1 baseline in `benchmark/results/`.
If per-class F1 for the new sign is below 0.75, review the video quality and diversity before merging.

---

## Validation Status Values

| Status | Meaning |
|---|---|
| `proposed` | New entry, not yet reviewed by any team member |
| `team_reviewed` | Reviewed against FSL reference materials by at least one team member |
| `community_validated` | Validated by a Deaf community partner or NFAD-affiliated reviewer |

Always add new signs as `proposed` unless you have conducted a formal review.

---

## Gesture Description Guidelines

The `description` field is the most important for reproducibility.
Write it from the perspective of a neutral observer describing what the hand is doing:

- Include the **handshape** (use ASL/FSL letter notation where applicable: A, B, C, D, F, H, L, N, O, S, T, W, Y, 1, 5, 8, flat_B, flat_O)
- Include the **movement** (direction, repetition, bilateral or unilateral)
- Include the **location** (chin, mouth, forehead, ear, neutral_space, palm, wrist)
- Do not include any interpretation or translation in the description field — that belongs in `mappings.json`

**Good:** `"Dominant hand in 'W' shape, tapped twice on chin"`
**Bad:** `"Water sign — means the person is thirsty"`
