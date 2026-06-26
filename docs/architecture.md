# SenyaSalin — Technical Architecture

## Overview

SenyaSalin is structured as a reproducible ML pipeline with a thin application layer on top.
The separation of concerns is intentional: the recognition pipeline (`senyasalin/`) is independent
of the web API (`app/`) and can be used standalone or embedded in other applications.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SenyaSalin Architecture                       │
├──────────────┬────────────────────┬────────────────┬────────────────┤
│  DATA LAYER  │   PIPELINE LAYER   │   APP LAYER    │  RESEARCH LAYER│
│              │                    │                │                │
│  data/raw/   │  extraction/       │  app/backend/  │  training/     │
│  data/       │  preprocessing/    │  (FastAPI)     │  benchmark/    │
│  processed/  │  senyasalin/       │                │  notebooks/    │
│  schemas/    │  (inference)       │  app/frontend/ │                │
│  models/     │                    │  (React+Vite)  │                │
└──────────────┴────────────────────┴────────────────┴────────────────┘
```

---

## Module Responsibilities

### `extraction/`
Converts raw video files into per-frame landmark arrays.

- `landmark_extractor.py` — wraps MediaPipe Hands; returns `FrameLandmarks` objects with left/right hand detection
- `video_processor.py` — iterates over video files in a directory, calls the extractor per frame, saves `.npy` output

MediaPipe detects up to 21 landmarks per hand (x, y, z in normalized image space).
The extractor handles missing hands (returns zero arrays) and runs in a single-pass, frame-by-frame loop.

### `preprocessing/`
Transforms raw landmarks into classifier-ready feature vectors.

- `normalizer.py` — applies wrist-relative translation and unit-scale normalization (removes position and scale dependence between signers)
- `feature_extractor.py` — aggregates per-frame vectors (currently `mean`; `std` and `concat` aggregations available)
- `data_loader.py` — loads processed `.npy` files, applies stratified train/val/test splitting (seed=42)

### `senyasalin/` (core inference package)
The production-facing recognition pipeline. This is the only package the application imports.

- `pipeline.py` — `GestureRecognizer`: accepts a BGR frame, runs extraction + normalization + classification, returns a `RecognitionResult`
- `mapper.py` — `IntentMapper`: maps gesture label → intent → natural language phrase using the JSON schemas

`GestureRecognizer` is stateless across frames. Temporal smoothing (voting over N consecutive frames) is deliberately left to the caller, since appropriate smoothing strategy depends on the use case.

### `training/`
Trains and saves baseline classifiers.

- `base_trainer.py` — abstract `BaseTrainer` with `build_model()`, `train()`, `save()`, `load()`
- `knn_trainer.py` — `KNeighborsClassifier` (k=5, distance-weighted, euclidean)
- `rf_trainer.py` — `RandomForestClassifier` (n=200, sqrt features)
- `mlp_trainer.py` — `MLPClassifier` (256→128, ReLU, dropout, early stopping)
- `train.py` — CLI entry point

### `benchmark/`
Evaluates trained models against the held-out test split.

- `evaluator.py` — runs inference on test set, records inference times, builds per-class metric dicts
- `metrics.py` — formats, prints, and saves results to `benchmark/results/`
- `run_benchmark.py` — CLI entry point

### `app/backend/`
FastAPI wrapper around the `senyasalin` inference package.

Endpoints: `GET /health`, `POST /recognize`, `GET /gestures`, `GET /benchmark`

The `/recognize` endpoint accepts base64-encoded JPEG frames from the browser webcam,
decodes them, and passes them through `GestureRecognizer`. The backend is a stateless process — it loads the model once at startup.

### `app/frontend/`
React + TypeScript + Vite demo application.

Uses the browser `getUserMedia` API to capture webcam frames, sends them to the backend `/recognize` endpoint at a configurable frame rate (default: 5 fps), and displays the recognized gesture with the Taglish phrase and confidence bar.

---

## Inference Pipeline (per frame)

```
BGR frame (from webcam or file)
    │
    ▼
LandmarkExtractor.extract_frame()
    │  MediaPipe Hands — detects left + right hand
    │  Returns FrameLandmarks: {left: HandLandmarks, right: HandLandmarks}
    │
    ▼
LandmarkNormalizer.normalize_frame()
    │  1. Translate all points relative to wrist (landmark 0)
    │  2. Scale to unit magnitude (largest distance from wrist = 1.0)
    │  3. Concatenate left (63) + right (63) → 126-dim vector
    │     (zero-fill if hand not detected)
    │
    ▼
Classifier.predict() + predict_proba()
    │  KNN / RF / MLP → class index + confidence
    │
    ▼
IntentMapper.get_phrase()
    │  class index → label → intent → Taglish phrase
    │
    ▼
RecognitionResult
    { label, intent, phrase, confidence, tts_lang_code, below_threshold }
```

---

## Design Decisions

**Why KNN as the default model?**
KNN requires no training time and is trivially interpretable — the nearest neighbors
are real training samples you can inspect. For a 15-class problem with ~4,700 training
samples, KNN at 87.3% accuracy is a reasonable interactive baseline. It can be swapped
for Random Forest (91.8%) via `MODEL_PATH` in `config.py` at the cost of higher
inference latency (~8ms vs ~2ms).

**Why frame-independent classification?**
Temporal aggregation strategies (voting, HMM, sliding window) are highly scenario-dependent.
A healthcare kiosk benefits from stability; a fast-paced emergency scenario needs lower latency.
By keeping the core pipeline frame-independent, we leave this decision to the integrating application.

**Why JSON schemas for phrases?**
Adding a new gesture or a new language requires only a JSON edit — no retraining, no code changes.
This is intentional for community extensibility: NGOs and language researchers can contribute
phrases without touching the Python codebase.

**Why wrist-relative normalization?**
FSL gestures are defined by handshape and relative movement, not by absolute screen position.
Normalizing relative to the wrist removes signer position and camera distance as variables,
making the classifier robust to different physical setups and signer heights.

---

## Adding a New Model Architecture

1. Create a new file in `training/` (e.g. `training/lstm_trainer.py`)
2. Subclass `BaseTrainer` and implement `build_model()` returning an sklearn-compatible estimator
3. Register it in `training/__init__.py` and in `training/train.py` `TRAINERS` dict
4. Run `python -m training.train --model your_model` and `python -m benchmark.run_benchmark`

The benchmark evaluator and API will treat it identically to the baseline models.
