"""
SenyaSalin FastAPI backend.

Exposes the gesture recognition pipeline as an HTTP API
for the React frontend demo.

Endpoints:
    GET  /health        — health check and model status
    POST /recognize     — classify a single webcam frame (base64 JPEG)
    GET  /gestures      — list supported gestures from the schema
    GET  /benchmark     — return latest benchmark results

The app serves as a thin wrapper around the senyasalin core package.
All business logic lives in senyasalin/, not here.
"""

from __future__ import annotations

import base64
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware

from senyasalin import GestureRecognizer
from app.backend.schemas import (
    RecognizeFrameResponse,
    GestureListResponse,
    GestureInfo,
    BenchmarkResultsResponse,
    BenchmarkResultSummary,
    HealthResponse,
)

# ─── Global state ────────────────────────────────────────────────────────────

recognizer: Optional[GestureRecognizer] = None
_gestures_schema: dict = {}
_benchmark_results: list[dict] = []

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/saved/knn.pkl"))
SCHEMAS_DIR = Path(os.getenv("SCHEMAS_DIR", "schemas"))
BENCHMARK_RESULTS_DIR = Path(os.getenv("BENCHMARK_RESULTS_DIR", "benchmark/results"))
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.70"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and schemas on startup, clean up on shutdown."""
    global recognizer, _gestures_schema, _benchmark_results

    # Load gesture schema for /gestures endpoint
    gestures_path = SCHEMAS_DIR / "gestures.json"
    if gestures_path.exists():
        with open(gestures_path, encoding="utf-8") as f:
            _gestures_schema = json.load(f).get("gestures", {})

    # Load benchmark results if available
    if BENCHMARK_RESULTS_DIR.exists():
        for result_file in sorted(BENCHMARK_RESULTS_DIR.glob("*.json")):
            try:
                with open(result_file) as f:
                    _benchmark_results.append(json.load(f))
            except Exception:
                pass

    # Load recognizer (fails gracefully — health endpoint reports model_loaded=False)
    if MODEL_PATH.exists():
        try:
            recognizer = GestureRecognizer.load(
                model_path=MODEL_PATH,
                gestures_path=SCHEMAS_DIR / "gestures.json",
                mappings_path=SCHEMAS_DIR / "mappings.json",
                min_confidence=MIN_CONFIDENCE,
            )
            print(f"Model loaded: {MODEL_PATH}")
        except Exception as e:
            print(f"[WARN] Could not load model: {e}")
    else:
        print(f"[WARN] No model found at {MODEL_PATH}. Train first with: python -m training.train")

    yield

    if recognizer:
        recognizer.close()


app = FastAPI(
    title="SenyaSalin API",
    description="Open multimodal FSL gesture recognition benchmark — demo API",
    version="0.1",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Endpoints ───────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """API health check. Reports whether a model is loaded and ready."""
    return HealthResponse(
        status="ok",
        model_loaded=recognizer is not None,
        model_name=MODEL_PATH.stem if recognizer else None,
        num_classes=len(recognizer._class_names) if recognizer else 0,
    )


@app.post("/recognize", response_model=RecognizeFrameResponse)
def recognize_frame(
    frame_b64: str = Body(..., embed=True, description="Base64-encoded JPEG/PNG frame")
) -> RecognizeFrameResponse:
    """
    Classify a gesture from a base64-encoded webcam frame.

    The frontend should capture a frame from the webcam canvas,
    encode it as base64 JPEG, and POST it here.

    Returns the gesture label, intent, phrase, and confidence.
    If confidence < min_confidence, phrase will be null and below_threshold=true.
    """
    if recognizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train and save a model first.",
        )

    try:
        img_bytes = base64.b64decode(frame_b64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        bgr_frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid frame data: {e}")

    if bgr_frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    result = recognizer.recognize(bgr_frame)

    return RecognizeFrameResponse(
        label=result.label,
        intent=result.intent,
        phrase=result.phrase,
        confidence=result.confidence,
        tts_lang_code=result.tts_lang_code,
        below_threshold=result.below_threshold,
    )


@app.get("/gestures", response_model=GestureListResponse)
def list_gestures() -> GestureListResponse:
    """
    List all supported gestures from the open schema.

    This endpoint is driven by schemas/gestures.json — no model required.
    """
    gestures = [
        GestureInfo(
            label=label,
            intent=data.get("intent", ""),
            description=data.get("description", ""),
            scenario_tags=data.get("scenario_tags", []),
        )
        for label, data in _gestures_schema.items()
    ]
    return GestureListResponse(total=len(gestures), gestures=gestures)


@app.get("/benchmark", response_model=BenchmarkResultsResponse)
def get_benchmark_results() -> BenchmarkResultsResponse:
    """
    Return the latest benchmark results for all evaluated models.

    Results are loaded from benchmark/results/*.json at startup.
    Run `python -m benchmark.run_benchmark` to generate/update them.
    """
    summaries = [
        BenchmarkResultSummary(
            model_name=r.get("model_name", "unknown"),
            accuracy=r.get("metrics", {}).get("accuracy", 0.0),
            f1_weighted=r.get("metrics", {}).get("f1_weighted", 0.0),
            inference_time_ms_mean=r.get("metrics", {}).get("inference_time_ms_mean", 0.0),
        )
        for r in _benchmark_results
    ]
    return BenchmarkResultsResponse(results=summaries)
