"""
Pydantic schemas for the SenyaSalin API.

Defines request/response models for all API endpoints.
Keeping schemas in a separate file allows the frontend team
to generate TypeScript types automatically via openapi-ts.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class RecognizeFrameResponse(BaseModel):
    """Response from POST /recognize"""
    label: Optional[str] = Field(None, description="Recognized gesture label, or null if below threshold")
    intent: Optional[str] = Field(None, description="Accessibility intent, e.g. 'request_assistance'")
    phrase: Optional[str] = Field(None, description="Natural language phrase for TTS output")
    confidence: float = Field(ge=0.0, le=1.0, description="Classifier confidence [0.0, 1.0]")
    tts_lang_code: str = Field("tl", description="BCP-47 language code for TTS")
    below_threshold: bool = Field(False, description="True if confidence < min_confidence")


class GestureInfo(BaseModel):
    label: str
    intent: str
    description: str
    scenario_tags: list[str]


class GestureListResponse(BaseModel):
    """Response from GET /gestures"""
    total: int
    gestures: list[GestureInfo]


class BenchmarkResultSummary(BaseModel):
    model_name: str
    accuracy: float
    f1_weighted: float
    inference_time_ms_mean: float


class BenchmarkResultsResponse(BaseModel):
    """Response from GET /benchmark"""
    results: list[BenchmarkResultSummary]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str]
    num_classes: int
    version: str = "0.1"
