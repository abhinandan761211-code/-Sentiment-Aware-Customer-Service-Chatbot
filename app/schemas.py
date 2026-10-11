"""Pydantic request/response models shared by the API layer."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

SentimentLabel = Literal["POSITIVE", "NEGATIVE", "NEUTRAL"]


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = Field(
        default=None,
        description="Reuse this id to keep conversation memory across turns.",
    )
    generate_reply: bool = Field(
        default=True,
        description="Set false to only get the sentiment detection result.",
    )


class SentimentResult(BaseModel):
    label: SentimentLabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    scores: Dict[str, float]
    backend: str = Field(..., description="'transformer' or 'lexicon'")
    signals: List[str] = Field(
        default_factory=list,
        description="Words/patterns that drove the decision (explainability).",
    )
    intensity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="How strongly the emotion is expressed (caps, '!', escalation words).",
    )


class ChatResponse(BaseModel):
    reply: str
    sentiment: SentimentResult
    session_id: str
    model: str
    latency_ms: Optional[int] = None


class SentimentOnlyRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)


class ResetRequest(BaseModel):
    session_id: str


class HealthResponse(BaseModel):
    status: str
    ollama: str
    sentiment_backend: str
    model: str
