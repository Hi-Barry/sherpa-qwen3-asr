"""
Pydantic models for sherpa-qwen3-asr request/response.
"""
from typing import Optional, List
from pydantic import BaseModel, Field


class Segment(BaseModel):
    """A single transcribed segment."""

    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Transcribed text")
    language: Optional[str] = Field(
        None, description="Detected or forced language code"
    )


class ProcessingStats(BaseModel):
    """Timing statistics for a request."""

    asr_time: float = Field(..., description="ASR processing time (seconds)")
    total_time: float = Field(..., description="Total processing time (seconds)")


class RecognitionResult(BaseModel):
    """Complete recognition result for an audio file."""

    language: str = Field(
        default="unknown", description="Detected language or forced language"
    )
    duration: float = Field(..., description="Audio duration in seconds")
    segments: List[Segment] = Field(
        default_factory=list, description="Transcribed segments"
    )
    stats: Optional[ProcessingStats] = Field(None, description="Timing statistics")
    text: str = Field(
        default="", description="Full concatenated transcription text"
    )


class RecognitionResponse(BaseModel):
    """Top-level API response."""

    success: bool = Field(..., description="Whether the request succeeded")
    result: Optional[RecognitionResult] = Field(
        None, description="Recognition result"
    )
    error: Optional[str] = Field(None, description="Error message if failed")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    models: dict = Field(..., description="Model loading status")
    memory_mb: Optional[float] = Field(None, description="Approximate memory usage")
    provider: str = Field("cpu", description="Current inference provider")


# ------------------------------------------------------------------
# OpenAI-compatible response models
# ------------------------------------------------------------------


class OpenAICompletionResponse(BaseModel):
    """Response for response_format=json (OpenAI compatible)."""

    text: str = Field(..., description="The transcribed text")


class OpenAIVerboseSegment(BaseModel):
    """A segment in verbose_json response."""

    id: int = Field(..., description="Segment index")
    seek: int = Field(0, description="Seek offset")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Transcribed text")
    tokens: list = Field(default_factory=list, description="Token IDs (unused)")
    temperature: float = Field(0.0, description="Temperature")
    avg_logprob: float = Field(0.0, description="Average log probability")
    compression_ratio: float = Field(0.0, description="Compression ratio")
    no_speech_prob: float = Field(0.0, description="No speech probability")


class OpenAIVerboseResponse(BaseModel):
    """Response for response_format=verbose_json."""

    task: str = Field("transcribe", description="Task type")
    language: str = Field(..., description="Detected language")
    duration: float = Field(..., description="Audio duration in seconds")
    text: str = Field(..., description="Full transcribed text")
    segments: List[OpenAIVerboseSegment] = Field(
        default_factory=list, description="Transcribed segments"
    )


class OpenAIModelObject(BaseModel):
    """A model entry in the model list."""

    id: str = Field(..., description="Model identifier")
    object: str = Field("model", description="Object type")
    created: int = Field(..., description="Unix timestamp")
    owned_by: str = Field(..., description="Organization/owner")


class OpenAIModelList(BaseModel):
    """Response for GET /v1/models."""

    object: str = Field("list", description="Object type")
    data: List[OpenAIModelObject] = Field(
        default_factory=list, description="Model list"
    )
