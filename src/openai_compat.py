"""
OpenAI-compatible API routes.

Provides OpenAI Whisper API compatible endpoints:
    POST /v1/audio/transcriptions
    POST /v1/audio/translations
    GET  /v1/models

This allows existing projects using the OpenAI SDK to
switch to sherpa-qwen3-asr by simply changing the base_url.
"""
import logging
import os
import uuid
from typing import Optional
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Request
from fastapi.responses import PlainTextResponse

from .engine import SpeechEngine, ALLOWED_EXTENSIONS
from .models import (
    OpenAICompletionResponse,
    OpenAIVerboseResponse,
    OpenAIVerboseSegment,
    OpenAIModelList,
    OpenAIModelObject,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1")

# Valid model aliases
MODEL_ALIASES = {
    "Qwen/Qwen3-ASR-0.6B": "qwen3-asr",
    "qwen3-asr": "qwen3-asr",
}

# Valid format options
RESPONSE_FORMATS = frozenset({"json", "text", "verbose_json"})


def _get_engine(request: Request) -> SpeechEngine:
    """Get the SpeechEngine from app state."""
    engine = request.app.state.engine
    if engine is None:
        raise HTTPException(status_code=503, detail="Service initializing")
    return engine


def _get_config(request: Request) -> dict:
    """Get config from app state."""
    return request.app.state.config


# ------------------------------------------------------------------
# POST /v1/audio/transcriptions
# ------------------------------------------------------------------


@router.post("/audio/transcriptions")
async def create_transcription(
    request: Request,
    file: UploadFile = File(..., description="Audio file to transcribe"),
    model: str = Form(..., description="Model name"),
    language: Optional[str] = Form("", description="Language code"),
    response_format: Optional[str] = Form("json", description="Response format"),
    prompt: Optional[str] = Form(None, description="Guidance prompt (ignored)"),
    temperature: Optional[float] = Form(None, description="Temperature (ignored)"),
):
    """
    OpenAI-compatible audio transcription endpoint.

    Compatible with OpenAI Whisper API.
    Supports json, text, and verbose_json response formats.
    """
    engine = _get_engine(request)

    # Validate model
    if model not in MODEL_ALIASES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model '{model}'. Supported: {', '.join(MODEL_ALIASES.keys())}",
        )

    # Validate response_format
    fmt = response_format or "json"
    if fmt not in RESPONSE_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid response_format '{fmt}'. Supported: {', '.join(sorted(RESPONSE_FORMATS))}",
        )

    # Validate file extension
    ext = Path(file.filename or "audio.wav").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # Save uploaded file to temp
    config = _get_config(request)
    temp_dir = config.get("processing", {}).get("temp_dir", "/tmp/sherpa-qwen3-asr")
    os.makedirs(temp_dir, exist_ok=True)

    safe_name = file.filename or "audio.wav"
    temp_path = os.path.join(temp_dir, f"openai_upload_{uuid.uuid4().hex}_{safe_name}")
    try:
        content = await file.read()

        # Check file size
        max_file_size = config.get("processing", {}).get("max_file_size", 50 * 1024 * 1024)
        if len(content) > max_file_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({len(content)} bytes). Max: {max_file_size} bytes",
            )

        with open(temp_path, "wb") as f:
            f.write(content)

        # Process audio
        result = engine.process(temp_path, language=language or "")

        # Format response
        if fmt == "text":
            return PlainTextResponse(content=result.text or "")

        if fmt == "verbose_json":
            segments = [
                OpenAIVerboseSegment(
                    id=i,
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                )
                for i, seg in enumerate(result.segments)
            ]

            return OpenAIVerboseResponse(
                task="transcribe",
                language=result.language,
                duration=result.duration,
                text=result.text or "",
                segments=segments,
            )

        # Default: json format (OpenAI compatible)
        return OpenAICompletionResponse(text=result.text or "")

    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Model not found: {e}")
    except ValueError as e:
        raise HTTPException(status_code=413, detail=str(e))
    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass


# ------------------------------------------------------------------
# POST /v1/audio/translations
# ------------------------------------------------------------------


@router.post("/audio/translations")
async def create_translation(
    request: Request,
    file: UploadFile = File(..., description="Audio file to translate"),
    model: str = Form(..., description="Model name"),
    language: Optional[str] = Form(None, description="Language code"),
    response_format: Optional[str] = Form("json", description="Response format"),
    prompt: Optional[str] = Form(None, description="Guidance prompt (ignored)"),
    temperature: Optional[float] = Form(None, description="Temperature (ignored)"),
):
    """
    OpenAI-compatible audio translation endpoint.

    Note: Qwen3-ASR does not translate to English.
    This endpoint falls back to transcription (same as /audio/transcriptions).
    """
    return await create_transcription(
        request=request,
        file=file,
        model=model,
        language=language,
        response_format=response_format,
        prompt=prompt,
        temperature=temperature,
    )


# ------------------------------------------------------------------
# GET /v1/models
# ------------------------------------------------------------------


@router.get("/models")
async def list_models(request: Request):
    """
    List available models (OpenAI-compatible format).
    """
    return OpenAIModelList(
        data=[
            OpenAIModelObject(
                id="Qwen/Qwen3-ASR-0.6B",
                created=1710000000,
                owned_by="Qwen",
            ),
        ]
    )
