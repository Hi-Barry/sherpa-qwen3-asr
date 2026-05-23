"""
sherpa-qwen3-asr Server.

FastAPI service providing:
    POST /api/v1/recognize  — Upload audio, get transcription
    GET  /api/v1/health      — Health check
    GET  /                    — API documentation (HTML)

Usage:
    python -m src.api
    # or:
    uvicorn src.api:app --host 0.0.0.0 --port 8000
"""
import os
import sys
import uuid
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from .engine import SpeechEngine, ALLOWED_EXTENSIONS
from .models import (
    RecognitionResponse,
    RecognitionResult,
    HealthResponse,
)
from .openai_compat import router as openai_router

# ---- Logging ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("sherpa-qwen3-asr")

# ---- Config ----
CONFIG_PATH = os.environ.get(
    "SHERPA_QWEN3_CONFIG",
    str(Path(__file__).parent.parent / "config" / "config.yaml"),
)

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

# ---- Global engine (singleton) ----
engine: SpeechEngine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load model. Shutdown: cleanup."""
    global engine
    logger.info("Starting sherpa-qwen3-asr server...")
    logger.info(f"Config: {CONFIG_PATH}")

    engine = SpeechEngine(config)
    app.state.engine = engine
    app.state.config = config

    logger.info("Server ready")

    yield

    logger.info("Shutting down...")


app = FastAPI(
    title="sherpa-qwen3-asr",
    description=(
        "Qwen3-ASR speech recognition API. "
        "Powered by Qwen3-ASR 0.6B int8 (ONNX) + sherpa-onnx. "
        "Supports 52 languages, pure CPU or GPU inference."
    ),
    version="0.1.2",
    lifespan=lifespan,
)

# Register OpenAI-compatible routes
app.include_router(openai_router)

# ---- Routes ----


@app.get("/", response_class=HTMLResponse)
async def root():
    """API documentation page."""
    return f"""<!DOCTYPE html>
<html>
<head>
    <title>sherpa-qwen3-asr</title>
    <meta charset="utf-8">
    <style>
        body {{ font-family: -apple-system, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #555; margin-top: 30px; }}
        code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px; }}
        pre {{ background: #f5f5f5; padding: 15px; border-radius: 6px; overflow-x: auto; }}
        .endpoint {{ background: #e8f4e8; padding: 10px 15px; border-radius: 6px; margin: 10px 0; }}
        .method {{ font-weight: bold; color: #2a7d2a; }}
    </style>
</head>
<body>
    <h1>sherpa-qwen3-asr</h1>
    <p>Qwen3-ASR 0.6B int8 speech recognition API.<br>
    Powered by <strong>Qwen3-ASR</strong> (ONNX int8) + <strong>sherpa-onnx</strong>.<br>
    Pure CPU or CUDA GPU inference, 52 languages supported.</p>

    <h2>Endpoints</h2>

    <div class="endpoint">
        <span class="method">POST</span> <code>/api/v1/recognize</code>
        <p>Upload an audio file and get transcription.</p>
        <pre>curl -X POST http://localhost:8000/api/v1/recognize \\
  -F "file=@audio.wav"</pre>
    </div>

    <div class="endpoint">
        <span class="method">POST</span> <code>/v1/audio/transcriptions</code>
        <p>OpenAI-compatible transcription endpoint.</p>
        <pre>curl -X POST http://localhost:8000/v1/audio/transcriptions \\
  -F "file=@audio.wav" \\
  -F "model=Qwen/Qwen3-ASR-0.6B"</pre>
    </div>

    <div class="endpoint">
        <span class="method">GET</span> <code>/api/v1/health</code>
        <p>Check service health and model status.</p>
    </div>

    <h2>Parameters</h2>
    <table>
        <tr><th>Param</th><th>Type</th><th>Default</th><th>Description</th></tr>
        <tr><td><code>file</code></td><td>file</td><td>required</td><td>Audio file (wav, mp3, flac, ogg, m4a)</td></tr>
        <tr><td><code>language</code></td><td>string</td><td>"" (auto)</td><td>Force language: "Chinese", "English", "Korean", etc.</td></tr>
    </table>

    <h2>Supported Formats</h2>
    <p>WAV, MP3, FLAC, OGG, M4A, AAC, OPUS, WEBM — auto-resampled to 16kHz mono.</p>

    <p style="margin-top: 30px; color: #888;">
        Docs: <a href="/docs">Swagger UI</a> | <a href="/redoc">ReDoc</a>
    </p>
</body>
</html>"""


@app.get("/api/v1/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    if engine is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "initializing",
                "models": {"asr": False},
                "provider": config.get("asr", {}).get("provider", "cpu"),
            },
        )

    try:
        import psutil

        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
    except ImportError:
        memory_mb = None

    return HealthResponse(
        status="ok",
        models={"asr": engine.is_ready},
        memory_mb=memory_mb,
        provider=engine.provider,
    )


@app.post("/api/v1/recognize", response_model=RecognitionResponse)
async def recognize(
    file: UploadFile = File(..., description="Audio file (wav, mp3, flac, ogg, m4a)"),
    language: str = Form(
        "", description="Force language: 'Chinese', 'English', 'Korean', etc. Empty=auto"
    ),
):
    """
    Transcribe audio using Qwen3-ASR.

    - **file**: Audio file to transcribe
    - **language**: Force transcription language, or empty string for auto-detect
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Service initializing, please wait")

    # Validate file extension
    ext = Path(file.filename or "audio.wav").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # Save uploaded file to temp
    temp_dir = config.get("processing", {}).get("temp_dir", "/tmp/sherpa-qwen3-asr")
    os.makedirs(temp_dir, exist_ok=True)

    safe_name = file.filename or "audio.wav"
    temp_path = os.path.join(temp_dir, f"upload_{uuid.uuid4().hex}_{safe_name}")
    try:
        content = await file.read()
        logger.info(f"Received file: {file.filename} ({len(content)} bytes)")

        # Check file size
        max_file_size = config.get("processing", {}).get("max_file_size", 50 * 1024 * 1024)
        if len(content) > max_file_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({len(content)} bytes). Max: {max_file_size} bytes",
            )

        with open(temp_path, "wb") as f:
            f.write(content)

        # Process
        result: RecognitionResult = engine.process(temp_path, language=language or "")

        return RecognitionResponse(success=True, result=result)

    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Model not found. Run scripts/download_models.sh first. Details: {e}",
        )
    except ValueError as e:
        raise HTTPException(status_code=413, detail=str(e))
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


# ---- Main ----

if __name__ == "__main__":
    import uvicorn

    svc = config.get("service", {})
    uvicorn.run(
        "src.api:app",
        host=svc.get("host", "0.0.0.0"),
        port=svc.get("port", 8000),
        log_level=config.get("logging", {}).get("level", "info").lower(),
    )
