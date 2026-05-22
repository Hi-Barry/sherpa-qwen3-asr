# ==========================================================================
# sherpa-qwen3-asr — Dockerfile (CPU)
#
# Build:
#   docker build -t sherpa-qwen3-asr .
#
# Run (CPU):
#   docker run -p 8000:8000 -v $(pwd)/models:/app/models sherpa-qwen3-asr
# ==========================================================================

FROM python:3.11-slim-bookworm

LABEL org.opencontainers.image.title="sherpa-qwen3-asr — Qwen3-ASR Speech Recognition API"
LABEL org.opencontainers.image.description="Speech recognition powered by Qwen3-ASR 0.6B int8 — CPU-only, ONNX-based"
LABEL org.opencontainers.image.version="0.1.0"

# System dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY config/ config/

# Models are mounted as a volume (not baked into image)
# Run scripts/download_models.sh on the host first

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/api/v1/health').read().decode())" || exit 1

CMD ["python", "-m", "src.api"]
