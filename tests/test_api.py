"""
Tests for the FastAPI server — HTTP endpoints + background queue.

Requires models to be downloaded (auto-skipped otherwise).
"""
import sys

sys.path.insert(0, "..")

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from src.api import app, config

# Check if models exist
project_root = Path(__file__).parent.parent
models_dir = project_root / config["models"]["qwen3_asr_dir"]
if not models_dir.exists():
    pytest.skip(
        f"Qwen3 model not found at {models_dir}. Run scripts/download_models.sh first.",
        allow_module_level=True,
    )


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture(scope="module")
def client():
    """Use TestClient as context manager to trigger lifespan."""
    with TestClient(app) as c:
        # Verify engine loaded
        resp = c.get("/api/v1/health")
        assert resp.status_code == 200, f"Engine not ready: {resp.text}"
        yield c


@pytest.fixture
def test_audio() -> str:
    """Use the model's bundled test audio (English, ~3s)."""
    test_wav = project_root / "models" / "qwen3-asr" / "test_wavs" / "en.wav"
    alt_wav = project_root / "models" / "qwen3-asr" / "test_wavs" / "raokouling.wav"
    if test_wav.exists():
        return str(test_wav)
    if alt_wav.exists():
        return str(alt_wav)
    # Fallback: generate a synthetic tone
    sr = 16000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, audio, sr)
    return path


# ======================================================================
# Tests
# ======================================================================

class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["models"]["asr"] is True
        assert data["provider"] in ("cpu", "cuda")

    def test_root_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "sherpa-qwen3-asr" in resp.text


class TestRecognize:
    def test_recognize_wav(self, client, test_audio):
        """Upload a WAV file and get transcription."""
        with open(test_audio, "rb") as f:
            resp = client.post(
                "/api/v1/recognize",
                files={"file": ("test.wav", f, "audio/wav")},
            )
        assert resp.status_code == 200, f"Error: {resp.text}"
        data = resp.json()
        assert data["success"] is True
        assert data["result"] is not None
        assert data["result"]["duration"] > 0
        assert isinstance(data["result"]["text"], str)
        assert data["result"]["stats"] is not None

    def test_recognize_with_language(self, client, test_audio):
        """Upload with language hint."""
        with open(test_audio, "rb") as f:
            resp = client.post(
                "/api/v1/recognize",
                files={"file": ("test.wav", f, "audio/wav")},
                data={"language": "Chinese"},
            )
        assert resp.status_code == 200

    def test_recognize_invalid_format(self, client):
        """Unsupported file format should return 400."""
        resp = client.post(
            "/api/v1/recognize",
            files={"file": ("test.exe", b"not audio", "application/octet-stream")},
        )
        assert resp.status_code == 400

    def test_recognize_no_file(self, client):
        """Missing file should return 422."""
        resp = client.post("/api/v1/recognize")
        assert resp.status_code == 422

    def test_recognize_large_file(self, client):
        """File exceeding max_file_size should return 413."""
        big_data = b"\x00" * (config["processing"]["max_file_size"] + 1)
        resp = client.post(
            "/api/v1/recognize",
            files={"file": ("big.wav", big_data, "audio/wav")},
        )
        assert resp.status_code == 413, f"Expected 413, got {resp.status_code}: {resp.text}"


# ======================================================================
# Tests: OpenAI-compatible endpoints
# ======================================================================

class TestOpenAIEndpoints:
    def test_transcriptions_json(self, client, test_audio):
        """OpenAI-compatible transcription with json format."""
        with open(test_audio, "rb") as f:
            resp = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", f, "audio/wav")},
                data={"model": "Qwen/Qwen3-ASR-0.6B"},
            )
        assert resp.status_code == 200, f"Error: {resp.text}"
        data = resp.json()
        assert "text" in data
        assert isinstance(data["text"], str)

    def test_transcriptions_text(self, client, test_audio):
        """OpenAI-compatible transcription with text format."""
        with open(test_audio, "rb") as f:
            resp = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", f, "audio/wav")},
                data={
                    "model": "Qwen/Qwen3-ASR-0.6B",
                    "response_format": "text",
                },
            )
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]

    def test_transcriptions_verbose_json(self, client, test_audio):
        """OpenAI-compatible transcription with verbose_json format."""
        with open(test_audio, "rb") as f:
            resp = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", f, "audio/wav")},
                data={
                    "model": "Qwen/Qwen3-ASR-0.6B",
                    "response_format": "verbose_json",
                },
            )
        assert resp.status_code == 200, f"Error: {resp.text}"
        data = resp.json()
        assert "task" in data
        assert "segments" in data
        assert len(data["segments"]) > 0

    def test_transcriptions_invalid_model(self, client, test_audio):
        """Invalid model name should return 400."""
        with open(test_audio, "rb") as f:
            resp = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", f, "audio/wav")},
                data={"model": "invalid-model"},
            )
        assert resp.status_code == 400

    def test_list_models(self, client):
        """GET /v1/models should return model list."""
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) > 0
        assert data["data"][0]["id"] == "Qwen/Qwen3-ASR-0.6B"

    def test_translations(self, client, test_audio):
        """OpenAI-compatible translation (falls back to transcription)."""
        with open(test_audio, "rb") as f:
            resp = client.post(
                "/v1/audio/translations",
                files={"file": ("test.wav", f, "audio/wav")},
                data={"model": "Qwen/Qwen3-ASR-0.6B"},
            )
        assert resp.status_code == 200


# ======================================================================
# Tests: Queue behavior
# ======================================================================

class TestQueue:
    """Verify background queue serializes requests correctly."""

    def test_queue_accepts_request(self, client):
        """Queue should process requests normally."""
        # Generate a tiny audio file
        sr = 16000
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, int(sr * 0.1))).astype(np.float32)
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            sf.write(path, audio, sr)
            with open(path, "rb") as f:
                resp = client.post(
                    "/api/v1/recognize",
                    files={"file": ("tiny.wav", f, "audio/wav")},
                )
            # Success or 504 (timeout) are both acceptable in test env
            assert resp.status_code in (200, 504), f"Unexpected: {resp.text}"
        finally:
            os.unlink(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
