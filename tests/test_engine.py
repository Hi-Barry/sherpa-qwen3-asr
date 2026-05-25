"""
Tests for SpeechEngine — Qwen3 ASR transcription.

Tests are split into:
  - Unit tests: no model required (audio loading)
  - Integration tests: require models to be downloaded (auto-skipped otherwise)
"""
import sys
sys.path.insert(0, "..")

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.engine import SpeechEngine, TARGET_SAMPLE_RATE, ALLOWED_EXTENSIONS


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def sample_audio() -> np.ndarray:
    """Generate a 1-second sine tone at 16kHz."""
    t = np.linspace(0, 1.0, TARGET_SAMPLE_RATE, endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio


@pytest.fixture
def audio_file(sample_audio) -> str:
    """Save sample audio to a temp WAV file."""
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, sample_audio, TARGET_SAMPLE_RATE)
    yield path
    os.unlink(path)


@pytest.fixture
def engine() -> SpeechEngine:
    """Load real Qwen3 model (skipped if models not downloaded)."""
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "config.yaml"

    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Resolve relative paths from project root
    models_dir = project_root / config["models"]["qwen3_asr_dir"]
    if not models_dir.exists():
        pytest.skip(f"Qwen3 model not found at {models_dir}. Run scripts/download_models.sh first.")

    config["models"]["qwen3_asr_dir"] = str(models_dir)

    return SpeechEngine(config)


# ======================================================================
# Tests: Audio Loading (no model needed)
# ======================================================================

class TestLoadAudio:
    def test_load_wav(self, audio_file):
        """Load a standard WAV file."""
        audio, sr = SpeechEngine.load_audio(audio_file)
        assert sr == TARGET_SAMPLE_RATE
        assert audio.dtype == np.float32
        assert len(audio) == TARGET_SAMPLE_RATE  # 1 second

    def test_stereo_to_mono(self):
        """6-channel audio should be converted to mono (channel 0)."""
        stereo = np.zeros((TARGET_SAMPLE_RATE, 6), dtype=np.float32)
        stereo[:, 0] = 0.5
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(path, stereo, TARGET_SAMPLE_RATE)
        try:
            audio, sr = SpeechEngine.load_audio(path)
            assert audio.ndim == 1
            assert np.allclose(audio[:100], 0.5)
        finally:
            os.unlink(path)

    def test_resample(self):
        """Audio at 48kHz should be resampled to 16kHz."""
        sr_48k = 48000
        t = np.linspace(0, 1.0, sr_48k, endpoint=False)
        audio_48k = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(path, audio_48k, sr_48k)
        try:
            audio, sr = SpeechEngine.load_audio(path)
            assert sr == TARGET_SAMPLE_RATE
            assert len(audio) < sr_48k  # Downsampled
        finally:
            os.unlink(path)

    def test_invalid_format(self):
        """Non-audio file should raise RuntimeError."""
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        with open(path, "w") as f:
            f.write("not audio data")
        try:
            with pytest.raises(RuntimeError):
                SpeechEngine.load_audio(path)
        finally:
            os.unlink(path)

    def test_allowed_extensions(self):
        """Verify common audio formats are supported."""
        assert ".wav" in ALLOWED_EXTENSIONS
        assert ".mp3" in ALLOWED_EXTENSIONS
        assert ".flac" in ALLOWED_EXTENSIONS
        assert ".ogg" in ALLOWED_EXTENSIONS
        assert ".m4a" in ALLOWED_EXTENSIONS


# ======================================================================
# Tests: Model Initialization (requires downloaded models)
# ======================================================================

class TestEngineInit:
    def test_init_with_real_model(self, engine):
        """Load real Qwen3 ASR model."""
        assert engine.is_ready
        assert engine._recognizer is not None

    def test_provider(self, engine):
        """Provider should be readable."""
        assert engine.provider in ("cpu", "cuda")


# ======================================================================
# Tests: Real ASR Inference (requires downloaded models + test audio)
# ======================================================================

class TestAsrInference:
    def test_process_silence(self, engine):
        """Processing near-silence should return empty or minimal text."""
        silent = np.zeros(TARGET_SAMPLE_RATE * 2, dtype=np.float32)  # 2s silence
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(path, silent, TARGET_SAMPLE_RATE)
        try:
            result = engine.process(path)
            assert result.duration == 2.0
            assert isinstance(result.text, str)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
