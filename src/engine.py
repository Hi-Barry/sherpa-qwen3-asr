"""
Speech Engine: Qwen3-ASR 0.6B int8 + optional Silero VAD.

Pure ONNX Runtime, no PyTorch. Follows the official sherpa-onnx
offline-qwen3-asr-decode-files.py example exactly — no reinventing.

Architecture:
    Upload audio → resample to 16kHz mono → [Optional VAD] →
    Qwen3 ASR (from_qwen3_asr) → merged JSON response
"""
import logging
import time
import threading
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import soundfile as sf
import librosa
import sherpa_onnx

from .models import Segment, RecognitionResult, ProcessingStats

logger = logging.getLogger(__name__)

# ---- Audio constants ----
TARGET_SAMPLE_RATE = 16000

# ---- Supported audio formats ----
ALLOWED_EXTENSIONS = frozenset(
    {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".webm"}
)


class SpeechEngine:
    """
    Pure Qwen3-ASR speech recognition engine with optional VAD.

    Usage:
        engine = SpeechEngine(config_dict)
        result = engine.process("audio.wav", language="")
    """

    def __init__(self, config: dict):
        self.config = config
        self._recognizer: Optional[sherpa_onnx.OfflineRecognizer] = None
        self._vad: Optional[sherpa_onnx.VoiceActivityDetector] = None
        self._lock = threading.Lock()  # Serialize single-request processing

        self._init_asr()
        self._init_vad()
        logger.info("SpeechEngine initialized successfully")

    # ------------------------------------------------------------------
    # Model Initialization
    # ------------------------------------------------------------------

    def _init_asr(self):
        """Load Qwen3-ASR 0.6B int8 model via official from_qwen3_asr()."""
        asr_cfg = self.config["asr"]
        model_dir = Path(self.config["models"]["qwen3_asr_dir"])

        if not model_dir.exists():
            raise FileNotFoundError(
                f"Qwen3-ASR model directory not found: {model_dir}. "
                f"Run scripts/download_models.sh first."
            )

        # Required model files
        conv_frontend = self._find_file(model_dir, "conv_frontend.onnx")
        encoder = self._find_file(model_dir, "encoder.int8.onnx", "encoder.onnx")
        decoder = self._find_file(model_dir, "decoder.int8.onnx", "decoder.onnx")
        tokenizer = model_dir / "tokenizer"

        if not tokenizer.exists():
            raise FileNotFoundError(
                f"Tokenizer directory not found: {tokenizer}. "
                f"Expected vocab.json + merges.txt inside."
            )

        logger.info(f"Loading Qwen3-ASR from: {model_dir}")
        logger.info(f"  conv_frontend: {conv_frontend}")
        logger.info(f"  encoder:       {encoder}")
        logger.info(f"  decoder:       {decoder}")
        logger.info(f"  tokenizer:     {tokenizer}")
        logger.info(f"  feature_dim:   {asr_cfg.get('feature_dim', 128)}")
        logger.info(f"  provider:      {asr_cfg.get('provider', 'cpu')}")

        self._recognizer = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
            conv_frontend=str(conv_frontend),
            encoder=str(encoder),
            decoder=str(decoder),
            tokenizer=str(tokenizer),
            num_threads=asr_cfg.get("num_threads", 2),
            sample_rate=TARGET_SAMPLE_RATE,
            feature_dim=asr_cfg.get("feature_dim", 128),
            provider=asr_cfg.get("provider", "cpu"),
            debug=asr_cfg.get("debug", False),
            max_total_len=asr_cfg.get("max_total_len", 512),
            max_new_tokens=asr_cfg.get("max_new_tokens", 128),
            temperature=asr_cfg.get("temperature", 1e-6),
            top_p=asr_cfg.get("top_p", 0.8),
            seed=asr_cfg.get("seed", 42),
            hotwords=asr_cfg.get("hotwords", ""),
        )
        logger.info("Qwen3-ASR model loaded")

    def _init_vad(self):
        """Load Silero VAD for speech segmentation (optional)."""
        vad_cfg = self.config.get("vad", {})
        if not vad_cfg.get("enabled", True):
            logger.info("VAD disabled by config")
            return

        model_cfg = self.config["models"]
        vad_dir = Path(model_cfg.get("vad_dir", "models/vad"))
        vad_file = model_cfg.get("vad_model_file", "silero_vad.onnx")
        vad_path = vad_dir / vad_file

        if not vad_path.exists():
            logger.warning(f"VAD model not found at {vad_path}, VAD disabled")
            return

        logger.info(f"Loading VAD model: {vad_path}")
        vad_config = sherpa_onnx.VadModelConfig()
        vad_config.silero_vad.model = str(vad_path)
        vad_config.silero_vad.threshold = vad_cfg.get("threshold", 0.5)
        vad_config.silero_vad.min_silence_duration = vad_cfg.get(
            "min_silence_duration", 0.25
        )
        vad_config.silero_vad.min_speech_duration = vad_cfg.get(
            "min_speech_duration", 0.25
        )
        vad_config.silero_vad.max_speech_duration = vad_cfg.get(
            "max_speech_duration", 30
        )
        vad_config.sample_rate = TARGET_SAMPLE_RATE

        self._vad = sherpa_onnx.VoiceActivityDetector(
            vad_config,
            buffer_size_in_seconds=100,
        )
        logger.info("VAD model loaded")

    # ------------------------------------------------------------------
    # Audio I/O
    # ------------------------------------------------------------------

    @staticmethod
    def load_audio(audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load and normalize audio to 16kHz mono float32.

        Returns:
            (samples, sample_rate) — always 16000 Hz.
        """
        try:
            audio, sr = sf.read(audio_path, dtype="float32")
        except sf.LibsndfileError as e:
            raise RuntimeError(
                f"Failed to decode audio file: {e}"
            ) from e

        # Convert to mono
        if audio.ndim > 1:
            audio = audio[:, 0]

        # Resample if needed
        if sr != TARGET_SAMPLE_RATE:
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE
            )
            logger.debug(f"Resampled from {sr}Hz to {TARGET_SAMPLE_RATE}Hz")

        audio = audio.astype(np.float32)
        return audio, TARGET_SAMPLE_RATE

    # ------------------------------------------------------------------
    # Core Processing
    # ------------------------------------------------------------------

    def process(
        self,
        audio_path: str,
        language: str = "",
    ) -> RecognitionResult:
        """
        Process an audio file: load → [VAD] → Qwen3 ASR → merged result.

        Args:
            audio_path: Path to audio file (wav, mp3, flac, etc.).
            language:   Force transcription language (e.g. "Korean", "Chinese").
                        Empty string = auto-detect.

        Returns:
            RecognitionResult with transcribed segments.
        """
        with self._lock:
            start_time = time.time()

            # Load and normalize
            audio, sr = self.load_audio(audio_path)
            duration = len(audio) / sr
            logger.info(
                f"Processing {audio_path}: {duration:.1f}s, "
                f"language={language or 'auto'}"
            )

            # Validate duration
            max_dur = self.config.get("processing", {}).get(
                "max_audio_duration", 3600
            )
            if duration > max_dur:
                raise ValueError(
                    f"Audio too long ({duration:.0f}s). Max: {max_dur}s"
                )

            # Step 1: Get speech segments (VAD or full audio)
            segments = self._get_speech_segments(audio, sr)
            logger.info(f"Speech segments: {len(segments)}")

            # Step 2: Run Qwen3 ASR on each segment
            detected_lang = language if language else "unknown"
            total_asr_time = 0.0
            result_segments: List[Segment] = []

            for seg in segments:
                start_sample = int(seg["start"] * sr)
                end_sample = int(seg["end"] * sr)
                audio_slice = audio[start_sample:end_sample]

                if len(audio_slice) < sr * 0.2:
                    continue  # Skip segments shorter than 200ms

                text, asr_time = self._run_qwen3(audio_slice, sr, language)
                total_asr_time += asr_time

                if text and text.strip():
                    result_segments.append(
                        Segment(
                            start=round(seg["start"], 3),
                            end=round(seg["end"], 3),
                            text=text.strip(),
                        )
                    )

            # Full text concatenation
            full_text = " ".join(s.text for s in result_segments).strip()

            total_time = time.time() - start_time

            return RecognitionResult(
                language=detected_lang,
                duration=round(duration, 2),
                segments=result_segments,
                text=full_text,
                stats=ProcessingStats(
                    asr_time=round(total_asr_time, 2),
                    total_time=round(total_time, 2),
                ),
            )

    def _run_qwen3(
        self, audio: np.ndarray, sr: int, language: str = ""
    ) -> Tuple[str, float]:
        """
        Run Qwen3-ASR on a single audio segment.
        
        Follows the official sherpa-onnx offline-qwen3-asr-decode-files.py
        example exactly — create stream, set language, accept waveform, decode.
        
        Returns:
            (transcribed_text, elapsed_seconds)
        """
        if self._recognizer is None:
            raise RuntimeError("ASR model not initialized")

        t0 = time.time()

        stream = self._recognizer.create_stream()
        if language:
            stream.set_option("language", language)
        stream.accept_waveform(sample_rate=sr, waveform=audio)
        self._recognizer.decode_stream(stream)

        result = stream.result  # OfflineRecognitionResult
        elapsed = time.time() - t0

        text = (result.text or "").strip()
        logger.debug(f"Qwen3 ASR ({elapsed:.2f}s): {text[:80]}...")

        return text, elapsed

    def _get_speech_segments(
        self, audio: np.ndarray, sr: int
    ) -> List[dict]:
        """
        Get speech segments via VAD, or a single full-audio segment.

        Returns:
            [{"start": float, "end": float}, ...]
        """
        if self._vad is not None:
            return self._vad_segment(audio, sr)
        # No VAD → treat full audio as one segment
        duration = len(audio) / sr
        return [{"start": 0.0, "end": duration}]

    def _vad_segment(self, audio: np.ndarray, sr: int) -> List[dict]:
        """
        Silero VAD: detect speech segments, merge adjacent with small gap.
        VAD is discarded after use — no caching.
        """
        window_size = self._vad.config.silero_vad.window_size
        segments = []
        started = False
        speech_start = 0.0

        # Feed audio in windows through VAD
        for i in range(0, len(audio), window_size):
            window = audio[i : i + window_size]
            if len(window) < window_size:
                window = np.pad(window, (0, window_size - len(window)))
            self._vad.accept_waveform(window)

            t = (i + window_size) / sr

            if not started and self._vad.is_speech_detected():
                started = True
                speech_start = max(0, (i - window_size) / sr)

            if started and not self._vad.is_speech_detected():
                segments.append({
                    "start": round(speech_start, 3),
                    "end": round(t, 3),
                })
                started = False

        # Flush remaining speech
        if started:
            segments.append({
                "start": round(speech_start, 3),
                "end": round(len(audio) / sr, 3),
            })

        # Merge adjacent segments with small gap
        if segments:
            segments = self._merge_adjacent_segments(segments, gap_threshold=0.5)

        if not segments:
            # VAD found nothing → use full audio as one segment
            duration = len(audio) / sr
            segments = [{"start": 0.0, "end": duration}]

        return segments

    @staticmethod
    def _merge_adjacent_segments(
        segments: List[dict], gap_threshold: float = 0.5
    ) -> List[dict]:
        """Merge adjacent segments if gap is within threshold."""
        if not segments:
            return segments

        merged = [dict(segments[0])]
        for seg in segments[1:]:
            gap = seg["start"] - merged[-1]["end"]
            if gap <= gap_threshold:
                merged[-1]["end"] = seg["end"]
            else:
                merged.append(dict(seg))

        if len(merged) < len(segments):
            logger.debug(
                f"Segment merge: {len(segments)} → {len(merged)} "
                f"(gap={gap_threshold}s)"
            )
        return merged

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_file(directory: Path, *names: str) -> Path:
        """Find a file by trying names in order, fallback to any .onnx."""
        for name in names:
            path = directory / name
            if path.exists():
                return path
        # Fallback: any .onnx file
        onnx_files = list(directory.glob("*.onnx"))
        if onnx_files:
            return onnx_files[0]
        raise FileNotFoundError(
            f"No model file found in {directory}. "
            f"Tried: {', '.join(names)}"
        )

    @property
    def is_ready(self) -> bool:
        return self._recognizer is not None

    @property
    def provider(self) -> str:
        return self.config.get("asr", {}).get("provider", "cpu")
