"""\nSpeech Engine: Qwen3-ASR 0.6B int8 — pure file transcription.\n\nPure ONNX Runtime, no PyTorch. Follows the official sherpa-onnx\noffline-qwen3-asr-decode-files.py example exactly — no reinventing.\n\nArchitecture:\n    Upload audio → resample to 16kHz mono → force-chunk →\n    Qwen3 ASR (from_qwen3_asr) → merged text response\n"""
import logging
import time
import subprocess
import tempfile
import os
import atexit
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import soundfile as sf
import librosa
import scipy.signal as signal
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

        self._init_asr()
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

    # ------------------------------------------------------------------
    # Audio I/O
    # ------------------------------------------------------------------

    # ---- Formats that soundfile natively supports ----
    _SF_FORMATS = frozenset({".wav", ".flac", ".ogg", ".aiff", ".w64", ".caf"})

    @staticmethod
    def load_audio(audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load and normalize audio to 16kHz mono float32.

        Uses soundfile for natively supported formats (wav/flac/ogg).
        Falls back to ffmpeg for M4A, AAC, MP3, OPUS, WEBM and other
        formats that libsndfile cannot decode.

        Returns:
            (samples, sample_rate) — always 16000 Hz.
        """
        ext = Path(audio_path).suffix.lower()

        # If soundfile doesn't support this format, use ffmpeg to
        # decode to a temporary WAV first
        if ext not in SpeechEngine._SF_FORMATS:
            try:
                audio_path = SpeechEngine._ffmpeg_decode(audio_path)
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                raise RuntimeError(
                    f"Failed to decode audio file: {e}"
                ) from e

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

    @staticmethod
    def _ffmpeg_decode(audio_path: str) -> str:
        """
        Decode audio to WAV using ffmpeg.

        M4A/AAC/MP3/OPUS/WEBM are not natively supported by soundfile
        (libsndfile), so we delegate decoding to ffmpeg which handles
        virtually all audio formats.

        Returns:
            Path to a temporary WAV file (caller should clean up).
        """
        temp = tempfile.NamedTemporaryFile(
            suffix=".wav", prefix="sherpa_decode_", delete=False
        )
        temp_path = temp.name
        temp.close()

        # ffmpeg: decode to 16kHz mono 16-bit PCM WAV
        cmd = [
            "ffmpeg", "-y",
            "-i", audio_path,
            "-ar", str(TARGET_SAMPLE_RATE),
            "-ac", "1",
            "-sample_fmt", "s16",
            "-f", "wav",
            temp_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)

        # Schedule cleanup of the temp file on exit
        atexit.register(lambda p=temp_path: os.unlink(p) if os.path.exists(p) else None)

        return temp_path

    # ------------------------------------------------------------------
    # Audio Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply optional audio preprocessing: normalize + highpass filter.

        Controlled by config processing.preprocess section.
        """
        pre = self.config.get("processing", {}).get("preprocess", {})

        # Volume normalization to [-1.0, 1.0]
        if pre.get("normalize", True):
            max_val = np.max(np.abs(audio))
            if max_val > 1e-10:
                audio = audio / max_val
            logger.debug("Audio normalized")

        # High-pass filter to remove low-frequency noise (e.g., fans, hum)
        cutoff = pre.get("highpass_cutoff", 0)
        if cutoff > 0 and sr > 2 * cutoff:
            sos = signal.butter(4, cutoff, btype="highpass", fs=sr, output="sos")
            audio = signal.sosfilt(sos, audio).astype(np.float32)
            logger.debug(f"High-pass filter applied: {cutoff}Hz")

        return audio

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
        start_time = time.time()

        # Load and normalize
        audio, sr = self.load_audio(audio_path)

        # Optional preprocessing: normalize + highpass filter
        audio = self._preprocess(audio, sr)

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

        # Step 1: Get speech segments (full audio, then force-chunk)
        segments = self._get_speech_segments(audio, sr)
        logger.info(f"Chunks after force-split: {len(segments)}")
        n_before = len(segments)

        # Step 1b: Force-split long segments to fit model context window
        segments = self._chunk_segments(segments)
        if len(segments) != n_before:
            logger.info(
                f"After chunking: {len(segments)} segments "
                f"(was {n_before})"
            )

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
        Return the full audio as a single segment.

        No VAD — the entire file is one segment. Long segments are
        force-split by _chunk_segments() in process() to stay within
        Qwen3-ASR's context window.
        """
        duration = len(audio) / sr
        return [{"start": 0.0, "end": duration}]

    # ──────────────────────────────────────────────────────────────────────
    # Chunking — Safety guard against Qwen3-ASR finite context window
    # ──────────────────────────────────────────────────────────────────────

    def _chunk_segments(self, segments: List[dict]) -> List[dict]:
        """
        Force-split VAD segments that exceed max_chunk_duration.

        This is the PRIMARY safety guard. Qwen3-ASR has a finite context
        window (controlled by max_total_len). Even if VAD produces a clean
        speech segment, it may be too long for the model (e.g., 10 minutes
        of continuous speech with no pauses).

        Strategy: if a segment is longer than max_chunk_duration, split it
        evenly into N chunks, each ≤ max_chunk_duration. No overlap — the
        risk of cutting a word at a clean 30s boundary is negligible, and
        overlap would require deduplication logic.

        Each chunk retains the original segment's start/end for correct
        timestamp alignment.

        Returns:
            List of chunk dicts, always sorted by start time.
        """
        max_dur = self.config.get("processing", {}).get(
            "max_chunk_duration", 30
        )
        if max_dur <= 0:
            return segments  # chunking disabled

        result: List[dict] = []
        for seg in segments:
            start = seg["start"]
            end = seg["end"]
            duration = end - start

            if duration <= max_dur:
                result.append(seg)
                continue

            # Split into evenly-sized chunks
            n_chunks = int(np.ceil(duration / max_dur))
            chunk_size = duration / n_chunks
            logger.info(
                f"Chunking {duration:.1f}s segment into {n_chunks} "
                f"chunks of {chunk_size:.1f}s each"
            )

            for i in range(n_chunks):
                cs = start + i * chunk_size
                ce = start + (i + 1) * chunk_size
                if i == n_chunks - 1:
                    ce = end  # last chunk: exact end
                result.append({
                    "start": round(cs, 3),
                    "end": round(ce, 3),
                })

        return result

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
