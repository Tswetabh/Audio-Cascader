"""Faster-Whisper Turbo STT Engine for Audio Cascader.

Provides optimized CUDA int8_float16 ASR transcription with sub-second latency
for Indian English spoken audio.
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS_DIR = BASE_DIR / "weights" / "faster-whisper-turbo"


class STTEngine:
    """Wrapper for Faster-Whisper Turbo GPU inference."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        compute_type: Optional[str] = None,
        cpu_threads: int = 4,
    ):
        self.model_path = Path(model_path) if model_path else DEFAULT_WEIGHTS_DIR
        self.device = device
        if compute_type is None:
            self.compute_type = "int8_float16" if self.device == "cuda" else "int8"
        else:
            self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self._model: Optional[WhisperModel] = None

    def load(self) -> WhisperModel:
        """Load Faster-Whisper model into memory/VRAM."""
        if self._model is not None:
            return self._model

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model weights not found at: {self.model_path}")

        try:
            logger.info("Loading Faster-Whisper from %s on %s (%s)...", self.model_path, self.device, self.compute_type)
            self._model = WhisperModel(
                str(self.model_path),
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
            )
            logger.info("Faster-Whisper model loaded successfully.")
        except Exception as exc:
            if self.device == "cuda" and self.compute_type != "int8":
                logger.warning("Failed on %s (%s). Retrying with int8_float16 / int8...", self.compute_type, exc)
                try:
                    self.compute_type = "int8_float16"
                    self._model = WhisperModel(
                        str(self.model_path),
                        device="cuda",
                        compute_type="int8_float16",
                        cpu_threads=self.cpu_threads,
                    )
                except Exception:
                    self.device = "cpu"
                    self.compute_type = "int8"
                    self._model = WhisperModel(
                        str(self.model_path),
                        device="cpu",
                        compute_type="int8",
                        cpu_threads=self.cpu_threads,
                    )
            else:
                raise exc

        return self._model

    def unload(self) -> None:
        """Unload model and free GPU VRAM."""
        if self._model is not None:
            del self._model
            self._model = None
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Faster-Whisper model unloaded from VRAM.")

    def transcribe(
        self,
        audio: Union[str, Path, np.ndarray],
        language: Optional[str] = "en",
        beam_size: int = 1,
        condition_on_previous_text: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Transcribe an audio waveform array or file path.

        Returns:
            Dictionary with:
              - text: Full combined transcribed text
              - segments: List of dicts {start, end, text}
              - language: Detected/specified language
              - duration: Audio duration in seconds
              - first_word: First recognized word
              - first_word_time_ms: Latency to decode first word in milliseconds
        """
        model = self.load()

        if isinstance(audio, (str, Path)):
            audio_path = Path(audio).resolve()
            if not audio_path.is_file():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            audio_data, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
            if getattr(audio_data, "ndim", 1) > 1:
                audio_data = np.mean(audio_data, axis=1)
            audio_data = np.asarray(audio_data, dtype=np.float32)
            if sr != 16000:
                try:
                    import librosa
                    audio_data = librosa.resample(audio_data, orig_sr=int(sr), target_sr=16000).astype(np.float32)
                except ImportError:
                    import scipy.signal
                    num_samples = int(len(audio_data) * 16000 / sr)
                    audio_data = scipy.signal.resample(audio_data, num_samples).astype(np.float32)
            audio_input = audio_data
        else:
            audio_input = np.asarray(audio, dtype=np.float32)

        duration = len(audio_input) / 16000.0

        t0_gen = time.perf_counter()
        segments_gen, info = model.transcribe(
            audio_input,
            language=language,
            beam_size=beam_size,
            condition_on_previous_text=condition_on_previous_text,
            **kwargs,
        )

        first_word_time_ms = None
        first_word = ""
        segments: List[Dict[str, Any]] = []

        for s in segments_gen:
            if first_word_time_ms is None and s.text.strip():
                first_word_time_ms = (time.perf_counter() - t0_gen) * 1000.0
                words = s.text.strip().split()
                if words:
                    first_word = words[0]
            segments.append({"start": s.start, "end": s.end, "text": s.text})

        full_text = " ".join(s["text"].strip() for s in segments).strip()

        return {
            "text": full_text,
            "segments": segments,
            "language": info.language if hasattr(info, "language") else language,
            "duration": duration,
            "first_word": first_word,
            "first_word_time_ms": round(first_word_time_ms, 1) if first_word_time_ms else None,
        }
