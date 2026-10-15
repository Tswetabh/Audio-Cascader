"""Kyutai Pocket TTS Engine with Voice Embedding Caching.

Provides low-latency GPU speech synthesis conditioned on a persistent American accent
voice embedding (.pt), with automatic leading silence trimming and volume normalization.
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any, Dict, Iterator, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REF_VOICE = BASE_DIR / "references" / "american accent.wav"
DEFAULT_EMBEDDING_PATH = BASE_DIR / "embeddings" / "american_accent.pt"


class TTSEngine:
    """Wrapper for Kyutai Pocket TTS synthesis with cached voice state."""

    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        embedding_path: Optional[Union[str, Path]] = None,
        reference_voice: Optional[Union[str, Path]] = None,
    ):
        self.device = device
        self.embedding_path = Path(embedding_path) if embedding_path else DEFAULT_EMBEDDING_PATH
        self.reference_voice = Path(reference_voice) if reference_voice else DEFAULT_REF_VOICE
        self._model = None
        self.voice_state: Optional[Dict[str, Any]] = None

    def load(self):
        """Load Pocket TTS model into VRAM."""
        if self._model is not None:
            return self._model

        import sys
        engine_dir = BASE_DIR
        if str(engine_dir) not in sys.path:
            sys.path.insert(0, str(engine_dir))

        from pocket_tts.models.tts_model import TTSModel

        logger.info("Loading Pocket TTS model on device: %s...", self.device)
        self._model = TTSModel.load_model(device=self.device)
        logger.info("Pocket TTS model loaded successfully.")

        # Ensure voice state is initialized
        self.get_or_create_voice_state()
        return self._model

    def get_or_create_voice_state(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Load voice state from cached .pt file or compute and save it."""
        if self.voice_state is not None and not force_refresh:
            return self.voice_state

        # 1. Try loading cached embedding (.pt)
        if self.embedding_path.exists() and not force_refresh:
            t0 = time.perf_counter()
            logger.info("Loading cached voice embedding from %s...", self.embedding_path)
            self.voice_state = torch.load(str(self.embedding_path), map_location=self.device, weights_only=False)
            logger.info("Voice embedding loaded in %.1fms", (time.perf_counter() - t0) * 1000.0)
            return self.voice_state

        # 2. Compute embedding from reference audio file
        if self._model is None:
            import sys
            if str(BASE_DIR) not in sys.path:
                sys.path.insert(0, str(BASE_DIR))
            from pocket_tts.models.tts_model import TTSModel
            self._model = TTSModel.load_model(device=self.device)

        if not self.reference_voice.exists():
            logger.warning("Reference voice %s not found! Using built-in voice 'alba'", self.reference_voice)
            self.voice_state = self._model.get_state_for_audio_prompt("alba")
            return self.voice_state

        logger.info("Computing voice embedding from reference audio: %s...", self.reference_voice)
        t0 = time.perf_counter()
        self.voice_state = self._model.get_state_for_audio_prompt(str(self.reference_voice))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info("Voice conditioning computed in %.1fms.", elapsed_ms)

        # 3. Save to disk for fast <50ms future reloads
        self.embedding_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.voice_state, str(self.embedding_path))
        logger.info("Saved persistent voice embedding to: %s", self.embedding_path)

        return self.voice_state

    @staticmethod
    def trim_leading_silence(audio_np: np.ndarray, sr: int = 24000, threshold: float = 0.015) -> np.ndarray:
        """Trim FlowLM prompt conditioning silence from audio start."""
        if len(audio_np) == 0:
            return audio_np
        above = np.where(np.abs(audio_np) >= threshold)[0]
        if len(above) > 0 and above[0] > int(sr * 0.10):
            start_idx = max(0, above[0] - int(sr * 0.04))
            return audio_np[start_idx:]
        return audio_np

    @staticmethod
    def normalize_audio(audio_np: np.ndarray, target_peak: float = 0.90) -> np.ndarray:
        """Normalize peak amplitude to guarantee audible, clear volume."""
        if len(audio_np) == 0:
            return audio_np
        peak = float(np.max(np.abs(audio_np)))
        if peak > 0.001:
            return (audio_np / peak) * target_peak
        return audio_np

    def synthesize_stream(self, text: str, voice_state: Optional[Dict[str, Any]] = None) -> Iterator[np.ndarray]:
        """Yield raw 24kHz float32 audio chunks as they are generated."""
        model = self.load()
        state = voice_state or self.voice_state
        if state is None:
            state = self.get_or_create_voice_state()

        stream = model.generate_audio_stream(
            state,
            text_to_generate=text,
            copy_state=True,
        )

        for chunk_tensor in stream:
            chunk_np = chunk_tensor.detach().cpu().numpy()
            yield chunk_np

    def synthesize_chunk(self, text: str, voice_state: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Synthesize a text chunk to a clean, normalized 24kHz float32 audio array."""
        chunks = list(self.synthesize_stream(text, voice_state))
        if not chunks:
            return np.zeros(0, dtype=np.float32)

        full_audio = np.concatenate(chunks)
        trimmed = self.trim_leading_silence(full_audio, sr=24000)
        normalized = self.normalize_audio(trimmed, target_peak=0.90)
        return normalized
