"""StyleTTS 2 Model Integration Adapter.

Provides a clean interface for loading, running, and unloading StyleTTS 2
with zero-shot voice cloning capabilities.
"""

from __future__ import annotations

import gc
import logging
import os
from pathlib import Path
import sys
from typing import Any, Optional, Tuple, Union

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WEIGHTS_DIR = REPO_ROOT / "weights" / "tts" / "styletts2"


class StyleTTS2Adapter:
    """Adapter for StyleTTS 2 inference and voice cloning."""

    def __init__(self, device: str = "auto"):
        if device == "cuda" or (device == "auto" and torch.cuda.is_available()):
            self.device = "cuda"
        else:
            self.device = "cpu"
        self._model = None
        self.sample_rate = 24000
        self.has_voice_cloning = True

    def load(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        config_path: Optional[Union[str, Path]] = None,
    ):
        """Load and return the underlying StyleTTS2 model instance."""
        if self._model is not None:
            return self._model

        logger.info(f"Loading StyleTTS 2 model on device: {self.device}")
        from styletts2 import tts

        kwargs = {}
        if checkpoint_path and Path(checkpoint_path).exists():
            kwargs["model_checkpoint_path"] = str(checkpoint_path)
        if config_path and Path(config_path).exists():
            kwargs["config_path"] = str(config_path)

        # Use locally cached checkpoints/configs directly without network overhead or timeouts
        import cached_path as cp_mod
        from cached_path._cached_path import get_cache_dir, find_latest_cached
        orig_cached_path = cp_mod.cached_path

        def _fast_cached_path(url_or_filename, *args, **kw):
            if isinstance(url_or_filename, str) and url_or_filename.startswith("http"):
                try:
                    cached_file = find_latest_cached(url_or_filename, get_cache_dir())
                    if cached_file and Path(cached_file).exists():
                        return Path(cached_file)
                except Exception:
                    pass
            return orig_cached_path(url_or_filename, *args, **kw)

        cp_mod.cached_path = _fast_cached_path
        tts.cached_path = _fast_cached_path

        # PyTorch 2.6+ defaults weights_only=True; wrap torch.load to permit StyleTTS2 weights
        orig_load = torch.load
        def _safe_load(*args, **kw):
            if "weights_only" not in kw:
                kw["weights_only"] = False
            return orig_load(*args, **kw)

        torch.load = _safe_load
        try:
            self._model = tts.StyleTTS2(**kwargs)
        finally:
            torch.load = orig_load
            cp_mod.cached_path = orig_cached_path

        # Ensure device matches if needed
        if hasattr(self._model, "device"):
            self._model.device = self.device
        return self._model

    def is_loaded(self) -> bool:
        """Check if model is currently loaded in memory."""
        return self._model is not None

    def unload(self):
        """Evict model from RAM/VRAM to free resources."""
        if self._model is not None:
            logger.info("Unloading StyleTTS 2 model from memory...")
            del self._model
            self._model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_voice_state(self, prompt: Union[str, Path]):
        """Compute conditioning style vector for a reference voice audio file."""
        model = self.load()
        path_str = str(prompt)
        if not Path(path_str).exists():
            logger.warning(f"Voice prompt path not found: {path_str}. Using default voice.")
            return None
        return model.compute_style(path_str)

    def generate(
        self,
        text: str,
        target_voice_path: Optional[Union[str, Path]] = None,
        ref_style: Any = None,
        output_wav_file: Optional[Union[str, Path]] = None,
        alpha: float = 0.3,
        beta: float = 0.7,
        diffusion_steps: int = 5,
        embedding_scale: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        """Synthesize speech from text using StyleTTS 2 voice cloning.

        Args:
            text: Text to synthesize.
            target_voice_path: Optional path to reference WAV file for voice cloning.
            ref_style: Optional pre-computed style tensor from get_voice_state().
            output_wav_file: Optional file path to save generated WAV.
            alpha: Timbre balance (default: 0.3).
            beta: Prosody balance (default: 0.7).
            diffusion_steps: Diffusion sampling steps (default: 5).
            embedding_scale: Emotional/expressive scaling (default: 1.0).

        Returns:
            Tuple of (audio_numpy_array, sample_rate).
        """
        model = self.load()

        # Compute style if not pre-computed and target voice is given
        if ref_style is None and target_voice_path:
            p = Path(target_voice_path)
            if p.is_file():
                ref_style = model.compute_style(str(p))

        audio = model.inference(
            text=text,
            target_voice_path=str(target_voice_path) if target_voice_path else None,
            output_wav_file=str(output_wav_file) if output_wav_file else None,
            output_sample_rate=self.sample_rate,
            alpha=alpha,
            beta=beta,
            diffusion_steps=diffusion_steps,
            embedding_scale=embedding_scale,
            ref_s=ref_style,
        )

        if not isinstance(audio, np.ndarray):
            if hasattr(audio, "cpu"):
                audio = audio.cpu().numpy()
            else:
                audio = np.array(audio, dtype=np.float32)

        return audio, self.sample_rate
