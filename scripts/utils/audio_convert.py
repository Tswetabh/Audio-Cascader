"""Audio format conversion utilities.

Converts .m4a, .mp3, etc. to 16kHz mono 16-bit PCM WAV using PyAV.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import soundfile as sf


def convert_to_wav(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    target_sr: int = 16000,
) -> Path:
    """Convert any supported audio file (e.g. .m4a, .mp3) to 16kHz mono WAV.

    If input_path is already a WAV file with target_sr, returns it directly
    (unless output_path is explicitly specified).
    """
    in_path = Path(input_path).resolve()
    if not in_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {in_path}")

    # If already a .wav and no distinct output path requested
    if in_path.suffix.lower() == ".wav" and output_path is None:
        return in_path

    if output_path is None:
        out_path = in_path.with_suffix(".wav")
    else:
        out_path = Path(output_path).resolve()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Use PyAV for robust decoding of m4a/aac/mp3/etc.
    try:
        import av
    except ImportError:
        # Fallback to soundfile directly if supported
        try:
            audio_data, sr = sf.read(str(in_path), dtype="float32")
            if getattr(audio_data, "ndim", 1) > 1:
                audio_data = np.mean(audio_data, axis=1)
            if sr != target_sr:
                import scipy.signal
                num_samples = int(len(audio_data) * target_sr / sr)
                audio_data = scipy.signal.resample(audio_data, num_samples).astype(np.float32)
            sf.write(str(out_path), audio_data, target_sr, subtype="PCM_16")
            return out_path
        except Exception as e:
            raise ImportError(
                "PyAV ('av') is required to decode compressed audio (.m4a/.mp3). "
                "Please run using the project virtual environment (DNXS-Spokenword-Pocket-TTS-GPU\\.venv) "
                f"or install av. Underneath error: {e}"
            )
    resampler = av.AudioResampler(format="s16", layout="mono", rate=target_sr)
    frames = []

    for frame in container.decode(audio=0):
        for resampled_frame in resampler.resample(frame):
            frames.append(resampled_frame.to_ndarray())

    # Flush resampler buffer
    for resampled_frame in resampler.resample(None):
        frames.append(resampled_frame.to_ndarray())

    if not frames:
        raise ValueError(f"No audio stream found in: {in_path}")

    audio_data = np.concatenate(frames, axis=1).squeeze()
    sf.write(str(out_path), audio_data, target_sr, subtype="PCM_16")

    return out_path
