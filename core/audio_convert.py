"""Audio format conversion utilities.

Converts .m4a, .mp3, etc. to 16kHz mono 16-bit PCM WAV using PyAV or soundfile.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import soundfile as sf


def convert_to_wav(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    target_sr: int = 16000,
) -> Path:
    """Convert any supported audio file to 16kHz mono WAV."""
    in_path = Path(input_path).resolve()
    if not in_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {in_path}")

    if in_path.suffix.lower() == ".wav" and output_path is None:
        # Check sample rate
        info = sf.info(str(in_path))
        if info.samplerate == target_sr and info.channels == 1:
            return in_path

    if output_path is None:
        out_path = in_path.with_name(f"{in_path.stem}_16k.wav")
    else:
        out_path = Path(output_path).resolve()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import av
        container = av.open(str(in_path))
        resampler = av.AudioResampler(format="s16", layout="mono", rate=target_sr)
        chunks = []
        for frame in container.decode(audio=0):
            frame.pts = None
            resampled_frames = resampler.resample(frame)
            if resampled_frames:
                for rf in resampled_frames:
                    chunks.append(rf.to_ndarray())
        if chunks:
            raw_audio = np.concatenate(chunks, axis=1)[0].astype(np.float32) / 32768.0
            sf.write(str(out_path), raw_audio, target_sr, subtype="PCM_16")
            return out_path
    except Exception:
        pass

    # Fallback to soundfile + scipy
    audio_data, sr = sf.read(str(in_path), dtype="float32")
    if getattr(audio_data, "ndim", 1) > 1:
        audio_data = np.mean(audio_data, axis=1)
    if sr != target_sr:
        try:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=int(sr), target_sr=target_sr).astype(np.float32)
        except ImportError:
            import scipy.signal
            num_samples = int(len(audio_data) * target_sr / sr)
            audio_data = scipy.signal.resample(audio_data, num_samples).astype(np.float32)
    sf.write(str(out_path), audio_data, target_sr, subtype="PCM_16")
    return out_path
