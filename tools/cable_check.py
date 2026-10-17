"""Prove the app -> VB-CABLE -> Google Meet audio path objectively.

Plays a real converted clip into the "CABLE Input" render device using the
same stream configuration the live app uses (OUTPUT_RATE + WASAPI
auto-convert), records the matching "CABLE Output" capture device - exactly
the signal Google Meet reads when its microphone is set to CABLE Output -
then aligns the recording against the played signal (FFT cross-correlation)
and scores it.

Usage:
    .venv\\Scripts\\python.exe tools\\cable_check.py [source.wav]

Defaults to the newest outputs/converted_*.wav, falling back to
samples/anoop_voice_line.wav. Writes the recording and a metrics JSON to
outputs/. Exit code: 0 = pass, 1 = fail, 2 = VB-CABLE unavailable.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import sounddevice as sd
import soundfile as sf

from app import OUTPUT_RATE, _resample, align_and_score, cable_check_verdict, device_extra_settings

LEAD_S = 0.25
MAX_SOURCE_S = 5.0
BLOCK = 1024


def find_cable_devices() -> tuple[int, int]:
    """WASAPI indices of (CABLE Input render, CABLE Output capture)."""
    wasapi = next(
        (i for i, host in enumerate(sd.query_hostapis()) if host["name"] == "Windows WASAPI"),
        None,
    )
    if wasapi is None:
        raise RuntimeError("Windows WASAPI host API not found")
    render = capture = None
    for index, device in enumerate(sd.query_devices()):
        if device["hostapi"] != wasapi:
            continue
        name = str(device["name"])
        if render is None and "CABLE Input" in name and device["max_output_channels"] > 0:
            render = index
        if capture is None and "CABLE Output" in name and device["max_input_channels"] > 0:
            capture = index
    if render is None or capture is None:
        raise RuntimeError("VB-CABLE not found: need 'CABLE Input' (render) and 'CABLE Output' (capture)")
    return render, capture


def pick_source() -> Path:
    converted = sorted((ROOT / "outputs").glob("converted_*.wav"))
    if converted:
        return converted[-1]
    return ROOT / "samples" / "anoop_voice_line.wav"


def run_check(source_path: Path) -> dict:
    render, capture = find_cable_devices()
    render_name = str(sd.query_devices(render)["name"])
    capture_name = str(sd.query_devices(capture)["name"])
    capture_rate = int(sd.query_devices(capture)["default_samplerate"])

    audio, source_rate = sf.read(str(source_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.size == 0:
        raise ValueError(f"empty source: {source_path}")
    audio = _resample(audio, int(source_rate), OUTPUT_RATE)[: int(MAX_SOURCE_S * OUTPUT_RATE)]
    peak = float(np.abs(audio).max())
    if peak > 0:
        audio = (audio / peak * 0.9).astype(np.float32)
    lead = np.zeros(int(LEAD_S * OUTPUT_RATE), dtype=np.float32)
    played = np.concatenate([lead, audio, lead])
    stereo = np.stack([played, played], axis=1)

    chunks: list[np.ndarray] = []
    # Capture in the device's native domain for the full played duration + margin.
    needed = int((len(played) / OUTPUT_RATE + 0.5) * capture_rate)
    with sd.InputStream(
        device=capture,
        samplerate=capture_rate,
        channels=2,
        dtype="float32",
        blocksize=BLOCK,
    ) as inp:
        # Play exactly like the live app: OUTPUT_RATE on a WASAPI endpoint.
        sd.play(stereo, samplerate=OUTPUT_RATE, device=render, extra_settings=device_extra_settings(render, "output"))
        collected = 0
        while collected < needed:
            chunk, _overflowed = inp.read(BLOCK)
            chunks.append(chunk[:, 0].astype(np.float32))
            collected += len(chunk)
        sd.stop()

    recorded = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    if capture_rate != OUTPUT_RATE:
        recorded = _resample(recorded, capture_rate, OUTPUT_RATE)

    score = align_and_score(played, recorded)
    rms = float(np.sqrt(np.mean(np.square(recorded)))) if recorded.size else 0.0
    verdict = cable_check_verdict(float(score["peak_correlation"]), rms)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    wav_path = out_dir / f"cable_check_{stamp}.wav"
    sf.write(str(wav_path), recorded, OUTPUT_RATE, subtype="PCM_16")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source_path),
        "render_device": render_name,
        "capture_device": capture_name,
        "played_duration_s": round(len(played) / OUTPUT_RATE, 3),
        "recorded_duration_s": round(len(recorded) / OUTPUT_RATE, 3),
        "peak_correlation": round(float(score["peak_correlation"]), 4),
        "lag_ms": round(float(score["lag_samples"]) / OUTPUT_RATE * 1000.0, 2),
        "recorded_rms": round(rms, 6),
        "verdict": verdict,
        "recorded_wav": str(wav_path),
    }
    (out_dir / f"cable_check_{stamp}_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str]) -> int:
    source_path = Path(argv[1]) if len(argv) > 1 else pick_source()
    print(f"Cable loop-back check\n  source: {source_path}")
    try:
        report = run_check(source_path)
    except RuntimeError as exc:
        print(f"  unavailable: {exc}")
        return 2
    for key in ("render_device", "capture_device", "peak_correlation", "lag_ms", "recorded_rms", "verdict"):
        print(f"  {key}: {report[key]}")
    print(f"  recorded: {report['recorded_wav']}")
    return 0 if report["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
