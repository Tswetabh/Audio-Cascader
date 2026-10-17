"""Exercise the full live path without a microphone.

Feeds a short spoken clip into "CABLE Input", then runs a real LiveSession
with input = "CABLE Output" - the live mic path (16 kHz capture, RMS
endpointing, incremental whisper, chunking, Pocket TTS, streamed playback)
runs exactly as in production. The converted voice plays briefly on the
default output device (speakers), and the session recording lands in
outputs/live_*.wav.

Usage:
    .venv\\Scripts\\python.exe tools\\live_cable_smoke.py

Exit code 0 = spoke + converted + recorded with no errors.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import sounddevice as sd
import soundfile as sf

from app import OUTPUT_RATE, _resample, device_extra_settings, LiveSession
from cable_check import find_cable_devices, pick_source

SETTLE_S = 3.0  # models preload inside start(); this covers 2 s ambient calibration
BUDGET_S = 120.0
MAX_CLIP_S = 3.0


def trim_to_content(audio: "np.ndarray", rate: int, pad_s: float = 0.25) -> "np.ndarray":
    mask = np.abs(audio) > 0.01
    if not mask.any():
        return audio
    first = max(0, int(np.argmax(mask)) - int(pad_s * rate))
    last = min(len(audio), len(audio) - int(np.argmax(mask[::-1])) + int(pad_s * rate))
    return audio[first:last]


def main() -> int:
    render, capture = find_cable_devices()
    default_output = sd.query_devices(kind="output")["index"]
    print("render (clip -> cable):", render, sd.query_devices(render)["name"])
    print("capture (session mic):", capture, sd.query_devices(capture)["name"])
    print("session output (speakers):", default_output, sd.query_devices(default_output)["name"])

    source = pick_source()
    audio, source_rate = sf.read(str(source), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = _resample(audio, int(source_rate), OUTPUT_RATE)
    audio = trim_to_content(audio, OUTPUT_RATE)[: int(MAX_CLIP_S * OUTPUT_RATE)]
    audio = (audio / max(float(np.abs(audio).max()), 1e-6) * 0.9).astype(np.float32)
    print(f"clip: {source.name} -> {len(audio) / OUTPUT_RATE:.2f} s")

    session = LiveSession()
    channel = session.hub.subscribe()
    spoken: list[str] = []
    converted: list[str] = []
    errors: list[str] = []
    try:
        session.start(input_device=capture, output_device=default_output)
        print(f"live session started; settling {SETTLE_S:.0f} s (load + calibration)...")
        time.sleep(SETTLE_S)

        print("playing clip into VB-CABLE...")
        stereo = np.stack([audio, audio], axis=1)
        sd.play(stereo, samplerate=OUTPUT_RATE, device=render,
                extra_settings=device_extra_settings(render, "output"), blocking=True)
        print("clip done; waiting for transcription and converted playback...")

        deadline = time.monotonic() + BUDGET_S
        last_event = time.monotonic()
        saw_audio = False
        quiet_since: float | None = None
        while time.monotonic() < deadline:
            try:
                event = channel.get(timeout=0.25)
            except Exception:
                event = None
            if event is not None:
                last_event = time.monotonic()
                quiet_since = None
                kind = event.get("kind")
                if kind == "spoken" and event.get("text", "").strip():
                    spoken.append(event["text"].strip())
                    print("  spoken   :", spoken[-1])
                elif kind == "converted" and event.get("text", "").strip():
                    converted.append(event["text"].strip())
                    print("  converted:", converted[-1])
                elif kind == "error":
                    errors.append(str(event.get("error")))
                    print("  error    :", errors[-1])
            if session.playback.samples > 0:
                saw_audio = True
            # TTS chunk events are published before generation; only stop once
            # audio has actually played and stayed quiet for a safe tail.
            if saw_audio and spoken and converted and session.playback.samples == 0:
                if quiet_since is None:
                    quiet_since = time.monotonic()
                elif time.monotonic() - quiet_since > 6.0:
                    break
            if channel.empty() and time.monotonic() - last_event > 25:
                print("  idle 25 s with no events; stopping early")
                break
    finally:
        session.hub.unsubscribe(channel)
        session.stop()

    recording = session.last_recording or {}
    duration = float(recording.get("duration_s") or 0.0)
    print("\nresult:")
    print("  spoken events   :", len(spoken))
    print("  converted chunks:", len(converted))
    print("  errors          :", errors or "none")
    print("  recording       :", recording.get("wav", "(none)"), f"({duration:.2f} s)")
    ok = bool(spoken) and bool(converted) and not errors and duration > 0.5
    print("  verdict         :", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
