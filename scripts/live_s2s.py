#!/usr/bin/env python3
"""Interactive CLI for Real-Time Speech-to-Speech (S2S) Live Voice Conversion.

Usage:
    python scripts/live_s2s.py --reference "Testinput/american accent.wav"
    python scripts/live_s2s.py --list-devices
"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import time

# Auto-redirect to dedicated virtual environment if executed with global Python
ROOT = Path(__file__).resolve().parents[1]
TARGET_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
if not TARGET_PYTHON.exists():
    TARGET_PYTHON = ROOT / "DNXS-Spokenword-Pocket-TTS-GPU" / ".venv" / "Scripts" / "python.exe"

if TARGET_PYTHON.exists():
    curr_exe = os.path.normcase(os.path.abspath(sys.executable))
    target_exe = os.path.normcase(str(TARGET_PYTHON.resolve()))
    if curr_exe != target_exe:
        result = subprocess.run([str(TARGET_PYTHON)] + sys.argv, check=False)
        sys.exit(result.returncode)

# Setup path imports
ENGINE_DIR = ROOT / "DNXS-Spokenword-Pocket-TTS-GPU"
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import torch
from scripts.live_engine import LiveS2SEngine
from scripts.utils.audio_stream import list_audio_devices

DEFAULT_REFERENCE_VOICE = ROOT / "Testinput" / "american accent.wav"


def format_vu(level: float, width: int = 14) -> str:
    """ASCII volume meter."""
    filled = min(width, int(level * 30 * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}]"


def print_devices():
    """Print available audio capture and playback devices."""
    inputs, outputs = list_audio_devices()
    print("\n" + "=" * 65)
    print("AVAILABLE AUDIO INPUT DEVICES (MICROPHONES):")
    print("=" * 65)
    for d in inputs:
        def_flag = " (DEFAULT)" if d["is_default_in"] else ""
        print(f"  [{d['index']:2d}] {d['name']}{def_flag}")

    print("\n" + "=" * 65)
    print("AVAILABLE AUDIO OUTPUT DEVICES (SPEAKERS / VIRTUAL CABLE):")
    print("=" * 65)
    for d in outputs:
        def_flag = " (DEFAULT)" if d["is_default_out"] else ""
        print(f"  [{d['index']:2d}] {d['name']}{def_flag}")
    print("=" * 65 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-Time Speech-to-Speech (S2S) Live Voice Conversion CLI"
    )
    parser.add_argument(
        "--reference",
        "-r",
        type=str,
        default=str(DEFAULT_REFERENCE_VOICE),
        help="Path to reference audio file for voice cloning (or predefined voice name)",
    )
    parser.add_argument(
        "--stt-model",
        type=str,
        default="tiny",
        choices=["tiny", "turbo"],
        help="Faster-Whisper model variant (tiny: ~60ms ultra-low latency, turbo: high accuracy)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Compute device for neural inference",
    )
    parser.add_argument(
        "--input-device",
        type=int,
        default=None,
        help="Audio input device index (run with --list-devices to view)",
    )
    parser.add_argument(
        "--output-device",
        type=int,
        default=None,
        help="Audio output device index (run with --list-devices to view)",
    )
    parser.add_argument(
        "--silence-ms",
        type=int,
        default=300,
        help="Acoustic pause duration (ms) triggering clause translation",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input and output devices and exit",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_devices:
        print_devices()
        return 0

    print("\n" + "=" * 70)
    print("   🎙️  REAL-TIME SPEECH-TO-SPEECH (S2S) LIVE CONVERSION ENGINE")
    print("=" * 70)
    print(f"  Inference Device   : {args.device.upper()}" + (f" ({torch.cuda.get_device_name(0)})" if args.device == "cuda" else ""))
    print(f"  ASR Architecture   : Faster-Whisper ({args.stt_model})")
    print(f"  TTS Architecture   : Kyutai Pocket TTS (FlowLM + Mimi 24kHz)")
    print(f"  Target Latency     : ~600ms – 1,000ms Glass-to-Glass")
    print(f"  Reference Voice    : {args.reference}")
    print(f"  Pause Threshold    : {args.silence_ms} ms")
    print("=" * 70 + "\n")

    current_status = ["Initializing..."]
    vu_state = [0.0, 0.0]

    def on_status_change(status: str):
        current_status[0] = status
        print(f"  [STATUS] {status}")

    def on_transcription_ready(text: str, stt_lat: float, dur: float):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n  [{ts}] 🗣️  Captured ({dur:.1f}s): \"{text}\" [ASR: {stt_lat:.1f}ms]")

    def on_latency_update(metrics: dict):
        total = metrics.get("total_latency_ms", 0.0)
        stt_fw = metrics.get("stt_first_word_ms", metrics.get("stt_latency_ms", 0.0))
        tts_fb = metrics.get("tts_first_bit_ms", metrics.get("ttfa_latency_ms", 0.0))
        tot_fb = metrics.get("total_first_bit_ms", stt_fw + tts_fb)
        first_word = metrics.get("first_word", "")
        fw_info = f" ('{first_word}')" if first_word else ""
        print(
            f"  ⚡ Live S2S Latency: {total:.0f} ms | "
            f"STT First Word: {stt_fw:.0f} ms{fw_info} | "
            f"TTS First Bit (TTFB): {tts_fb:.0f} ms | "
            f"Total First Audio: {tot_fb:.0f} ms | "
            f"Glass-to-Glass: {total:.0f} ms"
        )


    def on_vu_update(in_lvl: float, out_lvl: float):
        vu_state[0] = in_lvl
        vu_state[1] = out_lvl

    engine = LiveS2SEngine(
        reference_voice=args.reference,
        stt_model=args.stt_model,
        device=args.device,
        input_device_index=args.input_device,
        output_device_index=args.output_device,
        min_silence_duration_ms=args.silence_ms,
        on_status=on_status_change,
        on_transcription=on_transcription_ready,
        on_latency=on_latency_update,
        on_vu=on_vu_update,
    )

    try:
        engine.start()
        print("\n" + "-" * 70)
        print("  🟢 LIVE S2S RUNNING! Speak into your microphone...")
        print("  Press Ctrl+C to stop.")
        print("-" * 70 + "\n")

        # Keep alive
        while True:
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\n  Stopping Live S2S Engine...")
    finally:
        engine.stop()
        print("  Engine terminated cleanly. Goodbye!\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
