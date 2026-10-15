"""Audio Cascader — File-Mode Speech-to-Speech (S2S) Pipeline.

Takes an Indian English input audio file, transcribes verbatim via Faster-Whisper Turbo GPU,
chunks on punctuation/pauses, and generates American-accented speech with Kyutai Pocket TTS.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf
import torch

# Ensure package imports resolve
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core.audio_convert import convert_to_wav
from core.chunker import chunk_text
from core.stt_engine import STTEngine
from core.tts_engine import TTSEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AudioCascader")


def run_pipeline(
    input_audio: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    play_audio: bool = False,
) -> Path:
    """Execute end-to-end file S2S conversion."""
    t_start = time.perf_counter()
    logger.info("=" * 60)
    logger.info("Audio Cascader — File-Mode S2S Conversion")
    logger.info("=" * 60)

    # 1. Audio Ingestion & Normalization to 16kHz WAV
    in_path = Path(input_audio).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input audio file not found: {in_path}")

    logger.info("[Step 1/5] Ingesting input audio: %s", in_path.name)
    wav_16k = convert_to_wav(in_path)

    # 2. STT: Faster-Whisper Turbo GPU
    logger.info("[Step 2/5] Transcribing speech with Faster-Whisper Turbo (%s)...", device)
    stt = STTEngine(device=device)
    t_stt0 = time.perf_counter()
    transcription = stt.transcribe(wav_16k, language="en", beam_size=1)
    stt_duration = time.perf_counter() - t_stt0

    text = transcription["text"].strip()
    audio_dur = transcription["duration"]
    rtf_stt = stt_duration / max(audio_dur, 0.01)

    logger.info("Transcribed Indian English: \"%s\"", text)
    logger.info("STT Latency: %.2fs | Audio Duration: %.2fs | STT RTF: %.3f", stt_duration, audio_dur, rtf_stt)

    # Free STT VRAM and memory before TTS
    stt.unload()

    if not text:
        logger.warning("No speech transcribed. Output audio will not be generated.")
        return in_path

    # 3. Punctuation & Pause Chunking
    logger.info("[Step 3/5] Chunking transcribed text...")
    chunks = chunk_text(text, max_words=12)
    logger.info("Generated %d clauses for streaming synthesis:", len(chunks))
    for i, c in enumerate(chunks, 1):
        logger.info("  [%d/%d] \"%s\"", i, len(chunks), c)

    # 4. Pocket TTS American Accent Synthesis
    logger.info("[Step 4/5] Initializing Kyutai Pocket TTS with American accent embedding...")
    tts = TTSEngine(device=device)
    voice_state = tts.get_or_create_voice_state()

    generated_audio_chunks = []
    t_tts0 = time.perf_counter()
    first_chunk_latency = None

    for idx, clause in enumerate(chunks, 1):
        t_c0 = time.perf_counter()
        audio_chunk = tts.synthesize_chunk(clause, voice_state=voice_state)
        chunk_time = time.perf_counter() - t_c0
        if first_chunk_latency is None:
            first_chunk_latency = chunk_time

        logger.info("  Clause %d/%d rendered (%d samples, %.2fs)", idx, len(chunks), len(audio_chunk), chunk_time)
        if len(audio_chunk) > 0:
            generated_audio_chunks.append(audio_chunk)

    total_tts_duration = time.perf_counter() - t_tts0

    if not generated_audio_chunks:
        raise RuntimeError("TTS failed to generate any audio samples.")

    # 5. Concatenation, Normalization, and Export
    full_audio = np.concatenate(generated_audio_chunks)
    full_audio = TTSEngine.normalize_audio(full_audio, target_peak=0.90)

    # Determine output path
    if output_path is None:
        outputs_dir = BASE_DIR / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = outputs_dir / f"pipeline_output_{timestamp}.wav"
    else:
        out_file = Path(output_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("[Step 5/5] Writing output WAV to: %s", out_file)
    sf.write(str(out_file), full_audio, 24000, subtype="PCM_16")

    total_pipeline_time = time.perf_counter() - t_start
    output_duration = len(full_audio) / 24000.0

    print("\n" + "=" * 60)
    print("AUDIO CASCADER — PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Input Audio:         {in_path.name} ({audio_dur:.2f}s)")
    print(f"Transcribed Text:    \"{text}\"")
    print(f"Output File:         {out_file.name} ({output_duration:.2f}s)")
    print(f"STT Time:            {stt_duration*1000:.1f} ms (RTF: {rtf_stt:.3f})")
    print(f"First Chunk TTFB:    {(first_chunk_latency or 0)*1000:.1f} ms")
    print(f"Total TTS Time:      {total_tts_duration*1000:.1f} ms")
    print(f"Glass-to-Glass Total:{total_pipeline_time:.2f} s")
    print("=" * 60 + "\n")

    if play_audio:
        try:
            import sounddevice as sd
            logger.info("Playing output audio through default sound device...")
            sd.play(full_audio, 24000)
            sd.wait()
        except Exception as e:
            logger.warning("Could not play audio: %s", e)

    return out_file


def main():
    parser = argparse.ArgumentParser(description="Audio Cascader — Indian-to-American S2S Pipeline")
    parser.add_argument("--audio", "-a", required=True, help="Path to input audio file (.wav, .m4a, .mp3)")
    parser.add_argument("--output", "-o", default=None, help="Path to save output .wav file")
    parser.add_argument("--device", "-d", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--play", "-p", action="store_true", help="Play output audio upon completion")
    args = parser.parse_args()

    run_pipeline(
        input_audio=args.audio,
        output_path=args.output,
        device=args.device,
        play_audio=args.play,
    )


if __name__ == "__main__":
    main()
