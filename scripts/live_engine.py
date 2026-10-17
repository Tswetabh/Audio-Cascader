"""Real-Time Speech-to-Speech (S2S) Engine with Concurrent GPU Pipelining.

Connects continuous WASAPI microphone capture -> Silero VAD -> Faster-Whisper ASR
-> Kyutai Pocket TTS streaming audio synthesis -> Low-latency Audio Playback.
"""

from __future__ import annotations

import logging
from pathlib import Path
import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import soundfile as sf

from models.stt.faster_whisper.adapter import FasterWhisperAdapter
from pocket_tts.models.tts_model import TTSModel
from scripts.utils.audio_convert import convert_to_wav
from scripts.utils.audio_stream import (
    AudioCapture,
    StreamingAudioPlayer,
    StreamingVADSegmenter,
    list_audio_devices,
)

logger = logging.getLogger(__name__)


class LiveS2SEngine:
    """End-to-end real-time speech-to-speech voice changer pipeline."""

    def __init__(
        self,
        reference_voice: Optional[Union[str, Path]] = None,
        stt_model: str = "tiny",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        input_device_index: Optional[int] = None,
        output_device_index: Optional[int] = None,
        output_dir: Optional[Union[str, Path]] = None,
        min_silence_duration_ms: int = 300,
        max_speech_duration_s: float = 3.5,
        # Callbacks
        on_status: Optional[Callable[[str], None]] = None,
        on_transcription: Optional[Callable[[str, float, float], None]] = None,
        on_latency: Optional[Callable[[Dict[str, float]], None]] = None,
        on_vu: Optional[Callable[[float, float], None]] = None,
        on_audio_saved: Optional[Callable[[str, float], None]] = None,
    ):
        self.reference_voice_path = reference_voice
        self.stt_model_name = stt_model
        self.device = device
        self.input_device_index = input_device_index
        self.output_device_index = output_device_index
        self.output_dir = Path(output_dir) if output_dir else None
        self.min_silence_duration_ms = min_silence_duration_ms
        self.max_speech_duration_s = max_speech_duration_s

        # Event Callbacks
        self.on_status = on_status or (lambda status: None)
        self.on_transcription = on_transcription or (lambda text, lat, dur: None)
        self.on_latency = on_latency or (lambda metrics: None)
        self.on_vu = on_vu or (lambda in_vu, out_vu: None)
        self.on_audio_saved = on_audio_saved or (lambda path, dur: None)

        # Audio persistence buffers
        self._recorded_chunks: List[np.ndarray] = []
        self._recorded_lock = threading.Lock()
        self.sample_rate = 24000

        # Threading & Queues
        self._stop_event = threading.Event()
        self._stt_queue: queue.Queue[Tuple[np.ndarray, float, float]] = queue.Queue(maxsize=20)
        self._tts_queue: queue.Queue[Tuple[str, float, float]] = queue.Queue(maxsize=20)

        # Models
        self.stt_adapter: Optional[FasterWhisperAdapter] = None
        self.tts_model: Optional[TTSModel] = None
        self.voice_state: Optional[dict] = None

        # Audio Stream Handlers
        self.capture: Optional[AudioCapture] = None
        self.segmenter: Optional[StreamingVADSegmenter] = None
        self.player: Optional[StreamingAudioPlayer] = None

        # VU meter levels
        self._in_vu = 0.0
        self._out_vu = 0.0

        # Performance Metrics
        self.latest_metrics: Dict[str, float] = {
            "vad_latency_ms": 0.0,
            "stt_latency_ms": 0.0,
            "ttfa_latency_ms": 0.0,
            "total_latency_ms": 0.0,
        }

    def _update_in_vu(self, level: float):
        self._in_vu = level
        self.on_vu(self._in_vu, self._out_vu)

    def _update_out_vu(self, level: float):
        self._out_vu = level
        self.on_vu(self._in_vu, self._out_vu)

    def load_models(self):
        """Pre-load STT and TTS models concurrently onto target device."""
        self.on_status("Loading Faster-Whisper ASR model...")
        stt_compute = "float16" if self.device == "cuda" else "int8"
        self.stt_adapter = FasterWhisperAdapter(
            model_path=self.stt_model_name,
            device=self.device,
            compute_type=stt_compute,
        )
        self.stt_adapter.load()

        self.on_status("Loading Kyutai Pocket TTS model...")
        self.tts_model = TTSModel.load_model(device=self.device)

        # Set reference voice conditioning
        self.set_reference_voice(self.reference_voice_path)

        # Warmup GPU kernels for instant sub-second first response
        self.on_status("Warming up inference engines on GPU...")
        self._warmup()
        self.on_status("Engine ready")

    def set_reference_voice(self, ref_path: Optional[Union[str, Path]]):
        """Compute or update voice conditioning state."""
        self.reference_voice_path = ref_path
        if self.tts_model is None:
            return

        cached_pt = Path(__file__).resolve().parents[1] / "embeddings" / "american_accent.pt"
        if ref_path is not None and "american accent" in str(ref_path).lower() and cached_pt.exists():
            try:
                self.voice_state = torch.load(str(cached_pt), map_location=self.device, weights_only=False)
                self.on_status(f"Voice state pre-loaded from cached embedding: {cached_pt.name}")
                return
            except Exception:
                pass

        if ref_path is not None and Path(str(ref_path)).exists():
            resolved = Path(ref_path).resolve()
            if resolved.suffix.lower() != ".wav":
                resolved = convert_to_wav(resolved)
            self.voice_state = self.tts_model.get_state_for_audio_prompt(str(resolved))
            self.on_status(f"Voice state cached from: {resolved.name}")
        else:
            fallback = "alba"
            self.voice_state = self.tts_model.get_state_for_audio_prompt(fallback)
            self.on_status(f"Voice state set to built-in voice: {fallback}")

    def _warmup(self):
        """Warm up CUDA graphs and PyTorch runtime."""
        if self.stt_adapter and self.tts_model and self.voice_state:
            dummy_pcm = np.zeros(16000, dtype=np.float32)
            try:
                self.stt_adapter.transcribe(dummy_pcm, beam_size=1)
                for _ in self.tts_model.generate_audio_stream(
                    self.voice_state, "Ready.", copy_state=True
                ):
                    pass
            except Exception as e:
                logger.warning(f"Warmup notice: {e}")

    def start(self):
        """Start live capture, segmentation, and translation workers."""
        if self.stt_adapter is None or self.tts_model is None:
            self.load_models()

        self._stop_event.clear()

        # Initialize audio I/O
        self.segmenter = StreamingVADSegmenter(
            sample_rate=16000,
            chunk_size=512,
            speech_threshold=0.5,
            min_silence_duration_ms=self.min_silence_duration_ms,
            max_speech_duration_s=self.max_speech_duration_s,
        )

        self.capture = AudioCapture(
            sample_rate=16000,
            block_size=512,
            device_index=self.input_device_index,
            vu_callback=self._update_in_vu,
        )

        self.player = StreamingAudioPlayer(
            sample_rate=24000,
            block_size=1024,
            device_index=self.output_device_index,
            vu_callback=self._update_out_vu,
        )

        self.player.start()
        self.capture.start()

        # Start background workers
        self._threads = [
            threading.Thread(target=self._vad_worker, name="VADWorker", daemon=True),
            threading.Thread(target=self._stt_worker, name="STTWorker", daemon=True),
            threading.Thread(target=self._tts_worker, name="TTSWorker", daemon=True),
        ]
        for t in self._threads:
            t.start()

        self.on_status("Live conversion active - listening...")

    def stop(self):
        """Stop all streams and workers cleanly, and save complete live session audio."""
        self.on_status("Stopping live conversion...")
        self._stop_event.set()

        if self.capture:
            self.capture.stop()
        if self.player:
            self.player.stop()

        # Drain queues
        while not self._stt_queue.empty():
            try:
                self._stt_queue.get_nowait()
            except queue.Empty:
                break
        while not self._tts_queue.empty():
            try:
                self._tts_queue.get_nowait()
            except queue.Empty:
                break

        # Save session recording
        saved_file = self.save_recorded_audio()
        if saved_file and self.on_audio_saved:
            try:
                dur = sf.info(str(saved_file)).duration
                self.on_audio_saved(str(saved_file), dur)
            except Exception as e:
                logger.error(f"Error notifying saved live audio: {e}")

        self.on_status("Stopped")

    def save_recorded_audio(self, target_path: Optional[Union[str, Path]] = None) -> Optional[Path]:
        """Save all synthesized audio chunks from this live session to a WAV file."""
        with self._recorded_lock:
            if not self._recorded_chunks:
                return None
            full_audio = np.concatenate(self._recorded_chunks)

        if len(full_audio) == 0:
            return None

        # Peak normalize to 0.90 for clear, audible volume
        peak = float(np.max(np.abs(full_audio)))
        if peak > 0.001:
            full_audio = (full_audio / peak) * 0.90

        if target_path:
            out_file = Path(target_path)
        elif self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_file = self.output_dir / f"live_session_{ts}.wav"
        else:
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_file = Path("Output") / f"live_session_{ts}.wav"
            out_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            sf.write(str(out_file), full_audio, self.sample_rate, subtype="PCM_16")
            logger.info(f"Saved live session audio to: {out_file}")
            return out_file
        except Exception as e:
            logger.error(f"Failed to write live session WAV: {e}")
            return None

    def _vad_worker(self):
        """Pulls raw audio from mic and feeds VAD segmenter."""
        while not self._stop_event.is_set():
            chunk = self.capture.get_chunk(timeout=0.05)
            if chunk is None:
                continue

            result = self.segmenter.process_chunk(chunk)
            if result is not None:
                speech_audio, start_t, end_t = result
                try:
                    self._stt_queue.put_nowait((speech_audio, start_t, end_t))
                except queue.Full:
                    logger.warning("STT queue full, dropping segment")

    def _stt_worker(self):
        """Transcribes speech segments with Faster-Whisper."""
        while not self._stop_event.is_set():
            try:
                speech_audio, start_t, speech_end_t = self._stt_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            dur = len(speech_audio) / 16000.0
            t0 = time.perf_counter()
            self.on_status("Transcribing speech...")

            try:
                res = self.stt_adapter.transcribe(
                    speech_audio,
                    beam_size=1,
                    condition_on_previous_text=False,
                )
                text = res.get("text", "").strip()
            except Exception as e:
                logger.error(f"STT error: {e}")
                continue

            stt_latency_ms = (time.perf_counter() - t0) * 1000.0
            stt_first_word_ms = res.get("first_word_time_ms") or stt_latency_ms
            first_word = res.get("first_word", "")

            # Filter out empty or whitespace-only noise
            cleaned = "".join(c for c in text if c.isalnum() or c.isspace()).strip()
            if not cleaned or len(cleaned) < 2:
                self.on_status("Listening...")
                continue

            self.on_transcription(text, stt_latency_ms, dur)

            # Auto-chunk at punctuations (natural speech pauses) for instant low-latency synthesis
            try:
                from core.chunker import chunk_text
                clauses = chunk_text(text, max_words=8)
            except Exception:
                clauses = [text]

            for clause in clauses:
                try:
                    self._tts_queue.put_nowait((clause, speech_end_t, stt_latency_ms, stt_first_word_ms, first_word))
                except queue.Full:
                    logger.warning("TTS queue full, dropping utterance clause")

    def _tts_worker(self):
        """Streams TTS chunks from text using Pocket TTS."""
        while not self._stop_event.is_set():
            try:
                text, speech_end_t, stt_latency_ms, stt_first_word_ms, first_word = self._tts_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            self.on_status(f"Synthesizing: \"{text}\"")
            t_gen_start = time.perf_counter()
            first_chunk_emitted = False
            chunk_count = 0

            utterance_chunks = []
            try:
                stream_gen = self.tts_model.generate_audio_stream(
                    self.voice_state,
                    text_to_generate=text,
                    copy_state=True,
                )

                for chunk_tensor in stream_gen:
                    chunk_np = chunk_tensor.detach().cpu().numpy()

                    if not first_chunk_emitted:
                        first_chunk_emitted = True
                        now = time.perf_counter()
                        tts_first_bit_ms = (now - t_gen_start) * 1000.0
                        total_first_bit_ms = stt_first_word_ms + tts_first_bit_ms
                        total_glass_to_glass_ms = (now - speech_end_t) * 1000.0

                        self.latest_metrics = {
                            "stt_first_word_ms": round(stt_first_word_ms, 1),
                            "tts_first_bit_ms": round(tts_first_bit_ms, 1),
                            "total_first_bit_ms": round(total_first_bit_ms, 1),
                            "stt_latency_ms": round(stt_latency_ms, 1),
                            "ttfa_latency_ms": round(tts_first_bit_ms, 1),
                            "total_latency_ms": round(total_glass_to_glass_ms, 1),
                            "first_word": first_word,
                        }
                        self.on_latency(self.latest_metrics)

                    self.player.push_audio(chunk_np)
                    chunk_count += 1
                    utterance_chunks.append(chunk_np)
                    with self._recorded_lock:
                        self._recorded_chunks.append(chunk_np.copy())

                # Add brief pause spacing between clauses in the full session buffer
                with self._recorded_lock:
                    self._recorded_chunks.append(np.zeros(int(self.sample_rate * 0.2), dtype=np.float32))

                # Immediately persist completed utterance audio to disk
                if utterance_chunks and self.output_dir:
                    u_audio = np.concatenate(utterance_chunks)
                    peak = float(np.max(np.abs(u_audio)))
                    if peak > 0.001:
                        u_audio = (u_audio / peak) * 0.90
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    u_file = self.output_dir / f"live_utterance_{ts}.wav"
                    try:
                        sf.write(str(u_file), u_audio, self.sample_rate, subtype="PCM_16")
                        dur = len(u_audio) / self.sample_rate
                        self.on_audio_saved(str(u_file), dur)
                    except Exception as e:
                        logger.error(f"Error saving live utterance file: {e}")

            except Exception as e:
                logger.error(f"TTS streaming error: {e}")

            self.on_status("Listening...")
