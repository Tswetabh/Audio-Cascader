"""Audio capture, playback, and streaming Voice Activity Detection (VAD) utilities.

Provides low-latency, WASAPI-compatible audio streaming using sounddevice and
real-time utterance segmentation using Silero VAD.
"""

from __future__ import annotations

import collections
import logging
import queue
import threading
import time
from typing import Callable, List, Optional, Tuple

import numpy as np
import sounddevice as sd
import torch

logger = logging.getLogger(__name__)


def list_audio_devices() -> Tuple[List[dict], List[dict]]:
    """Enumerate available input and output devices.

    Returns:
        Tuple of (input_devices, output_devices)
    """
    devices = sd.query_devices()
    default_in = sd.default.device[0]
    default_out = sd.default.device[1]

    input_devs = []
    output_devs = []

    for idx, d in enumerate(devices):
        info = {
            "index": idx,
            "name": d["name"],
            "hostapi": d["hostapi"],
            "max_inputs": d["max_input_channels"],
            "max_outputs": d["max_output_channels"],
            "default_samplerate": d["default_samplerate"],
            "is_default_in": (idx == default_in),
            "is_default_out": (idx == default_out),
        }
        if d["max_input_channels"] > 0:
            input_devs.append(info)
        if d["max_output_channels"] > 0:
            output_devs.append(info)

    return input_devs, output_devs


class AudioCapture:
    """Continuously captures microphone audio in low-latency blocks.

    Captures mono 16kHz float32 audio and pushes chunks to an internal queue
    while reporting real-time RMS volume.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        block_size: int = 512,
        device_index: Optional[int] = None,
        vu_callback: Optional[Callable[[float], None]] = None,
    ):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.device_index = device_index
        self.vu_callback = vu_callback

        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)
        self._stream: Optional[sd.InputStream] = None
        self._running = False

    def _audio_callback(self, indata, frames, time_info, status):
        if status and status.input_overflow:
            logger.warning("Mic input overflow")
        # Mono float32 copy
        chunk = indata[:, 0].copy()
        if self.vu_callback:
            rms = float(np.sqrt(np.mean(chunk**2) + 1e-12))
            self.vu_callback(rms)
        try:
            self.audio_queue.put_nowait(chunk)
        except queue.Full:
            pass

    def start(self):
        if self._running:
            return
        self._running = True
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.block_size,
            device=self.device_index,
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self):
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def get_chunk(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None


class StreamingVADSegmenter:
    """Consumes 512-sample 16kHz audio chunks and groups speech into clauses/utterances.

    Uses Silero VAD to detect speech onset and offset. Emits completed speech segments
    promptly when speech pause exceeds silence threshold or max duration is reached.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 512,
        speech_threshold: float = 0.5,
        min_silence_duration_ms: int = 300,
        min_speech_duration_ms: int = 250,
        max_speech_duration_s: float = 3.5,
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.speech_threshold = speech_threshold
        self.min_silence_chunks = int((min_silence_duration_ms / 1000.0) * sample_rate / chunk_size)
        self.min_speech_chunks = int((min_speech_duration_ms / 1000.0) * sample_rate / chunk_size)
        self.max_speech_chunks = int((max_speech_duration_s * sample_rate) / chunk_size)

        # Lazy load Silero VAD
        from silero_vad import load_silero_vad
        self.model = load_silero_vad()
        self.model.eval()

        # State buffers
        self.current_speech_chunks: List[np.ndarray] = []
        self.silence_chunk_count = 0
        self.in_speech = False
        self.speech_start_time: Optional[float] = None
        self.pre_buffer = collections.deque(maxlen=3)  # Pre-roll ~96ms

    def reset(self):
        self.current_speech_chunks.clear()
        self.silence_chunk_count = 0
        self.in_speech = False
        self.speech_start_time = None
        self.pre_buffer.clear()

    def process_chunk(self, chunk: np.ndarray) -> Optional[Tuple[np.ndarray, float, float]]:
        """Process 512 samples.

        Returns:
            Tuple of (audio_array, start_timestamp, end_timestamp) if an utterance is ready,
            else None.
        """
        tensor_chunk = torch.from_numpy(chunk).float()
        with torch.no_grad():
            prob = self.model(tensor_chunk, self.sample_rate).item()

        now = time.perf_counter()

        if prob >= self.speech_threshold:
            # Voice detected
            if not self.in_speech:
                self.in_speech = True
                self.speech_start_time = now
                # Include pre-buffer for smooth onset
                self.current_speech_chunks.extend(list(self.pre_buffer))
            self.current_speech_chunks.append(chunk)
            self.silence_chunk_count = 0

            # Guard against runaway continuous speech: chunk at max threshold
            if len(self.current_speech_chunks) >= self.max_speech_chunks:
                segment_audio = np.concatenate(self.current_speech_chunks)
                start_t = self.speech_start_time or now
                self.reset()
                return segment_audio, start_t, now

        else:
            # Silence detected
            if self.in_speech:
                self.current_speech_chunks.append(chunk)
                self.silence_chunk_count += 1

                if self.silence_chunk_count >= self.min_silence_chunks:
                    # Clause / utterance boundary reached!
                    if len(self.current_speech_chunks) >= self.min_speech_chunks:
                        segment_audio = np.concatenate(self.current_speech_chunks)
                        start_t = self.speech_start_time or (now - 0.5)
                        self.reset()
                        return segment_audio, start_t, now
                    else:
                        # Too short (noise / click), discard
                        self.reset()
            else:
                self.pre_buffer.append(chunk)

        return None


class StreamingAudioPlayer:
    """Low-latency audio playback engine with jitter buffer.

    Consumes 24kHz float32 audio chunks and streams them smoothly to the
    chosen output sound device without underflow pops or stutter.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        device_index: Optional[int] = None,
        block_size: int = 1024,
        vu_callback: Optional[Callable[[float], None]] = None,
    ):
        self.sample_rate = sample_rate
        self.device_index = device_index
        self.block_size = block_size
        self.vu_callback = vu_callback

        self._lock = threading.Lock()
        self._buffer: collections.deque[np.ndarray] = collections.deque()
        self._current_chunk: Optional[np.ndarray] = None
        self._current_pos: int = 0
        self._stream: Optional[sd.OutputStream] = None
        self._running = False
        self._is_playing = False

    def push_audio(self, audio_data: np.ndarray):
        """Append audio samples (1D float32) to playback queue."""
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        with self._lock:
            self._buffer.append(audio_data)

    def _audio_callback(self, outdata, frames, time_info, status):
        if status and status.output_underflow:
            logger.warning("Playback underflow")

        outdata.fill(0.0)
        needed = frames
        write_idx = 0

        with self._lock:
            while needed > 0:
                if self._current_chunk is None or self._current_pos >= len(self._current_chunk):
                    if len(self._buffer) > 0:
                        self._current_chunk = self._buffer.popleft()
                        self._current_pos = 0
                    else:
                        self._current_chunk = None
                        self._is_playing = False
                        break

                self._is_playing = True
                available = len(self._current_chunk) - self._current_pos
                to_copy = min(needed, available)
                outdata[write_idx : write_idx + to_copy, 0] = self._current_chunk[
                    self._current_pos : self._current_pos + to_copy
                ]

                self._current_pos += to_copy
                write_idx += to_copy
                needed -= to_copy

        if self.vu_callback:
            rms = float(np.sqrt(np.mean(outdata[:, 0] ** 2) + 1e-12))
            self.vu_callback(rms)

    def start(self):
        if self._running:
            return
        self._running = True
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.block_size,
            device=self.device_index,
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self):
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self.clear()

    def clear(self):
        with self._lock:
            self._buffer.clear()
            self._current_chunk = None
            self._current_pos = 0
            self._is_playing = False

    @property
    def is_playing(self) -> bool:
        return self._is_playing or len(self._buffer) > 0
