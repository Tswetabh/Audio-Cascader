"""Minimal local Audio Cascader POC.

The module keeps model imports lazy so pure text helpers and the web shell can
be tested without allocating GPU memory.  Neural inference is CUDA-only.
"""

from __future__ import annotations

import hashlib
import csv
import io
import json
import logging
import os
import re
import shutil
import statistics
import subprocess
import tempfile
import threading
import queue
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

# Module-level import so FastAPI can resolve the stringified `UploadFile`
# annotation in /api/process: `from __future__ import annotations` turns
# annotations into ForwardRefs that FastAPI resolves against module globals.
from fastapi import UploadFile  # noqa: F401

ROOT = Path(__file__).resolve().parent
SAMPLE_RATE = 16_000
OUTPUT_RATE = 24_000
DEFAULT_STT = "turbo"
INCREMENTAL_INTERVAL_S = 0.6
INCREMENTAL_WINDOW_S = 10.0
LIVE_BLOCK_SIZE = 512
PRE_ROLL_MS = 100
LIVE_PLAYBACK_MAX_S = 4.0
LIVE_PLAYBACK_PREFILL_S = 0.15

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("audio_cascader")


def chunk_text(text: str, max_words: int = 12) -> list[str]:
    """Split text at punctuation, then at a word cap without losing punctuation."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    parts = [part.strip() for part in re.findall(r"[^,.?!;:]+[,.?!;:]*", cleaned)]
    result: list[str] = []
    for part in parts:
        words = part.split()
        for start in range(0, len(words), max_words):
            piece = " ".join(words[start : start + max_words])
            if piece:
                result.append(piece)
    return result


def stable_prefix(previous: str, current: str) -> str:
    """Return the unchanged, complete-word prefix shared by two STT passes."""
    previous_words = previous.strip().split()
    current_words = current.strip().split()
    if previous_words and current_words:
        last_index = min(len(previous_words), len(current_words)) - 1
        if previous_words[last_index] != current_words[last_index] and current_words[last_index].startswith(previous_words[last_index]):
            previous_words = previous_words[:-1]
    common: list[str] = []
    for left, right in zip(previous_words, current_words):
        if left != right:
            break
        common.append(left)
    return " ".join(common)


class IncrementalTranscriber:
    """Commits newly stable words across successive transcription passes.

    Feed every partial transcription of the segment to :meth:`update`; it
    returns only words that stayed unchanged across two consecutive passes and
    were not emitted yet. At the endpoint, :meth:`flush` releases the
    remaining tail. :meth:`reset` starts a fresh segment.
    """

    def __init__(self) -> None:
        self._last_pass = ""
        self._emitted_words: list[str] = []

    def update(self, transcription: str) -> str:
        candidate = stable_prefix(self._last_pass, transcription).split()
        self._last_pass = transcription
        emitted = self._emitted_words
        if candidate[: len(emitted)] != emitted:
            return ""
        new_words = candidate[len(emitted) :]
        if not new_words:
            return ""
        self._emitted_words = emitted + new_words
        return " ".join(new_words)

    def flush(self, transcription: str) -> str:
        """Release the remaining tail once the segment has ended."""
        words = transcription.strip().split()
        emitted = self._emitted_words
        if words[: len(emitted)] != emitted:
            return ""
        new_words = words[len(emitted) :]
        if new_words:
            self._emitted_words = emitted + new_words
        return " ".join(new_words)

    def reset(self) -> None:
        self._last_pass = ""
        self._emitted_words = []


class RmsEndpointDetector:
    """Small non-neural speech endpoint detector for fixed-size audio blocks."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        block_size: int = 512,
        threshold: float = 0.02,
        silence_ms: int = 500,
        min_speech_ms: int = 300,
        max_segment_ms: int = 5000,
        calibrate_ms: int = 2000,
    ):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.min_threshold = threshold
        self.threshold = threshold
        self.silence_blocks = max(1, round(silence_ms * sample_rate / block_size / 1000))
        self.min_samples = round(min_speech_ms * sample_rate / 1000)
        self.max_samples = round(max_segment_ms * sample_rate / 1000)
        self._calibrate_blocks = round(calibrate_ms * sample_rate / block_size / 1000)
        self._calibration: list[float] = []
        self._calibrating = self._calibrate_blocks > 0
        self._pre_roll: deque[Any] = deque(maxlen=max(1, round(PRE_ROLL_MS * sample_rate / block_size / 1000)))
        self._blocks: list[Any] = []
        self._samples = 0
        self._silent = 0

    @property
    def calibrated(self) -> bool:
        return not self._calibrating

    @property
    def in_speech(self) -> bool:
        return bool(self._blocks)

    def _reset(self) -> None:
        self._blocks = []
        self._samples = 0
        self._silent = 0
        self._pre_roll.clear()

    def push(self, block: Any) -> list[Any]:
        import numpy as np

        audio = np.asarray(block, dtype=np.float32).reshape(-1)
        if not len(audio):
            return []
        rms = float(np.sqrt(np.mean(np.square(audio))))
        if self._calibrating:
            self._calibration.append(rms)
            if len(self._calibration) >= self._calibrate_blocks:
                floor = float(np.median(self._calibration))
                self.threshold = max(self.min_threshold, floor * 3.0)
                self._calibrating = False
            return []
        active = rms >= self.threshold
        if not self._blocks:
            if not active:
                self._pre_roll.append(audio)
                return []
            seed = list(self._pre_roll)
            self._pre_roll.clear()
            self._blocks = seed + [audio]
            self._samples = sum(len(piece) for piece in seed) + len(audio)
            self._silent = 0
        elif active:
            self._blocks.append(audio)
            self._samples += len(audio)
            self._silent = 0
        else:
            self._blocks.append(audio)
            self._samples += len(audio)
            self._silent += 1
        if self._samples >= self.max_samples or self._silent >= self.silence_blocks:
            result = np.concatenate(self._blocks)
            self._reset()
            if len(result) >= self.min_samples:
                return [result]
        return []


def auto_convert_settings(host_api_name: str) -> Any:
    """WASAPI auto-convert settings when the host API is WASAPI, else None.

    WASAPI shared-mode endpoints only accept their configured mix rate
    (48 kHz here), while our streams run at 16 kHz (mic) and 24 kHz
    (playback). Without auto-convert they fail with "Invalid sample rate".
    """
    if "WASAPI" not in host_api_name:
        return None
    import sounddevice as sd

    return sd.WasapiSettings(auto_convert=True)


def device_extra_settings(device: int | None, kind: str) -> Any:
    """extra_settings needed to open a stream on `device` (None = default)."""
    import sounddevice as sd

    try:
        info: Any = sd.query_devices(kind=kind) if device is None else sd.query_devices(device)
        host = sd.query_hostapis(info["hostapi"])["name"]
    except Exception:
        return None
    return auto_convert_settings(host)


def list_audio_devices() -> list[dict[str, Any]]:
    import sounddevice as sd

    hostapis = sd.query_hostapis()
    return [
        {
            "index": index,
            "name": device["name"],
            "host_api": hostapis[device["hostapi"]]["name"],
            "inputs": device["max_input_channels"],
            "outputs": device["max_output_channels"],
        }
        for index, device in enumerate(sd.query_devices())
        if device["max_input_channels"] or device["max_output_channels"]
    ]


class PlaybackBuffer:
    """FIFO of PCM frames for the output stream with a bounded backlog.

    When the queued audio exceeds ``max_samples``, the oldest frames are
    dropped so playback never runs far behind live speech.
    """

    def __init__(self, max_samples: int, capacity: int = 64):
        self.max_samples = max_samples
        self.samples = 0
        self.dropped_frames = 0
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=capacity)
        self._pending: Any | None = None
        self._lock = threading.Lock()

    def push(self, audio: Any) -> None:
        with self._lock:
            while self.samples and self.samples + len(audio) > self.max_samples:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                self.samples -= len(item)
                self.dropped_frames += 1
            try:
                self._queue.put_nowait(audio)
                self.samples += len(audio)
            except queue.Full:
                self.dropped_frames += 1

    def pop(self) -> Any | None:
        with self._lock:
            if self._pending is not None:
                item = self._pending
                self._pending = None
                self.samples -= len(item)
                return item
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return None
            self.samples -= len(item)
            return item

    def pop_frames(self, count: int) -> Any:
        """Pop up to ``count`` contiguous samples across queued TTS chunks."""
        import numpy as np

        if count <= 0:
            return np.zeros(0, dtype=np.float32)
        parts: list[Any] = []
        remaining = count
        with self._lock:
            while remaining:
                if self._pending is not None:
                    item = self._pending
                    self._pending = None
                else:
                    try:
                        item = self._queue.get_nowait()
                    except queue.Empty:
                        break
                take = min(len(item), remaining)
                if take:
                    parts.append(item[:take])
                    self.samples -= take
                    remaining -= take
                if take < len(item):
                    self._pending = item[take:]
                    break
        return np.concatenate(parts).astype(np.float32, copy=False) if parts else np.zeros(0, dtype=np.float32)

    def clear(self) -> None:
        with self._lock:
            self._pending = None
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self.samples = 0


class EventHub:
    """Fan-out of JSON-serializable events to SSE subscribers."""

    def __init__(self, capacity: int = 256):
        self.capacity = capacity
        self._subscribers: list[queue.Queue[Any]] = []
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue[Any]:
        channel: queue.Queue[Any] = queue.Queue(maxsize=self.capacity)
        with self._lock:
            self._subscribers.append(channel)
        return channel

    def unsubscribe(self, channel: queue.Queue[Any]) -> None:
        with self._lock:
            if channel in self._subscribers:
                self._subscribers.remove(channel)

    def publish(self, event: dict[str, Any]) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for channel in subscribers:
            try:
                channel.put_nowait(event)
            except queue.Full:
                pass


@dataclass
class SegmentDiagnostics:
    """JSON-safe timings and measurements for one live speech segment."""

    segment_id: int
    started_at: float
    ended_at: float
    endpoint_reason: str = ""
    audio_duration_s: float = 0.0
    rms: float = 0.0
    peak: float = 0.0
    stt: dict[str, Any] = field(default_factory=dict)
    tts: dict[str, Any] = field(default_factory=dict)
    playback: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class DiagnosticsSession:
    """In-memory diagnostics for one session; never writes itself to disk."""

    session_id: str
    started_at: float
    ended_at: float | None = None
    config: dict[str, Any] = field(default_factory=dict)
    counters: dict[str, Any] = field(default_factory=dict)
    segments: list[SegmentDiagnostics] = field(default_factory=list)
    system_samples: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        counters = dict(payload["counters"])
        counters["segment_count"] = len(self.segments)
        counters["system_sample_count"] = len(self.system_samples)
        payload["counters"] = counters
        return payload


def summarize_diagnostics(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Return robust latency summaries from a diagnostics snapshot."""
    metrics = ("stt_latency_ms", "tts_first_frame_latency_ms", "end_to_end_latency_ms")
    result: dict[str, Any] = {}
    segments = snapshot.get("segments", [])
    for metric in metrics:
        values = [
            float(segment.get("timings", {}).get(metric))
            for segment in segments
            if segment.get("timings", {}).get(metric) is not None
        ]
        if not values:
            result[metric] = {"count": 0, "average": None, "p50": None, "p95": None}
            continue
        ordered = sorted(values)
        p95 = ordered[0] if len(ordered) == 1 else statistics.quantiles(ordered, n=100, method="inclusive")[94]
        result[metric] = {
            "count": len(values),
            "average": round(statistics.mean(values), 3),
            "p50": round(statistics.median(values), 3),
            "p95": round(p95, 3),
        }
    return result


def sample_system_usage() -> dict[str, Any]:
    """Collect one best-effort process, GPU memory, utilization, and temperature sample."""
    import psutil

    process = psutil.Process(os.getpid())
    sample: dict[str, Any] = {
        "t": time.monotonic(),
        "cpu_percent": psutil.cpu_percent(interval=None),
        "process_rss_mb": round(process.memory_info().rss / (1024 * 1024), 2),
        "gpu_utilization_percent": None,
        "gpu_vram_used_mb": None,
        "gpu_vram_total_mb": None,
        "gpu_temperature_c": None,
    }
    try:
        import torch

        if torch.cuda.is_available():
            sample["gpu_vram_used_mb"] = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2)
            sample["gpu_vram_total_mb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 2)
    except (ImportError, RuntimeError):
        pass
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=0.5,
            check=False,
        )
        fields = [field.strip() for field in result.stdout.split(",", 1)]
        if result.returncode == 0 and len(fields) == 2:
            sample["gpu_utilization_percent"] = float(fields[0])
            sample["gpu_temperature_c"] = float(fields[1])
    except (FileNotFoundError, OSError, subprocess.SubprocessError, ValueError):
        pass
    return sample


class SessionRecorder:
    """Collects live audio frames and text events; writes artifacts on stop."""

    def __init__(self) -> None:
        self.audio: list[Any] = []
        self.events: list[dict[str, Any]] = []
        self.started = time.monotonic()

    def add_audio(self, frame: Any) -> None:
        if len(frame):
            self.audio.append(frame)

    def add_event(self, kind: str, text: str) -> None:
        self.events.append({"t": round(time.monotonic() - self.started, 3), "kind": kind, "text": text})

    def finalize(self, output_dir: Path, sample_rate: int) -> dict[str, Any]:
        import numpy as np
        import soundfile as sf

        output_dir.mkdir(parents=True, exist_ok=True)
        audio = np.concatenate(self.audio) if self.audio else np.zeros(0, dtype=np.float32)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_path = output_dir / f"live_{stamp}.wav"
        metrics_path = output_dir / f"live_{stamp}_metrics.json"
        sf.write(str(wav_path), audio, sample_rate, subtype="PCM_16")
        metrics = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "duration_s": round(len(audio) / sample_rate, 3),
            "sample_rate": sample_rate,
            "events": self.events,
        }
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return {"wav": str(wav_path), "metrics": str(metrics_path), "duration_s": metrics["duration_s"]}


class LiveSession:
    """Capture microphone blocks, endpoint them, and play converted segments."""

    def __init__(self):
        self.running = False
        self.input_queue: queue.Queue[Any] = queue.Queue(maxsize=256)
        self.tts_queue: queue.Queue[tuple[int, str]] = queue.Queue(maxsize=32)
        self.playback = PlaybackBuffer(max_samples=round(LIVE_PLAYBACK_MAX_S * OUTPUT_RATE))
        self.playback_started = False
        self.playback_underruns = 0
        self.dropped_blocks = 0
        self.input_stream: Any = None
        self.output_stream: Any = None
        self.worker: threading.Thread | None = None
        self.tts_worker: threading.Thread | None = None
        self.diagnostics_worker: threading.Thread | None = None
        self.hub = EventHub()
        self.stt_model = DEFAULT_STT
        self.recorder: SessionRecorder | None = None
        self.last_recording: dict[str, Any] | None = None
        self.diagnostics: DiagnosticsSession | None = None
        self._diagnostics_lock = threading.Lock()

    def begin_diagnostics(self, config: dict[str, Any]) -> None:
        with self._diagnostics_lock:
            self.diagnostics = DiagnosticsSession(
                session_id=uuid.uuid4().hex,
                started_at=time.monotonic(),
                config=dict(config),
            )

    def record_segment_diagnostics(self, segment: SegmentDiagnostics) -> None:
        with self._diagnostics_lock:
            if self.diagnostics is not None:
                self.diagnostics.segments.append(segment)

    def update_segment_diagnostics(self, segment_id: int, section: str, values: dict[str, Any]) -> None:
        with self._diagnostics_lock:
            if self.diagnostics is None:
                return
            for segment in self.diagnostics.segments:
                if segment.segment_id == segment_id:
                    target = getattr(segment, section)
                    target.update(values)
                    return

    def diagnostics_time(self) -> float:
        with self._diagnostics_lock:
            if self.diagnostics is None:
                return 0.0
            return time.monotonic() - self.diagnostics.started_at

    def segment_started_at(self, segment_id: int) -> float | None:
        with self._diagnostics_lock:
            if self.diagnostics is None:
                return None
            for segment in self.diagnostics.segments:
                if segment.segment_id == segment_id:
                    return segment.started_at
        return None

    def add_system_sample(self, sample: dict[str, Any]) -> None:
        with self._diagnostics_lock:
            if self.diagnostics is not None:
                self.diagnostics.system_samples.append(dict(sample))

    def diagnostics_snapshot(self) -> dict[str, Any]:
        with self._diagnostics_lock:
            if self.diagnostics is None:
                snapshot = DiagnosticsSession(session_id="", started_at=0.0).to_dict()
            else:
                snapshot = self.diagnostics.to_dict()
        snapshot["counters"].update(
            {
                "input_queue_drops": self.dropped_blocks,
                "tts_queue_depth": self.tts_queue.qsize(),
                "playback_backlog_s": round(self.playback.samples / OUTPUT_RATE, 3),
                "playback_drops": self.playback.dropped_frames,
                "playback_underruns": self.playback_underruns,
            }
        )
        snapshot["summary"] = summarize_diagnostics(snapshot)
        return snapshot

    def start(
        self,
        input_device: int | None = None,
        output_device: int | None = None,
        stt_model: str = DEFAULT_STT,
    ) -> None:
        import sounddevice as sd

        if self.running:
            return
        require_cuda()
        detector = RmsEndpointDetector()
        self.stt_model = stt_model
        self.recorder = SessionRecorder()
        self.last_recording = None
        self.playback_started = False
        self.playback_underruns = 0
        self.begin_diagnostics(
            {
                "input_device": input_device,
                "output_device": output_device,
                "stt_model": stt_model,
                "input_sample_rate": SAMPLE_RATE,
                "output_sample_rate": OUTPUT_RATE,
                "input_block_size": LIVE_BLOCK_SIZE,
                "output_block_size": 512,
            }
        )

        def on_input(indata: Any, _frames: int, _timing: Any, _status: Any) -> None:
            try:
                self.input_queue.put_nowait(indata[:, 0].copy())
            except queue.Full:
                self.dropped_blocks += 1

        def on_output(outdata: Any, _frames: int, _timing: Any, _status: Any) -> None:
            outdata.fill(0)
            if not self.playback_started:
                if self.playback.samples < round(LIVE_PLAYBACK_PREFILL_S * OUTPUT_RATE):
                    return
                self.playback_started = True
            audio = self.playback.pop_frames(len(outdata))
            if len(audio):
                outdata[: len(audio), 0] = audio
            if len(audio) < len(outdata):
                self.playback_underruns += 1
                self.playback_started = False

        self.running = True
        # Load both models before opening streams: detector calibration then
        # measures real ambience, and the first words spoken are never dropped
        # while the models warm up. The silent warm-ups pay the one-time CUDA
        # autotune (~8 s) and the first TTS decoder pass (~3 s) at Start
        # instead of on the user's first sentence.
        import numpy as np

        get_whisper_runner(stt_model)
        get_pocket_runner()
        get_whisper_runner(stt_model).transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))
        for _ in get_pocket_runner().stream("Hi."):
            pass
        self.input_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=LIVE_BLOCK_SIZE,
            channels=1,
            dtype="float32",
            device=input_device,
            extra_settings=device_extra_settings(input_device, "input"),
            callback=on_input,
        )
        self.output_stream = sd.OutputStream(
            samplerate=OUTPUT_RATE,
            blocksize=512,
            channels=1,
            dtype="float32",
            device=output_device,
            extra_settings=device_extra_settings(output_device, "output"),
            callback=on_output,
        )
        self.input_stream.start()
        self.output_stream.start()

        def diagnostics_worker() -> None:
            while self.running:
                self.add_system_sample(sample_system_usage())
                time.sleep(1.0)

        self.diagnostics_worker = threading.Thread(
            target=diagnostics_worker, name="audio-cascader-diagnostics", daemon=True
        )
        self.diagnostics_worker.start()

        def tts_worker() -> None:
            tts = get_pocket_runner()
            while self.running:
                try:
                    segment_id, piece = self.tts_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if not piece:
                    continue
                self.hub.publish({"kind": "converted", "text": piece})
                if self.recorder is not None:
                    self.recorder.add_event("converted", piece)
                try:
                    queued_at = self.diagnostics_time()
                    tts_started = time.monotonic()
                    tts_started_at = self.diagnostics_time()
                    first_frame_at: float | None = None
                    generated_samples = 0
                    # TTS owns a separate model from STT. Keeping it in its
                    # own worker prevents synthesis from stopping microphone
                    # capture and dropping input blocks.
                    for audio in tts.stream(piece):
                        if len(audio):
                            if first_frame_at is None:
                                first_frame_at = self.diagnostics_time()
                            self.playback.push(audio)
                            generated_samples += len(audio)
                            if self.recorder is not None:
                                self.recorder.add_audio(audio)
                    finished_at = self.diagnostics_time()
                    generation_ms = (time.monotonic() - tts_started) * 1000.0
                    self.update_segment_diagnostics(
                        segment_id,
                        "tts",
                        {
                            "piece": piece,
                            "queued_at": queued_at,
                            "started_at": tts_started_at,
                            "first_frame_at": first_frame_at,
                            "finished_at": finished_at,
                            "generated_audio_duration_s": generated_samples / OUTPUT_RATE,
                            "generation_ms": round(generation_ms, 3),
                            "queue_wait_ms": round((tts_started_at - queued_at) * 1000.0, 3),
                            "real_time_factor": round(
                                (generated_samples / OUTPUT_RATE) / max(generation_ms / 1000.0, 1e-9), 3
                            ),
                        },
                    )
                    self.update_segment_diagnostics(
                        segment_id,
                        "timings",
                        {
                            "tts_first_frame_latency_ms": round(
                                ((first_frame_at or finished_at) - queued_at) * 1000.0, 3
                            ),
                            "end_to_end_latency_ms": round(
                                (finished_at - (self.segment_started_at(segment_id) or finished_at)) * 1000.0,
                                3,
                            ),
                        },
                    )
                except Exception as exc:
                    log.exception("TTS worker failed")
                    self.hub.publish({"kind": "error", "error": str(exc)})
                    self.running = False

        self.tts_worker = threading.Thread(target=tts_worker, name="audio-cascader-tts", daemon=True)
        self.tts_worker.start()

        def worker() -> None:
            import numpy as np

            stt = get_whisper_runner(stt_model)
            tracker = IncrementalTranscriber()
            pending: deque[Any] = deque(maxlen=max(1, round(INCREMENTAL_WINDOW_S * SAMPLE_RATE / LIVE_BLOCK_SIZE)))
            last_inference = 0.0

            def emit(text: str, segment_id: int) -> None:
                for piece in chunk_text(text):
                    try:
                        self.tts_queue.put_nowait((segment_id, piece))
                    except queue.Full:
                        log.warning("TTS queue full; dropping converted chunk: %s", piece)

            try:
                while self.running:
                    try:
                        block = self.input_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    finished = detector.push(block)
                    if detector.in_speech:
                        pending.append(block)
                    if finished:
                        for segment in finished:
                            segment_id = len(self.diagnostics.segments) + 1 if self.diagnostics else 1
                            segment_started = self.diagnostics_time() - len(segment) / SAMPLE_RATE
                            segment_ended = self.diagnostics_time()
                            rms = float(np.sqrt(np.mean(np.square(segment)))) if len(segment) else 0.0
                            peak = float(abs(segment).max()) if len(segment) else 0.0
                            self.record_segment_diagnostics(
                                SegmentDiagnostics(
                                    segment_id=segment_id,
                                    started_at=segment_started,
                                    ended_at=segment_ended,
                                    endpoint_reason="silence_or_max",
                                    audio_duration_s=len(segment) / SAMPLE_RATE,
                                    rms=rms,
                                    peak=peak,
                                )
                            )
                            log.info("Segment %.2f s, level %.3f", len(segment) / SAMPLE_RATE, peak)
                            stt_started = time.monotonic()
                            with _GPU_LOCK:
                                final_text = stt.transcribe(segment)["text"]
                            stt_finished = time.monotonic()
                            stt_latency_ms = (stt_finished - stt_started) * 1000.0
                            self.update_segment_diagnostics(
                                segment_id,
                                "stt",
                                {
                                    "model": stt_model,
                                    "text": final_text,
                                    "window_duration_s": len(segment) / SAMPLE_RATE,
                                    "latency_ms": round(stt_latency_ms, 3),
                                },
                            )
                            stt_finished_at = self.diagnostics_time()
                            self.update_segment_diagnostics(
                                segment_id,
                                "timings",
                                {
                                    "stt_latency_ms": round(stt_latency_ms, 3),
                                    "end_to_end_latency_ms": round(
                                        (stt_finished_at - segment_started) * 1000.0, 3
                                    ),
                                },
                            )
                            self.hub.publish({"kind": "spoken", "text": final_text})
                            if self.recorder is not None:
                                self.recorder.add_event("spoken", final_text)
                            emit(tracker.flush(final_text), segment_id)
                        tracker.reset()
                        pending.clear()
                        last_inference = 0.0
                        continue
                    now = time.monotonic()
                    if not detector.in_speech or not detector.calibrated:
                        continue
                    if len(pending) * LIVE_BLOCK_SIZE < SAMPLE_RATE * 0.6 or now - last_inference < INCREMENTAL_INTERVAL_S:
                        continue
                    window = np.concatenate(tuple(pending)) if pending else np.zeros(0, dtype=np.float32)
                    stt_started = time.monotonic()
                    with _GPU_LOCK:
                        text = stt.transcribe(window)["text"]
                        new_text = tracker.update(text)
                    if new_text.strip():
                        emit(new_text, len(self.diagnostics.segments) if self.diagnostics else 1)
                    # Stamp after the call: a slow inference must not reopen the
                    # interval gate immediately and thrash on tiny windows.
                    last_inference = time.monotonic()
            except Exception as exc:
                log.exception("Live worker failed")
                self.hub.publish({"kind": "error", "error": str(exc)})
                self.running = False

        self.worker = threading.Thread(target=worker, name="audio-cascader-live", daemon=True)
        self.worker.start()
        self.hub.publish({"kind": "state", "state": "running", "stt_model": stt_model})

    def stop(self) -> None:
        self.running = False
        if self.input_stream is not None:
            self.input_stream.stop()
            self.input_stream.close()
        if self.output_stream is not None:
            self.output_stream.stop()
            self.output_stream.close()
        self.input_stream = None
        self.output_stream = None
        if self.worker is not None:
            self.worker.join(timeout=2.0)
            self.worker = None
        if self.tts_worker is not None:
            self.tts_worker.join(timeout=2.0)
            self.tts_worker = None
        if self.diagnostics_worker is not None:
            self.diagnostics_worker.join(timeout=2.0)
            self.diagnostics_worker = None
        if self.recorder is not None:
            self.last_recording = self.recorder.finalize(ROOT / "outputs", OUTPUT_RATE)
            self.recorder = None
        while not self.input_queue.empty():
            self.input_queue.get_nowait()
        while not self.tts_queue.empty():
            self.tts_queue.get_nowait()
        self.playback.clear()
        self.playback_started = False
        with self._diagnostics_lock:
            if self.diagnostics is not None:
                self.diagnostics.ended_at = time.monotonic()
        self.hub.publish({"kind": "state", "state": "ready", "recording": self.last_recording})


def require_cuda() -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required: no CUDA-capable GPU is available")
    return torch.cuda.get_device_name(0)


def _resample(audio: Any, source_rate: int, target_rate: int) -> Any:
    import numpy as np

    audio = np.asarray(audio, dtype=np.float32)
    if source_rate == target_rate:
        return audio
    target_length = max(1, round(len(audio) * target_rate / source_rate))
    old_x = np.linspace(0.0, 1.0, len(audio), endpoint=False)
    new_x = np.linspace(0.0, 1.0, target_length, endpoint=False)
    return np.interp(new_x, old_x, audio).astype(np.float32)


def decode_audio(path: Path) -> Any:
    """Decode WAV or compressed audio to mono float32 at 16 kHz."""
    import av
    import numpy as np

    if not path.is_file():
        raise FileNotFoundError(f"Input audio not found: {path}")
    try:
        container = av.open(str(path))
    except av.error.FFmpegError as exc:
        raise ValueError(f"Could not decode {path.name}: {exc}") from exc
    try:
        stream = next((item for item in container.streams if item.type == "audio"), None)
        if stream is None:
            raise ValueError(f"No audio stream found in {path.name}")
        pieces: list[Any] = []
        for frame in container.decode(stream):
            array = frame.to_ndarray()
            if np.issubdtype(array.dtype, np.integer):
                array = array.astype(np.float32) / (float(np.iinfo(array.dtype).max) + 1.0)
            else:
                array = array.astype(np.float32)
            channels = frame.layout.nb_channels
            if frame.format.is_planar:
                mono = array.mean(axis=0) if channels > 1 else array.reshape(-1)
            else:
                mono = array.reshape(-1, channels).mean(axis=1) if channels > 1 else array.reshape(-1)
            pieces.append(mono)
        if not pieces:
            raise ValueError(f"Audio file is empty: {path.name}")
        audio = np.concatenate(pieces).astype(np.float32)
    except av.error.FFmpegError as exc:
        raise ValueError(f"Could not decode {path.name}: {exc}") from exc
    finally:
        container.close()
    return _resample(audio, int(stream.sample_rate or SAMPLE_RATE), SAMPLE_RATE)


def voice_cache_is_current(cache_path: Path, manifest_path: Path, digest: str) -> bool:
    """True when the cached voice state was built from the same reference audio."""
    if not cache_path.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return manifest.get("sha256_reference") == digest


def align_and_score(source: Any, recorded: Any) -> dict[str, Any]:
    """Locate `source` inside `recorded` via FFT cross-correlation.

    Returns the delay of `recorded` relative to `source` in samples
    (positive = recorded is late) and the normalized peak correlation.
    Used by the VB-CABLE loop-back check.
    """
    import numpy as np

    src = np.asarray(source, dtype=np.float64)
    rec = np.asarray(recorded, dtype=np.float64)
    if src.size == 0 or rec.size == 0:
        raise ValueError("align_and_score needs non-empty signals")
    src = src - src.mean()
    rec = rec - rec.mean()
    length = src.size + rec.size - 1
    fft_size = 1 << max(1, (length - 1).bit_length())
    corr = np.fft.irfft(np.fft.rfft(rec, fft_size) * np.conj(np.fft.rfft(src, fft_size)), fft_size)
    peak = int(np.argmax(np.abs(corr)))
    # Negative lags wrap to the tail of the circular correlation: indices
    # [fft_size - (len(rec) - 1), fft_size) hold lags -(len(rec) - 1) .. -1.
    if peak >= fft_size - (rec.size - 1):
        peak -= fft_size
    denom = float(np.sqrt(np.sum(src * src) * np.sum(rec * rec)))
    peak_value = float(corr[peak])
    return {
        "lag_samples": peak,
        "peak_correlation": peak_value / denom if denom else 0.0,
    }


def cable_check_verdict(peak_correlation: float, recorded_rms: float, threshold: float = 0.75) -> str:
    """Classify the outcome of a VB-CABLE loop-back check."""
    if recorded_rms <= 1e-5:
        return "silent"
    return "pass" if peak_correlation >= threshold else "mismatch"


class WhisperRunner:
    def __init__(self, model_name: str = DEFAULT_STT):
        self.model_name = model_name
        self.model: Any = None

    @property
    def model_path(self) -> Path:
        if self.model_name == "tiny":
            return ROOT / "weights" / "faster-whisper-tiny"
        return ROOT / "weights" / "faster-whisper-turbo"

    def load(self) -> Any:
        if self.model is not None:
            return self.model
        require_cuda()
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"Faster-Whisper weights not found at {self.model_path}")
        from faster_whisper import WhisperModel

        self.model = WhisperModel(str(self.model_path), device="cuda", compute_type="int8_float16")
        return self.model

    def transcribe(self, audio: Any) -> dict[str, Any]:
        started = time.perf_counter()
        segments, info = self.load().transcribe(
            audio,
            language="en",
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        rows = [{"start": segment.start, "end": segment.end, "text": segment.text.strip()} for segment in segments]
        return {
            "text": " ".join(row["text"] for row in rows if row["text"]).strip(),
            "segments": rows,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "language": getattr(info, "language", "en"),
        }


class PocketRunner:
    def __init__(self):
        self.model: Any = None
        self.voice_state: Any = None

    @property
    def reference_path(self) -> Path:
        return ROOT / "voices" / "american_accent.wav"

    @property
    def cache_path(self) -> Path:
        return ROOT / "voices" / "american_accent.pt"

    @property
    def manifest_path(self) -> Path:
        return ROOT / "voices" / "manifest.json"

    def load(self) -> Any:
        if self.model is not None:
            return self.model
        import torch

        require_cuda()
        if not self.reference_path.is_file():
            raise FileNotFoundError(f"Voice reference not found at {self.reference_path}")
        from pocket_tts.models.tts_model import TTSModel

        self.model = TTSModel.load_model(device="cuda")
        self.voice_state = self._load_or_create_state(torch)
        return self.model

    def _load_or_create_state(self, torch: Any) -> Any:
        digest = hashlib.sha256(self.reference_path.read_bytes()).hexdigest()
        if voice_cache_is_current(self.cache_path, self.manifest_path, digest):
            try:
                return torch.load(str(self.cache_path), map_location="cuda", weights_only=False)
            except (OSError, ValueError, RuntimeError, EOFError):
                log.warning("Voice cache is stale or corrupt; recomputing it")
        state = self.model.get_state_for_audio_prompt(str(self.reference_path))
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, str(self.cache_path))
        self.manifest_path.write_text(
            json.dumps(
                {
                    "voice": "american_accent",
                    "reference_wav": str(self.reference_path.relative_to(ROOT)),
                    "state_pt": str(self.cache_path.relative_to(ROOT)),
                    "sha256_reference": digest,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "pocket_config": "b6369a24",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return state

    def synthesize(self, text: str) -> Any:
        import numpy as np

        if not text.strip():
            return np.zeros(0, dtype=np.float32)
        started = time.perf_counter()
        chunks = self.load().generate_audio(self.voice_state, text_to_generate=text, copy_state=True)
        audio = chunks.detach().cpu().numpy().astype(np.float32)
        log.info("TTS %.1f ms: %s", (time.perf_counter() - started) * 1000, text)
        return audio

    def stream(self, text: str) -> Iterator[Any]:
        """Yield generated PCM chunks as soon as the model decodes them."""
        if not text.strip():
            return
        started = time.perf_counter()
        first = True
        for chunk in self.load().generate_audio_stream(self.voice_state, text_to_generate=text, copy_state=True):
            audio = chunk.detach().cpu().numpy()
            if first:
                log.info("TTS first frame %.1f ms: %s", (time.perf_counter() - started) * 1000, text)
                first = False
            yield audio


_GPU_LOCK = threading.RLock()
_stt_runners: dict[str, WhisperRunner] = {}
_tts_runner: PocketRunner | None = None


def get_whisper_runner(model_name: str = DEFAULT_STT) -> WhisperRunner:
    """Return the shared Whisper runner for a model name (models load lazily)."""
    with _GPU_LOCK:
        if model_name not in _stt_runners:
            _stt_runners[model_name] = WhisperRunner(model_name)
        return _stt_runners[model_name]


def get_pocket_runner() -> PocketRunner:
    """Return the shared Pocket TTS runner (model loads lazily)."""
    global _tts_runner
    with _GPU_LOCK:
        if _tts_runner is None:
            _tts_runner = PocketRunner()
        return _tts_runner


def process_file(input_path: Path, output_dir: Path | None = None, model_name: str = DEFAULT_STT) -> dict[str, Any]:
    """Run one file through STT → chunking → TTS and save reproducible artifacts."""
    import numpy as np
    import soundfile as sf

    output_dir = output_dir or ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    audio = decode_audio(input_path)
    with _GPU_LOCK:
        transcript = get_whisper_runner(model_name).transcribe(audio)
        texts = chunk_text(transcript["text"])
        tts = get_pocket_runner()
        rendered = [tts.synthesize(text) for text in texts]
    output = np.concatenate(rendered) if rendered else np.zeros(0, dtype=np.float32)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"converted_{stamp}.wav"
    metrics_path = output_dir / f"converted_{stamp}_metrics.json"
    sf.write(str(output_path), output, OUTPUT_RATE, subtype="PCM_16")
    metrics = {
        "gpu": require_cuda(),
        "input": str(input_path),
        "transcript": transcript,
        "converted_text": texts,
        "output": str(output_path),
        "output_duration_s": round(len(output) / OUTPUT_RATE, 3),
        "end_to_end_latency_ms": round((time.perf_counter() - started) * 1000, 1),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {"output": output_path, "metrics": metrics_path, **metrics}


def create_web_app() -> Any:
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import FileResponse, PlainTextResponse, Response
    from fastapi.staticfiles import StaticFiles

    web_app = FastAPI(title="Audio Cascader")
    session = LiveSession()

    frontend_dist = ROOT / "web" / "dist"
    if frontend_dist.is_dir() and (frontend_dist / "index.html").is_file():
        web_app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")

        @web_app.get("/")
        def index() -> Any:
            return FileResponse(frontend_dist / "index.html")
    else:

        @web_app.get("/")
        def index() -> Any:
            return PlainTextResponse(
                "React frontend is not built. Run `npm install` and `npm run build` in web/.",
                status_code=503,
            )

    @web_app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @web_app.get("/api/output")
    def output_file(path: str) -> Any:
        output_root = (ROOT / "outputs").resolve()
        candidate = Path(path).expanduser().resolve()
        if output_root not in candidate.parents or not candidate.is_file():
            raise HTTPException(status_code=404, detail="Output file not found")
        return FileResponse(candidate, media_type="audio/wav", filename=candidate.name)

    @web_app.get("/api/voice-reference")
    def voice_reference() -> Any:
        reference = PocketRunner().reference_path
        if not reference.is_file():
            raise HTTPException(status_code=404, detail="Reference voice not found")
        return FileResponse(reference, media_type="audio/wav", filename=reference.name)

    @web_app.post("/api/synthesize")
    def synthesize(payload: dict[str, Any]) -> dict[str, Any]:
        import numpy as np
        import soundfile as sf

        text = str(payload.get("text", "")).strip()
        if not text:
            raise HTTPException(status_code=400, detail="text is required")
        started = time.perf_counter()
        with _GPU_LOCK:
            audio = get_pocket_runner().synthesize(text)
        output_dir = ROOT / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"synthesized_{stamp}.wav"
        sf.write(str(output_path), np.asarray(audio, dtype=np.float32), OUTPUT_RATE, subtype="PCM_16")
        return {
            "output": str(output_path),
            "text": text,
            "duration_s": round(len(audio) / OUTPUT_RATE, 3),
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 1),
        }

    @web_app.get("/api/status")
    def status() -> dict[str, Any]:
        try:
            import torch

            gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        except ImportError:
            gpu = None
        return {
            "gpu": gpu,
            "stt_model": session.stt_model,
            "tts_model": "pocket_tts",
            "state": "running" if session.running else "ready",
            "queue_drops": session.dropped_blocks,
            "playback_backlog_s": round(session.playback.samples / OUTPUT_RATE, 2),
            "playback_drops": session.playback.dropped_frames,
            "playback_underruns": session.playback_underruns,
            "recording": session.last_recording,
        }

    @web_app.get("/api/diagnostics")
    def diagnostics() -> dict[str, Any]:
        return session.diagnostics_snapshot()

    @web_app.get("/api/diagnostics/export")
    def export_diagnostics(format: str = "json") -> Response:
        snapshot = session.diagnostics_snapshot()
        if format == "json":
            return Response(
                content=json.dumps(snapshot, indent=2),
                media_type="application/json",
                headers={"Content-Disposition": "attachment; filename=audio-cascader-diagnostics.json"},
            )
        if format == "csv":
            fields = [
                "segment_id",
                "started_at",
                "ended_at",
                "audio_duration_s",
                "rms",
                "peak",
                "stt_text",
                "stt_latency_ms",
                "tts_queue_wait_ms",
                "tts_first_frame_latency_ms",
                "tts_generation_ms",
                "tts_real_time_factor",
                "end_to_end_latency_ms",
                "error",
            ]
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for segment in snapshot["segments"]:
                writer.writerow(
                    {
                        "segment_id": segment["segment_id"],
                        "started_at": segment["started_at"],
                        "ended_at": segment["ended_at"],
                        "audio_duration_s": segment["audio_duration_s"],
                        "rms": segment["rms"],
                        "peak": segment["peak"],
                        "stt_text": segment.get("stt", {}).get("text", ""),
                        "stt_latency_ms": segment.get("stt", {}).get("latency_ms", ""),
                        "tts_queue_wait_ms": segment.get("tts", {}).get("queue_wait_ms", ""),
                        "tts_first_frame_latency_ms": segment.get("timings", {}).get(
                            "tts_first_frame_latency_ms", ""
                        ),
                        "tts_generation_ms": segment.get("tts", {}).get("generation_ms", ""),
                        "tts_real_time_factor": segment.get("tts", {}).get("real_time_factor", ""),
                        "end_to_end_latency_ms": segment.get("timings", {}).get("end_to_end_latency_ms", ""),
                        "error": segment.get("error", "") or "",
                    }
                )
            return Response(
                content=output.getvalue(),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=audio-cascader-diagnostics.csv"},
            )
        raise HTTPException(status_code=400, detail="format must be json or csv")

    @web_app.get("/api/devices")
    def devices() -> list[dict[str, Any]]:
        try:
            return list_audio_devices()
        except (ImportError, RuntimeError) as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @web_app.post("/api/start")
    def start(
        input_device: int | None = None,
        output_device: int | None = None,
        stt_model: str = DEFAULT_STT,
    ) -> dict[str, Any]:
        if stt_model not in ("turbo", "tiny"):
            raise HTTPException(status_code=400, detail=f"Unknown STT model: {stt_model}")
        try:
            session.start(input_device=input_device, output_device=output_device, stt_model=stt_model)
            return {"state": "running", "stt_model": stt_model}
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            session.stop()
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @web_app.post("/api/stop")
    def stop() -> dict[str, Any]:
        session.stop()
        return {"state": "ready", "recording": session.last_recording}

    @web_app.get("/api/events")
    def events() -> Any:
        from fastapi.responses import StreamingResponse

        def generate() -> Iterator[str]:
            channel = session.hub.subscribe()
            try:
                while True:
                    try:
                        event = channel.get(timeout=15)
                    except queue.Empty:
                        yield ": keepalive\n\n"
                        continue
                    yield f"data: {json.dumps(event)}\n\n"
            finally:
                session.hub.unsubscribe(channel)

        return StreamingResponse(generate(), media_type="text/event-stream")

    @web_app.post("/api/process")
    def process(upload: UploadFile = File(...)) -> dict[str, Any]:
        # Sync (not async) on purpose: FastAPI runs it in the threadpool, so
        # status polling and the SSE stream keep working during conversion.
        suffix = Path(upload.filename or "input.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
            shutil.copyfileobj(upload.file, temporary)
            temporary_path = Path(temporary.name)
        try:
            result = process_file(temporary_path)
            return {key: str(value) if isinstance(value, Path) else value for key, value in result.items()}
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            temporary_path.unlink(missing_ok=True)

    return web_app


def main() -> None:
    import socket

    import uvicorn

    # One server only: on Windows SO_REUSEADDR lets a second `python app.py`
    # bind port 8000 as well, and two servers share the same GPU (seen
    # 2026-10-17: two Pocket TTS loads, ~0.14x real-time, interleaved logs).
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.3)
        if probe.connect_ex(("127.0.0.1", 8000)) == 0:
            raise SystemExit("Audio Cascader is already running on http://127.0.0.1:8000")
    uvicorn.run(create_web_app(), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
