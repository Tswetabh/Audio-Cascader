"""Minimal local Audio Cascader POC.

The module keeps model imports lazy so pure text helpers and the web shell can
be tested without allocating GPU memory.  Neural inference is CUDA-only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import tempfile
import threading
import queue
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent
SAMPLE_RATE = 16_000
OUTPUT_RATE = 24_000
DEFAULT_STT = "turbo"

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


class RmsEndpointDetector:
    """Small non-neural speech endpoint detector for fixed-size audio blocks."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        block_size: int = 512,
        threshold: float = 0.02,
        silence_ms: int = 300,
        min_speech_ms: int = 200,
        max_segment_ms: int = 3500,
    ):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.threshold = threshold
        self.silence_blocks = max(1, round(silence_ms * sample_rate / block_size / 1000))
        self.min_samples = round(min_speech_ms * sample_rate / 1000)
        self.max_samples = round(max_segment_ms * sample_rate / 1000)
        self._blocks: list[Any] = []
        self._samples = 0
        self._silent = 0

    def _reset(self) -> None:
        self._blocks = []
        self._samples = 0
        self._silent = 0

    def push(self, block: Any) -> list[Any]:
        import numpy as np

        audio = np.asarray(block, dtype=np.float32).reshape(-1)
        if not len(audio):
            return []
        rms = float(np.sqrt(np.mean(np.square(audio))))
        active = rms >= self.threshold
        if active or self._blocks:
            self._blocks.append(audio)
            self._samples += len(audio)
        if active:
            self._silent = 0
        elif self._blocks:
            self._silent += 1
        if not self._blocks:
            return []
        if self._samples >= self.max_samples or self._silent >= self.silence_blocks:
            result = np.concatenate(self._blocks)
            self._reset()
            if len(result) >= self.min_samples:
                return [result]
        return []


def list_audio_devices() -> list[dict[str, Any]]:
    import sounddevice as sd

    return [
        {
            "index": index,
            "name": device["name"],
            "inputs": device["max_input_channels"],
            "outputs": device["max_output_channels"],
        }
        for index, device in enumerate(sd.query_devices())
        if device["max_input_channels"] or device["max_output_channels"]
    ]


class LiveSession:
    """Capture microphone blocks, endpoint them, and play converted segments."""

    def __init__(self):
        self.running = False
        self.input_queue: queue.Queue[Any] = queue.Queue(maxsize=64)
        self.playback: queue.Queue[Any] = queue.Queue(maxsize=16)
        self.dropped_blocks = 0
        self.input_stream: Any = None
        self.output_stream: Any = None
        self.worker: threading.Thread | None = None

    def start(self, input_device: int | None = None, output_device: int | None = None) -> None:
        import sounddevice as sd

        if self.running:
            return
        require_cuda()
        detector = RmsEndpointDetector()

        def on_input(indata: Any, _frames: int, _timing: Any, _status: Any) -> None:
            try:
                self.input_queue.put_nowait(indata[:, 0].copy())
            except queue.Full:
                self.dropped_blocks += 1

        def on_output(outdata: Any, _frames: int, _timing: Any, _status: Any) -> None:
            outdata.fill(0)
            try:
                audio = self.playback.get_nowait()
            except queue.Empty:
                return
            length = min(len(audio), len(outdata))
            outdata[:length, 0] = audio[:length]
            if length < len(audio):
                try:
                    self.playback.put_nowait(audio[length:])
                except queue.Full:
                    pass

        self.running = True
        self.input_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=512,
            channels=1,
            dtype="float32",
            device=input_device,
            callback=on_input,
        )
        self.output_stream = sd.OutputStream(
            samplerate=OUTPUT_RATE,
            blocksize=512,
            channels=1,
            dtype="float32",
            device=output_device,
            callback=on_output,
        )
        self.input_stream.start()
        self.output_stream.start()

        def worker() -> None:
            stt = get_whisper_runner()
            tts = get_pocket_runner()
            while self.running:
                try:
                    block = self.input_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                for segment in detector.push(block):
                    with _GPU_LOCK:
                        result = stt.transcribe(segment)
                        for text in chunk_text(result["text"]):
                            audio = tts.synthesize(text)
                            if len(audio):
                                try:
                                    self.playback.put(audio, timeout=0.5)
                                except queue.Full:
                                    pass

        self.worker = threading.Thread(target=worker, name="audio-cascader-live", daemon=True)
        self.worker.start()

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
        while not self.input_queue.empty():
            self.input_queue.get_nowait()
        while not self.playback.empty():
            self.playback.get_nowait()


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
    container = av.open(str(path))
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
    container.close()
    if not pieces:
        raise ValueError(f"Audio file is empty: {path.name}")
    audio = np.concatenate(pieces).astype(np.float32)
    return _resample(audio, int(stream.sample_rate or SAMPLE_RATE), SAMPLE_RATE)


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
        if self.cache_path.is_file() and self.manifest_path.is_file():
            try:
                manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                if manifest.get("sha256_reference") == digest:
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
    from fastapi.responses import FileResponse

    web_app = FastAPI(title="Audio Cascader")
    session = LiveSession()

    @web_app.get("/")
    def index() -> Any:
        return FileResponse(ROOT / "web" / "index.html")

    @web_app.get("/api/status")
    def status() -> dict[str, Any]:
        try:
            import torch

            gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        except ImportError:
            gpu = None
        return {
            "gpu": gpu,
            "stt_model": DEFAULT_STT,
            "tts_model": "pocket_tts",
            "state": "running" if session.running else "ready",
            "queue_drops": session.dropped_blocks,
        }

    @web_app.get("/api/devices")
    def devices() -> list[dict[str, Any]]:
        try:
            return list_audio_devices()
        except (ImportError, RuntimeError) as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @web_app.post("/api/start")
    def start(input_device: int | None = None, output_device: int | None = None) -> dict[str, Any]:
        try:
            session.start(input_device=input_device, output_device=output_device)
            return {"state": "running"}
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            session.stop()
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @web_app.post("/api/stop")
    def stop() -> dict[str, Any]:
        session.stop()
        return {"state": "ready"}

    @web_app.post("/api/process")
    async def process(upload: UploadFile = File(...)) -> dict[str, Any]:
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
    import uvicorn

    uvicorn.run(create_web_app(), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
