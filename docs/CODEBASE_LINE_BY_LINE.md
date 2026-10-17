# Audio Cascader — Comprehensive Line-by-Line Code Walkthrough

> **Stale (2026-10-17): pre-simplification walkthrough.** Describes the
> Gen-1/Gen-2 hybrid code that has been superseded by `app.py` +
> `web/index.html`. Kept for reference only.

This document breaks down every script and module in the project line by line. It avoids abstract jargon and focuses directly on what the code does, why each statement is written, and how data moves through the pipeline.

---

## Table of Contents
1. [`run_gui.bat`](#1-run_guibat)
2. [`core/chunker.py`](#2-corechunkerpy)
3. [`core/audio_convert.py`](#3-coreaudio_convertpy)
4. [`core/stt_engine.py`](#4-corestt_enginepy)
5. [`core/tts_engine.py`](#5-coretts_enginepy)
6. [`core/audio_stream.py`](#6-coreaudio_streampy)
7. [`pipeline.py`](#7-pipelinepy)
8. [`gui.py`](#8-guipy)
9. [`pocket_tts/` Overview](#9-pocket_tts-overview)

---

## 1. `run_gui.bat`

This is the Windows batch script used to launch the desktop application with a single double-click.

```bat
1: @echo off
2: title Audio Cascader - S2S Voice Converter
3: cd /d "%~dp0"
4: set PYTHONPATH=%~dp0
5: echo ==============================================================
6: echo Launching Audio Cascader (Indian to American S2S Converter)
7: echo ==============================================================
8: ".venv\Scripts\python.exe" gui.py
9: pause
```

### Line-by-Line Explanation:
* **Line 1 (`@echo off`)**: Stops the Windows Command Prompt from printing every single command to the terminal before executing it. Only explicit `echo` messages will appear.
* **Line 2 (`title Audio Cascader - S2S Voice Converter`)**: Sets the title text displayed on the top border of the Windows Command Prompt window.
* **Line 3 (`cd /d "%~dp0"`)**: Changes the current directory (`cd`) across drive letters (`/d`) to the exact directory where this batch file lives (`%~dp0`). This guarantees all relative paths (like `weights/` and `core/`) resolve properly no matter where you double-click from.
* **Line 4 (`set PYTHONPATH=%~dp0`)**: Informs Python that this project directory is a root import location. This allows imports like `from core.chunker import chunk_text` to work without `ModuleNotFoundError`.
* **Lines 5–7**: Prints a user-friendly greeting box in the terminal window.
* **Line 8 (`".venv\Scripts\python.exe" gui.py`)**: Runs Python using the dedicated virtual environment interpreter (`.venv`) and passes `gui.py` to run. It does **not** use your global Windows Python, ensuring all installed CUDA and PyTorch libraries match perfectly.
* **Line 9 (`pause`)**: If `gui.py` crashes or exits, this prevents the terminal window from instantly closing, allowing you to read any error traceback messages.

---

## 2. `core/chunker.py`

This script splits long transcribed sentences into smaller clauses based on punctuation and word limits. This allows the system to begin synthesizing audio on early words instead of waiting for an entire speech to finish.

```python
1: """Streaming punctuation & pause chunker for S2S pipeline.
...
5: """
6: 
7: from __future__ import annotations
8: 
9: import re
10: from typing import List, Sequence
11: 
12: DEFAULT_DELIMITERS = (",", ".", "?", "!", ";", ":")
```

### Lines 1–12 Explanation:
* **Line 7 (`from __future__ import annotations`)**: Enables modern type hinting syntax in Python without evaluating type annotations at runtime.
* **Line 9 (`import re`)**: Imports Python's built-in Regular Expressions engine, used to search and match punctuation patterns inside text.
* **Line 10 (`from typing import List, Sequence`)**: Imports type hint helpers (`List` for lists, `Sequence` for tuples/lists).
* **Line 12 (`DEFAULT_DELIMITERS = ...`)**: A tuple containing the punctuation marks where a speaker naturally pauses or breathes.

```python
15: def chunk_text(
16:     text: str,
17:     delimiters: Sequence[str] = DEFAULT_DELIMITERS,
18:     max_words: int = 12,
19: ) -> List[str]:
20:     """Split text on punctuation marks or word counts while keeping punctuation."""
21:     cleaned = text.strip()
22:     if not cleaned:
23:         return []
```

### Lines 15–23 Explanation:
* **Lines 15–19**: Defines the `chunk_text` function. It takes an input string `text`, optional delimiters, and `max_words` (default 12 words per chunk), returning a list of strings (`List[str]`).
* **Line 21 (`cleaned = text.strip()`)**: Removes leading and trailing whitespace or newline characters from the input text.
* **Lines 22–23**: If the text is empty after stripping, returns an empty list immediately.

```python
34:     escaped_delims = "".join(re.escape(d) for d in delimiters)
35:     pattern = rf"[^{escaped_delims}]+[{escaped_delims}]*"
36:     chunks = [m.group(0).strip() for m in re.finditer(pattern, cleaned) if m.group(0).strip()]
```

### Lines 34–36 Explanation:
* **Line 34 (`re.escape(d)`)**: Escapes special regex characters (like `?` and `.`, which mean "wildcard" in regex) so they are treated as literal punctuation characters.
* **Line 35 (`pattern = rf"[^{escaped_delims}]+[{escaped_delims}]*"`)**:
  - `[^{escaped_delims}]+`: Match one or more characters that are NOT punctuation (the words).
  - `[{escaped_delims}]*`: Match any trailing punctuation marks that belong to that clause.
* **Line 36 (`re.finditer`)**: Iterates through all matches in the text, pulls each string match (`m.group(0)`), strips surrounding spaces, and stores them in `chunks`.

```python
38:     final_chunks: List[str] = []
39:     for chunk in chunks:
40:         words = chunk.split()
41:         if len(words) <= max_words:
42:             final_chunks.append(chunk)
43:         else:
44:             for i in range(0, len(words), max_words):
45:                 sub = " ".join(words[i : i + max_words])
46:                 if i + max_words < len(words) and sub[-1] not in delimiters:
47:                     sub += ","
48:                 final_chunks.append(sub)
49: 
50:     return [c.strip() for c in final_chunks if c.strip()]
```

### Lines 38–50 Explanation:
* **Line 38**: Initializes `final_chunks` as an empty list.
* **Line 39**: Loops through each punctuation-delimited chunk.
* **Line 40**: Splits the chunk into individual words by spaces (`chunk.split()`).
* **Lines 41–42**: If the clause has fewer than or equal to `max_words` (e.g. 12 words), it is already small enough, so it is appended to `final_chunks`.
* **Lines 43–48**: If someone spoke a very long run-on sentence with no commas (e.g. 30 words):
  - Line 44 slices the list in increments of `max_words` (0 to 12, 12 to 24, etc.).
  - Line 45 joins those words back into a smaller phrase.
  - Line 46–47: If this is an intermediate slice and doesn't already end in punctuation, it adds a comma `","` so the TTS engine creates a natural speaking pause.
  - Line 48: Appends the sub-clause.
* **Line 50**: Cleans up each chunk and returns the final list.

---

## 3. `core/audio_convert.py`

This module converts incoming audio files of any format (`.mp3`, `.m4a`, `.flac`, etc.) into 16 kHz mono 16-bit PCM `.wav`, which is the format required by the Speech-to-Text model.

```python
1: """Audio format conversion utilities.
...
5: """
6: 
7: from __future__ import annotations
8: 
9: from pathlib import Path
10: from typing import Optional, Union
11: 
12: import numpy as np
13: import soundfile as sf
```

### Lines 1–13 Explanation:
* **Line 9 (`from pathlib import Path`)**: Imports Python's object-oriented file path utility (`Path`), which handles cross-platform Windows/Linux path slashes safely.
* **Line 10**: Imports typing helpers (`Optional` means a parameter can be `None`, `Union` means it can accept multiple types like `str` or `Path`).
* **Line 12 (`import numpy as np`)**: Imports NumPy, the standard library for multi-dimensional numerical arrays (used here to hold audio sample numbers).
* **Line 13 (`import soundfile as sf`)**: Imports SoundFile, which reads and writes WAV, FLAC, and OGG files to and from disk.

```python
15: def convert_to_wav(
16:     input_path: Union[str, Path],
17:     output_path: Optional[Union[str, Path]] = None,
18:     target_sr: int = 16000,
19: ) -> Path:
20:     """Convert any supported audio file to 16kHz mono WAV."""
21:     in_path = Path(input_path).resolve()
22:     if not in_path.is_file():
23:         raise FileNotFoundError(f"Audio file not found: {in_path}")
```

### Lines 15–23 Explanation:
* **Lines 15–19**: Defines `convert_to_wav`. Defaults the target sample rate `target_sr` to 16000 Hz (16 kHz).
* **Line 21 (`.resolve()`)**: Turns relative file paths into absolute, unambiguous paths on your hard drive.
* **Lines 22–23**: Checks if the file actually exists. If not, raises a `FileNotFoundError` immediately.

```python
25:     if in_path.suffix.lower() == ".wav" and output_path is None:
26:         # Check sample rate
27:         info = sf.info(str(in_path))
28:         if info.samplerate == target_sr and info.channels == 1:
29:             return in_path
```

### Lines 25–29 Explanation:
* Fast bypass optimization: If the file is already a `.wav`, has a sample rate of 16000 Hz, and is already 1 channel (mono), no conversion is needed. It returns `in_path` directly to save time and disk space.

```python
31:     if output_path is None:
32:         out_path = in_path.with_name(f"{in_path.stem}_16k.wav")
33:     else:
34:         out_path = Path(output_path).resolve()
35: 
36:     out_path.parent.mkdir(parents=True, exist_ok=True)
```

### Lines 31–36 Explanation:
* **Lines 31–32**: If no output path was passed, creates a new file name next to the original file with `_16k.wav` added (e.g. `speech.mp3` becomes `speech_16k.wav`).
* **Line 36 (`mkdir(parents=True, exist_ok=True)`)**: Creates any parent folders on disk if they don't already exist.

```python
38:     try:
39:         import av
40:         container = av.open(str(in_path))
41:         resampler = av.AudioResampler(format="s16", layout="mono", rate=target_sr)
42:         chunks = []
43:         for frame in container.decode(audio=0):
44:             frame.pts = None
45:             resampled_frames = resampler.resample(frame)
46:             if resampled_frames:
47:                 for rf in resampled_frames:
48:                     chunks.append(rf.to_ndarray())
49:         if chunks:
50:             raw_audio = np.concatenate(chunks, axis=1)[0].astype(np.float32) / 32768.0
51:             sf.write(str(out_path), raw_audio, target_sr, subtype="PCM_16")
52:             return out_path
53:     except Exception:
54:         pass
```

### Lines 38–54 Explanation (PyAV decoding):
* **Line 39**: Attempts to import `av` (the PyAV FFmpeg binding).
* **Line 40**: Opens the media file container.
* **Line 41**: Configures a hardware-grade resampler to produce 16-bit signed integer (`s16`), single channel (`mono`), at 16000 Hz.
* **Lines 43–48**: Loops over decoded audio frames, resamples each frame, converts it into a NumPy array, and stores it in `chunks`.
* **Line 50**: Joins all decoded chunks along the time axis. Divides by `32768.0` to convert 16-bit integer values (`-32768` to `+32767`) into standard 32-bit floating-point audio numbers (`-1.0` to `+1.0`).
* **Line 51**: Writes the array out as a standard 16-bit PCM WAV file.
* **Line 53–54**: If `av` fails or is not installed, catches the exception and falls through to the backup method below.

```python
56:     # Fallback to soundfile + scipy
57:     audio_data, sr = sf.read(str(in_path), dtype="float32")
58:     if getattr(audio_data, "ndim", 1) > 1:
59:         audio_data = np.mean(audio_data, axis=1)
60:     if sr != target_sr:
61:         try:
62:             import librosa
63:             audio_data = librosa.resample(audio_data, orig_sr=int(sr), target_sr=target_sr).astype(np.float32)
64:         except ImportError:
65:             import scipy.signal
66:             num_samples = int(len(audio_data) * target_sr / sr)
67:             audio_data = scipy.signal.resample(audio_data, num_samples).astype(np.float32)
68:     sf.write(str(out_path), audio_data, target_sr, subtype="PCM_16")
69:     return out_path
```

### Lines 56–69 Explanation (Fallback decoding):
* **Line 57**: Reads audio directly using `soundfile.read`.
* **Lines 58–59**: If the audio is stereo (2 channels, so `ndim > 1`), averages the left and right channels (`np.mean(..., axis=1)`) to make it mono.
* **Lines 60–67**: If the original sampling rate `sr` does not match 16 kHz:
  - Tries `librosa.resample` (highest quality).
  - If `librosa` isn't installed, uses `scipy.signal.resample` as a second fallback.
* **Lines 68–69**: Saves the file and returns the destination `Path`.

---

## 4. `core/stt_engine.py`

This module wraps **Faster-Whisper Turbo** to transcribe English speech with low latency on the GPU.

```python
1: """Faster-Whisper Turbo STT Engine for Audio Cascader.
...
5: """
6: 
7: from __future__ import annotations
8: 
9: import logging
10: from pathlib import Path
11: import time
12: from typing import Any, Dict, List, Optional, Union
13: 
14: import numpy as np
15: import soundfile as sf
16: import torch
17: from faster_whisper import WhisperModel
18: 
19: logger = logging.getLogger(__name__)
20: 
21: BASE_DIR = Path(__file__).resolve().parents[1]
22: DEFAULT_WEIGHTS_DIR = BASE_DIR / "weights" / "faster-whisper-turbo"
```

### Lines 1–22 Explanation:
* **Line 16 (`import torch`)**: Imports PyTorch to check for CUDA graphics card availability (`torch.cuda.is_available()`).
* **Line 17 (`from faster_whisper import WhisperModel`)**: Imports the Faster-Whisper model class, which runs on CTranslate2 (an optimized inference engine for transformer models).
* **Line 21 (`BASE_DIR = Path(__file__).resolve().parents[1]`)**: Finds the root directory of the project by navigating two levels up from `core/stt_engine.py`.
* **Line 22**: Locates the offline model weights folder at `weights/faster-whisper-turbo/`.

```python
25: class STTEngine:
26:     """Wrapper for Faster-Whisper Turbo GPU inference."""
27: 
28:     def __init__(
29:         self,
30:         model_path: Optional[Union[str, Path]] = None,
31:         device: str = "cuda" if torch.cuda.is_available() else "cpu",
32:         compute_type: Optional[str] = None,
33:         cpu_threads: int = 4,
34:     ):
35:         self.model_path = Path(model_path) if model_path else DEFAULT_WEIGHTS_DIR
36:         self.device = device
37:         if compute_type is None:
38:             self.compute_type = "int8_float16" if self.device == "cuda" else "int8"
39:         else:
40:             self.compute_type = compute_type
41:         self.cpu_threads = cpu_threads
42:         self._model: Optional[WhisperModel] = None
```

### Lines 25–42 (`__init__`) Explanation:
* **Line 31**: Automatically chooses `"cuda"` if an NVIDIA GPU is detected; otherwise falls back to `"cpu"`.
* **Lines 37–40**: Sets the mathematical precision. On GPU, `"int8_float16"` runs computations in 16-bit floating point while keeping weights in 8-bit integer format. This halves VRAM usage while retaining full model accuracy. On CPU, it uses `"int8"`.
* **Line 42 (`self._model = None`)**: The model is lazily loaded (not loaded into VRAM until actually requested).

```python
44:     def load(self) -> WhisperModel:
45:         """Load Faster-Whisper model into memory/VRAM."""
46:         if self._model is not None:
47:             return self._model
48: 
49:         if not self.model_path.exists():
50:             raise FileNotFoundError(f"Model weights not found at: {self.model_path}")
```

### Lines 44–50 Explanation:
* **Lines 46–47**: If the model is already in memory, returns it immediately without reloading.
* **Lines 49–50**: Verifies that the weights directory exists on disk.

```python
52:         try:
53:             logger.info("Loading Faster-Whisper from %s on %s (%s)...", self.model_path, self.device, self.compute_type)
54:             self._model = WhisperModel(
55:                 str(self.model_path),
56:                 device=self.device,
57:                 compute_type=self.compute_type,
58:                 cpu_threads=self.cpu_threads,
59:             )
60:             logger.info("Faster-Whisper model loaded successfully.")
61:         except Exception as exc:
62:             if self.device == "cuda" and self.compute_type != "int8":
...
74:                     self.device = "cpu"
75:                     self.compute_type = "int8"
76:                     self._model = WhisperModel(...)
```

### Lines 52–84 Explanation (Model loading & fallback):
* **Lines 54–59**: Instantiates `WhisperModel`.
* **Lines 61–80**: Error handling fallback: If your GPU does not support `int8_float16` instructions or runs out of memory, it catches the error and retries with pure CPU mode so the app never crashes.

```python
86:     def unload(self) -> None:
87:         """Unload model and free GPU VRAM."""
88:         if self._model is not None:
89:             del self._model
90:             self._model = None
91:             import gc
92:             gc.collect()
93:             if torch.cuda.is_available():
94:                 torch.cuda.empty_cache()
95:             logger.info("Faster-Whisper model unloaded from VRAM.")
```

### Lines 86–95 (`unload`) Explanation:
* **Line 89 (`del self._model`)**: Deletes the Python reference to the neural network.
* **Line 91–92 (`gc.collect()`)**: Triggers Python's Garbage Collector to free the memory immediately.
* **Lines 93–94 (`torch.cuda.empty_cache()`)**: Commands the NVIDIA CUDA driver to release cached GPU memory back to Windows. This is vital when switching between STT and TTS in batch mode on GPUs with 4GB–6GB VRAM.

```python
97:     def transcribe(
98:         self,
99:         audio: Union[str, Path, np.ndarray],
100:         language: Optional[str] = "en",
101:         beam_size: int = 1,
102:         condition_on_previous_text: bool = False,
103:         **kwargs: Any,
104:     ) -> Dict[str, Any]:
```

### Lines 97–104 (`transcribe` signature) Explanation:
* Accepts `audio` as either a file path or a live NumPy waveform array.
* Sets `beam_size = 1`: Uses greedy decoding for maximum speed (lowest latency).
* `condition_on_previous_text = False`: Prevents Whisper from hallucinating or getting stuck repeating phrases from earlier sentences.

```python
116:         model = self.load()
117: 
118:         if isinstance(audio, (str, Path)):
119:             audio_path = Path(audio).resolve()
...
126:             if sr != 16000:
...
134:             audio_input = audio_data
135:         else:
136:             audio_input = np.asarray(audio, dtype=np.float32)
137: 
138:         duration = len(audio_input) / 16000.0
```

### Lines 116–138 Explanation:
* Loads the model if not already loaded.
* Reads and normalizes input audio: If a file path was passed, reads it with `sf.read` and resamples to 16 kHz. If an in-memory NumPy array was passed (from the live mic), formats it directly as 32-bit floats.
* Calculates `duration` in seconds: number of samples divided by 16000.

```python
140:         t0_gen = time.perf_counter()
141:         segments_gen, info = model.transcribe(
142:             audio_input,
143:             language=language,
144:             beam_size=beam_size,
145:             condition_on_previous_text=condition_on_previous_text,
146:             **kwargs,
147:         )
148: 
149:         first_word_time_ms = None
150:         first_word = ""
151:         segments: List[Dict[str, Any]] = []
152: 
153:         for s in segments_gen:
154:             if first_word_time_ms is None and s.text.strip():
155:                 first_word_time_ms = (time.perf_counter() - t0_gen) * 1000.0
156:                 words = s.text.strip().split()
157:                 if words:
158:                     first_word = words[0]
159:             segments.append({"start": s.start, "end": s.end, "text": s.text})
160: 
161:         full_text = " ".join(s["text"].strip() for s in segments).strip()
162: 
163:         return {
164:             "text": full_text,
165:             "segments": segments,
166:             "language": info.language if hasattr(info, "language") else language,
167:             "duration": duration,
168:             "first_word": first_word,
169:             "first_word_time_ms": round(first_word_time_ms, 1) if first_word_time_ms else None,
170:         }
```

### Lines 140–170 Explanation:
* **Line 140 (`time.perf_counter()`)**: High-precision benchmark timer.
* **Line 141 (`model.transcribe`)**: Runs Whisper inference. Note that `segments_gen` is a generator (yields transcribed segments one by one).
* **Lines 153–159**: Loops through each segment. The moment the first segment yields, it computes `first_word_time_ms` (measuring exactly how many milliseconds it took to identify the first word).
* **Line 161**: Combines all segment texts into `full_text`.
* **Lines 163–170**: Returns a clean Python dictionary containing the text, segments, durations, and latency statistics.

---

## 5. `core/tts_engine.py`

This module wraps Kyutai's Pocket TTS model, caches the reference American voice embedding to avoid repeated conditioning overhead, and handles streaming synthesis.

```python
1: """Kyutai Pocket TTS Engine with Voice Embedding Caching.
...
5: """
6: 
7: from __future__ import annotations
8: 
9: import logging
10: from pathlib import Path
11: import time
12: from typing import Any, Dict, Iterator, Optional, Union
13: 
14: import numpy as np
15: import torch
16: 
17: logger = logging.getLogger(__name__)
18: 
19: BASE_DIR = Path(__file__).resolve().parents[1]
20: DEFAULT_REF_VOICE = BASE_DIR / "references" / "american accent.wav"
21: DEFAULT_EMBEDDING_PATH = BASE_DIR / "embeddings" / "american_accent.pt"
```

### Lines 1–21 Explanation:
* **Line 20**: Points to the 13-second audio clip of the target American speaker (`references/american accent.wav`).
* **Line 21**: Points to the precomputed voice conditioning cache file (`embeddings/american_accent.pt`).

```python
24: class TTSEngine:
25:     """Wrapper for Kyutai Pocket TTS synthesis with cached voice state."""
26: 
27:     def __init__(
28:         self,
29:         device: str = "cuda" if torch.cuda.is_available() else "cpu",
30:         embedding_path: Optional[Union[str, Path]] = None,
31:         reference_voice: Optional[Union[str, Path]] = None,
32:     ):
33:         self.device = device
34:         self.embedding_path = Path(embedding_path) if embedding_path else DEFAULT_EMBEDDING_PATH
35:         self.reference_voice = Path(reference_voice) if reference_voice else DEFAULT_REF_VOICE
36:         self._model = None
37:         self.voice_state: Optional[Dict[str, Any]] = None
```

### Lines 24–37 (`__init__`) Explanation:
* Stores configuration parameters: GPU or CPU device, the path to the cached embedding, and the reference voice.
* `self._model` and `self.voice_state` are initialized to `None` (lazy initialization).

```python
39:     def load(self):
40:         """Load Pocket TTS model into VRAM."""
41:         if self._model is not None:
42:             return self._model
43: 
44:         import sys
45:         engine_dir = BASE_DIR
46:         if str(engine_dir) not in sys.path:
47:             sys.path.insert(0, str(engine_dir))
48: 
49:         from pocket_tts.models.tts_model import TTSModel
50: 
51:         logger.info("Loading Pocket TTS model on device: %s...", self.device)
52:         self._model = TTSModel.load_model(device=self.device)
53:         logger.info("Pocket TTS model loaded successfully.")
54: 
55:         # Ensure voice state is initialized
56:         self.get_or_create_voice_state()
57:         return self._model
```

### Lines 39–57 (`load`) Explanation:
* **Lines 44–47**: Ensures the project directory is in Python's search path (`sys.path`) so that the vendored `pocket_tts` package can be imported directly.
* **Line 49**: Imports `TTSModel`.
* **Line 52**: Calls `TTSModel.load_model(device=self.device)`. This loads the FlowLM transformer and Mimi neural audio codec into VRAM.
* **Line 56**: Calls `get_or_create_voice_state()` so that the voice embedding is loaded and ready before any synthesis requests arrive.

```python
59:     def get_or_create_voice_state(self, force_refresh: bool = False) -> Dict[str, Any]:
60:         """Load voice state from cached .pt file or compute and save it."""
61:         if self.voice_state is not None and not force_refresh:
62:             return self.voice_state
63: 
64:         # 1. Try loading cached embedding (.pt)
65:         if self.embedding_path.exists() and not force_refresh:
66:             t0 = time.perf_counter()
67:             logger.info("Loading cached voice embedding from %s...", self.embedding_path)
68:             self.voice_state = torch.load(str(self.embedding_path), map_location=self.device, weights_only=False)
69:             logger.info("Voice embedding loaded in %.1fms", (time.perf_counter() - t0) * 1000.0)
70:             return self.voice_state
```

### Lines 59–70 Explanation (Fast cache loading):
* **Lines 61–62**: If the voice state is already stored in memory, returns it immediately.
* **Lines 65–70**: If `embeddings/american_accent.pt` exists on disk, loads it using `torch.load` mapped to the active device. This takes ~30–50 ms, completely bypassing the heavy computation of analyzing the reference audio file.

```python
72:         # 2. Compute embedding from reference audio file
73:         if self._model is None:
74:             ...
78:             self._model = TTSModel.load_model(device=self.device)
79: 
80:         if not self.reference_voice.exists():
81:             logger.warning("Reference voice %s not found! Using built-in voice 'alba'", self.reference_voice)
82:             self.voice_state = self._model.get_state_for_audio_prompt("alba")
83:             return self.voice_state
84: 
85:         logger.info("Computing voice embedding from reference audio: %s...", self.reference_voice)
86:         t0 = time.perf_counter()
87:         self.voice_state = self._model.get_state_for_audio_prompt(str(self.reference_voice))
88:         elapsed_ms = (time.perf_counter() - t0) * 1000.0
89:         logger.info("Voice conditioning computed in %.1fms.", elapsed_ms)
90: 
91:         # 3. Save to disk for fast <50ms future reloads
92:         self.embedding_path.parent.mkdir(parents=True, exist_ok=True)
93:         torch.save(self.voice_state, str(self.embedding_path))
94:         logger.info("Saved persistent voice embedding to: %s", self.embedding_path)
95: 
96:         return self.voice_state
```

### Lines 72–96 Explanation (Embedding computation & persistence):
* If the `.pt` file does not exist yet:
  - Line 80: Checks if the reference audio file exists. If missing, falls back to the default voice (`"alba"`).
  - Line 87: Passes `american accent.wav` through FlowLM's audio conditioner (`get_state_for_audio_prompt`). This encodes the speaker's vocal characteristics (pitch, formant, rhythm) into a PyTorch key-value state dictionary.
  - Lines 92–93: Saves that state dictionary to `embeddings/american_accent.pt` using `torch.save`. Next time, it will load instantly via Line 68.

```python
98:     @staticmethod
99:     def trim_leading_silence(audio_np: np.ndarray, sr: int = 24000, threshold: float = 0.015) -> np.ndarray:
100:         """Trim FlowLM prompt conditioning silence from audio start."""
101:         if len(audio_np) == 0:
102:             return audio_np
103:         above = np.where(np.abs(audio_np) >= threshold)[0]
104:         if len(above) > 0 and above[0] > int(sr * 0.10):
105:             start_idx = max(0, above[0] - int(sr * 0.04))
106:             return audio_np[start_idx:]
107:         return audio_np
```

### Lines 98–107 (`trim_leading_silence`) Explanation:
* Neural TTS models often output a brief 100–200 ms silent gap at the start of synthesis while priming internal autoregressive states.
* Line 103: Uses `np.where(np.abs(audio_np) >= threshold)` to locate the first audio sample with actual sound volume.
* Lines 104–106: If there is more than 100 ms of leading silence, it slices the array so audio begins immediately with a subtle 40 ms lead-in, eliminating dead air.

```python
109:     @staticmethod
110:     def normalize_audio(audio_np: np.ndarray, target_peak: float = 0.90) -> np.ndarray:
111:         """Normalize peak amplitude to guarantee audible, clear volume."""
112:         if len(audio_np) == 0:
113:             return audio_np
114:         peak = float(np.max(np.abs(audio_np)))
115:         if peak > 0.001:
116:             return (audio_np / peak) * target_peak
117:         return audio_np
```

### Lines 109–117 (`normalize_audio`) Explanation:
* Line 114: Calculates the highest volume peak (`np.max(np.abs(audio_np))`).
* Line 116: Scales all audio samples proportionally so the highest peak hits exactly 90% (`0.90`) of maximum volume. This prevents quiet whispering or loud clipping.

```python
119:     def synthesize_stream(self, text: str, voice_state: Optional[Dict[str, Any]] = None) -> Iterator[np.ndarray]:
120:         """Yield raw 24kHz float32 audio chunks as they are generated."""
121:         model = self.load()
122:         state = voice_state or self.voice_state
123:         if state is None:
124:             state = self.get_or_create_voice_state()
125: 
126:         stream = model.generate_audio_stream(
127:             state,
128:             text_to_generate=text,
129:             copy_state=True,
130:         )
131: 
132:         for chunk_tensor in stream:
133:             chunk_np = chunk_tensor.detach().cpu().numpy()
134:             yield chunk_np
```

### Lines 119–135 (`synthesize_stream`) Explanation:
* **`Iterator[np.ndarray]`**: This function is a **generator**.
* **Line 126 (`model.generate_audio_stream`)**: Pocket TTS generates audio frame by frame.
* **Line 129 (`copy_state=True`)**: Clones the voice state so generation does not overwrite the master embedding in memory.
* **Lines 132–134**: For each small batch of audio generated by the GPU:
  - `.detach()`: Disconnects the tensor from PyTorch's gradient calculation graph.
  - `.cpu()`: Moves the data from GPU VRAM to system RAM.
  - `.numpy()`: Converts it to a NumPy array.
  - `yield chunk_np`: Sends the chunk to the caller immediately without waiting for the rest of the sentence.

```python
136:     def synthesize_chunk(self, text: str, voice_state: Optional[Dict[str, Any]] = None) -> np.ndarray:
137:         """Synthesize a text chunk to a clean, normalized 24kHz float32 audio array."""
138:         chunks = list(self.synthesize_stream(text, voice_state))
139:         if not chunks:
140:             return np.zeros(0, dtype=np.float32)
141: 
142:         full_audio = np.concatenate(chunks)
143:         trimmed = self.trim_leading_silence(full_audio, sr=24000)
144:         normalized = self.normalize_audio(trimmed, target_peak=0.90)
145:         return normalized
```

### Lines 136–146 (`synthesize_chunk`) Explanation:
* Collects all yielded stream pieces into a single list (`list(...)`), concatenates them into one continuous waveform (`np.concatenate`), trims leading silence, normalizes the volume, and returns a single array. Used in batch file mode.

---

## 6. `core/audio_stream.py`

This module provides hardware-level audio streaming: recording from your microphone, detecting when you start and stop talking using Silero VAD, and playing audio smoothly through your speakers.

```python
1: """Audio capture, playback, and streaming Voice Activity Detection (VAD).
...
5: """
6: 
7: from __future__ import annotations
8: 
9: import collections
10: import logging
11: import queue
12: import threading
13: import time
14: from typing import Callable, List, Optional, Tuple
15: 
16: import numpy as np
17: import sounddevice as sd
18: import torch
```

### Lines 1–18 Explanation:
* **Line 9 (`import collections`)**: Imports `deque` (a double-ended queue), used for low-overhead audio buffering.
* **Line 11 (`import queue`)**: Thread-safe FIFO queues for transferring audio chunks between threads.
* **Line 12 (`import threading`)**: Thread synchronization tools (`Lock`, `Event`).
* **Line 17 (`import sounddevice as sd`)**: Interface to PortAudio/Windows WASAPI for low-latency microphone input and speaker output.

```python
23: def list_audio_devices() -> Tuple[List[dict], List[dict]]:
24:     """Enumerate available input and output audio devices."""
25:     devices = sd.query_devices()
26:     default_in = sd.default.device[0]
27:     default_out = sd.default.device[1]
28: 
29:     input_devs = []
30:     output_devs = []
31: 
32:     for idx, d in enumerate(devices):
33:         info = {
34:             "index": idx,
35:             "name": d["name"],
36:             "hostapi": d["hostapi"],
37:             "max_inputs": d["max_input_channels"],
38:             "max_outputs": d["max_output_channels"],
39:             "default_samplerate": d["default_samplerate"],
40:             "is_default_in": (idx == default_in),
41:             "is_default_out": (idx == default_out),
42:         }
43:         if d["max_input_channels"] > 0:
44:             input_devs.append(info)
45:         if d["max_output_channels"] > 0:
46:             output_devs.append(info)
47: 
48:     return input_devs, output_devs
```

### Lines 23–48 (`list_audio_devices`) Explanation:
* **Line 25**: Queries Windows for all connected microphones, headsets, virtual audio cables, and speakers.
* **Lines 26–27**: Identifies which devices are configured as system defaults.
* **Lines 32–46**: Sorts devices into `input_devs` (if `max_input_channels > 0`) and `output_devs` (if `max_output_channels > 0`) and returns them as a tuple.

```python
51: class AudioCapture:
52:     """Continuously captures microphone audio in low-latency 512-sample blocks."""
53: 
54:     def __init__(
55:         self,
56:         sample_rate: int = 16000,
57:         block_size: int = 512,
58:         device_index: Optional[int] = None,
59:         vu_callback: Optional[Callable[[float], None]] = None,
60:     ):
61:         self.sample_rate = sample_rate
62:         self.block_size = block_size
63:         self.device_index = device_index
64:         self.vu_callback = vu_callback
65: 
66:         self.audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)
67:         self._stream: Optional[sd.InputStream] = None
68:         self._running = False
```

### Lines 51–68 (`AudioCapture.__init__`) Explanation:
* Initializes microphone capture settings:
  - `sample_rate = 16000`: 16 kHz sample rate.
  - `block_size = 512`: Delivers audio in chunks of 512 samples (~32 milliseconds of audio at a time).
  - `vu_callback`: A function that receives the volume level so the GUI can update the live microphone volume bar.
  - `self.audio_queue`: Holds recorded audio blocks for the VAD to process.

```python
70:     def _audio_callback(self, indata, frames, time_info, status):
71:         chunk = indata[:, 0].copy()
72:         if self.vu_callback:
73:             rms = float(np.sqrt(np.mean(chunk**2) + 1e-12))
74:             self.vu_callback(rms)
75:         try:
76:             self.audio_queue.put_nowait(chunk)
77:         except queue.Full:
78:             pass
```

### Lines 70–78 (`_audio_callback`) Explanation:
* This callback is executed by the sound card driver hardware thread every 32 ms:
  - Line 71: Grabs channel 0 (mono).
  - Lines 72–74: Computes the RMS (Root Mean Square) energy of the sound wave (its physical volume) and sends it to `self.vu_callback` for the visual meter.
  - Line 76: Pushes the audio slice into `self.audio_queue`. If the queue is full (the system is lagging), drops the chunk rather than blocking the audio driver thread.

```python
80:     def start(self):
81:         if self._running:
82:             return
83:         self._running = True
84:         self._stream = sd.InputStream(
85:             samplerate=self.sample_rate,
86:             channels=1,
87:             dtype="float32",
88:             blocksize=self.block_size,
89:             device=self.device_index,
90:             callback=self._audio_callback,
91:         )
92:         self._stream.start()
93: 
94:     def stop(self):
...
104:     def get_chunk(self, timeout: float = 0.1) -> Optional[np.ndarray]:
```

### Lines 80–109 Explanation:
* **Lines 80–92 (`start`)**: Creates and starts a `sounddevice.InputStream`.
* **Lines 94–102 (`stop`)**: Stops and closes the hardware stream cleanly.
* **Lines 104–109 (`get_chunk`)**: Safely pulls the next recorded chunk out of the queue with a timeout.

```python
111: class StreamingVADSegmenter:
112:     """Segments continuous microphone audio using Silero VAD."""
113: 
114:     def __init__(
115:         self,
116:         sample_rate: int = 16000,
117:         chunk_size: int = 512,
118:         speech_threshold: float = 0.5,
119:         min_silence_duration_ms: int = 300,
120:         min_speech_duration_ms: int = 250,
121:         max_speech_duration_s: float = 3.5,
122:     ):
```

### Lines 111–122 (`StreamingVADSegmenter.__init__`) Explanation:
* **`speech_threshold = 0.5`**: Silero outputs a probability score from 0.0 to 1.0. A score >= 0.5 indicates human voice.
* **`min_silence_duration_ms = 300`**: After speaking, if 300 ms of silence is detected, it considers that utterance finished and triggers transcription.
* **`min_speech_duration_ms = 250`**: Ignores brief noises under 250 ms (like a cough or mouse click).
* **`max_speech_duration_s = 3.5`**: If you speak without pausing, it cuts the audio at 3.5 seconds so speech-to-text processing can begin without waiting endlessly.

```python
130:         from silero_vad import load_silero_vad
131:         self.model = load_silero_vad()
132:         self.model.eval()
133: 
134:         self.current_speech_chunks: List[np.ndarray] = []
135:         self.silence_chunk_count = 0
136:         self.in_speech = False
137:         self.speech_start_time: Optional[float] = None
138:         self.pre_buffer = collections.deque(maxlen=3)
```

### Lines 130–138 Explanation:
* **Line 131**: Loads the lightweight Silero VAD neural network.
* **Line 138 (`self.pre_buffer`)**: A circular buffer storing the previous 3 audio chunks (~100 ms). This ensures the very start of the first word (like "P" or "T" sounds) isn't clipped before the neural network detects speech.

```python
147:     def process_chunk(self, chunk: np.ndarray) -> Optional[Tuple[np.ndarray, float, float]]:
148:         tensor_chunk = torch.from_numpy(chunk).float()
149:         with torch.no_grad():
150:             prob = self.model(tensor_chunk, self.sample_rate).item()
151: 
152:         now = time.perf_counter()
```

### Lines 147–152 Explanation:
* Converts the 512-sample audio chunk into a PyTorch tensor.
* Runs the chunk through `self.model`: Evaluates speech probability `prob`.

```python
154:         if prob >= self.speech_threshold:
155:             if not self.in_speech:
156:                 self.in_speech = True
157:                 self.speech_start_time = now
158:                 self.current_speech_chunks.extend(list(self.pre_buffer))
159:             self.current_speech_chunks.append(chunk)
160:             self.silence_chunk_count = 0
161: 
162:             if len(self.current_speech_chunks) >= self.max_speech_chunks:
163:                 segment_audio = np.concatenate(self.current_speech_chunks)
164:                 start_t = self.speech_start_time or now
165:                 self.reset()
166:                 return segment_audio, start_t, now
```

### Lines 154–166 Explanation (Speech Detected):
* If `prob >= 0.5`:
  - If we were not previously in speech, sets `self.in_speech = True`, logs the start timestamp, and adds the pre-buffer audio.
  - Appends the current chunk to `self.current_speech_chunks`.
  - Resets silence count.
  - If the utterance exceeds the maximum duration (3.5 seconds, line 162), finishes the segment and returns the full audio array and timestamps.

```python
167:         else:
168:             if self.in_speech:
169:                 self.current_speech_chunks.append(chunk)
170:                 self.silence_chunk_count += 1
171:                 if self.silence_chunk_count >= self.min_silence_chunks:
172:                     if len(self.current_speech_chunks) >= self.min_speech_chunks:
173:                         segment_audio = np.concatenate(self.current_speech_chunks)
174:                         start_t = self.speech_start_time or (now - 0.5)
175:                         self.reset()
176:                         return segment_audio, start_t, now
177:                     else:
178:                         self.reset()
179:             else:
180:                 self.pre_buffer.append(chunk)
181: 
182:         return None
```

### Lines 167–182 Explanation (Silence Detected):
* If `prob < 0.5`:
  - If currently speaking, increments the silence counter.
  - If silence persists for 300 ms (line 171) and total speech exceeds 250 ms (line 172), finishes the utterance and returns the audio segment to send to Whisper STT.
  - If not in speech, adds the chunk to `self.pre_buffer` (line 180).
  - Returns `None` while still listening.

```python
185: class StreamingAudioPlayer:
186:     """Low-latency audio playback engine with jitter buffer."""
187: 
188:     def __init__(
189:         self,
190:         sample_rate: int = 24000,
191:         device_index: Optional[int] = None,
192:         block_size: int = 1024,
193:         vu_callback: Optional[Callable[[float], None]] = None,
194:     ):
```

### Lines 185–194 (`StreamingAudioPlayer.__init__`) Explanation:
* Sets up audio output at 24000 Hz (24 kHz, which matches the output rate of the Mimi codec used by Pocket TTS).
* `block_size = 1024`: Delivers audio to the speakers in 1024-sample blocks.

```python
200:         self._lock = threading.Lock()
201:         self._buffer: collections.deque[np.ndarray] = collections.deque()
202:         self._current_chunk: Optional[np.ndarray] = None
203:         self._current_pos: int = 0
204:         self._stream: Optional[sd.OutputStream] = None
205:         self._running = False
206:         self._is_playing = False
```

### Lines 200–206 Explanation:
* **`self._lock`**: Thread mutex lock. Prevents data race conditions when one thread pushes audio while the audio card callback is pulling audio.
* **`self._buffer`**: A FIFO queue holding generated audio arrays waiting to be played.
* **`self._current_pos`**: Remembers how many samples of the current chunk have been sent to the speaker so far.

```python
208:     def push_audio(self, audio_data: np.ndarray):
209:         if audio_data.dtype != np.float32:
210:             audio_data = audio_data.astype(np.float32)
211:         with self._lock:
212:             self._buffer.append(audio_data)
```

### Lines 208–212 (`push_audio`) Explanation:
* Called by the TTS worker thread whenever a new chunk of synthesized audio is generated. Safely appends the audio array to `self._buffer`.

```python
214:     def _audio_callback(self, outdata, frames, time_info, status):
215:         outdata.fill(0.0)
216:         needed = frames
217:         write_idx = 0
218: 
219:         with self._lock:
220:             while needed > 0:
221:                 if self._current_chunk is None or self._current_pos >= len(self._current_chunk):
222:                     if len(self._buffer) > 0:
223:                         self._current_chunk = self._buffer.popleft()
224:                         self._current_pos = 0
225:                     else:
226:                         self._current_chunk = None
227:                         self._is_playing = False
228:                         break
229: 
230:                 self._is_playing = True
231:                 available = len(self._current_chunk) - self._current_pos
232:                 to_copy = min(needed, available)
233:                 outdata[write_idx : write_idx + to_copy, 0] = self._current_chunk[
234:                     self._current_pos : self._current_pos + to_copy
235:                 ]
236: 
237:                 self._current_pos += to_copy
238:                 write_idx += to_copy
239:                 needed -= to_copy
```

### Lines 214–239 (`_audio_callback`) Explanation:
* The sound card calls this function to request exactly `frames` (1024) audio samples:
  - Line 215: Fills output buffer with silence (`0.0`).
  - Line 221–228: If the current audio chunk is finished, grabs the next one from `self._buffer`. If no audio is ready, breaks out (resulting in silence, preventing buffer underrun crashes).
  - Lines 231–235: Copies available samples directly into the sound card's `outdata` buffer.
  - Lines 237–239: Updates index pointers.

```python
241:         if self.vu_callback:
242:             rms = float(np.sqrt(np.mean(outdata[:, 0] ** 2) + 1e-12))
243:             self.vu_callback(rms)
```

### Lines 241–243 Explanation:
* Calculates the RMS output volume and sends it to the speaker VU meter in the GUI.

```python
245:     def start(self):
...
259:     def stop(self):
...
270:     def clear(self):
...
278:     def is_playing(self) -> bool:
```

### Lines 245–280 Explanation:
* **`start()`**: Launches `sd.OutputStream`.
* **`stop()`**: Shuts down the output stream.
* **`clear()`**: Empties playback queues immediately (used when stopping live conversion).
* **`is_playing`**: Boolean property indicating whether audio is actively playing through the speakers.

---

## 7. `pipeline.py`

This is the batch file conversion pipeline. It takes an existing audio file on disk and produces a converted American-accented WAV file.

```python
1: """Audio Cascader — File-Mode Speech-to-Speech (S2S) Pipeline.
...
5: """
6: 
7: from __future__ import annotations
8: 
9: import argparse
10: from datetime import datetime
11: import logging
12: from pathlib import Path
13: import sys
14: import time
15: 
16: import numpy as np
17: import soundfile as sf
18: import torch
```

### Lines 1–18 Explanation:
* Standard imports: `argparse` for parsing command-line flags (`--audio`, `--output`, `--play`), `datetime` for timestamped filenames, and PyTorch/SoundFile for audio operations.

```python
21: BASE_DIR = Path(__file__).resolve().parent
22: if str(BASE_DIR) not in sys.path:
23:     sys.path.insert(0, str(BASE_DIR))
24: 
25: from core.audio_convert import convert_to_wav
26: from core.chunker import chunk_text
27: from core.stt_engine import STTEngine
28: from core.tts_engine import TTSEngine
```

### Lines 21–28 Explanation:
* Connects the root directory to `sys.path` and imports our four core components: audio conversion, text chunking, STT, and TTS.

```python
34: def run_pipeline(
35:     input_audio: Union[str, Path],
36:     output_path: Optional[Union[str, Path]] = None,
37:     device: str = "cuda" if torch.cuda.is_available() else "cpu",
38:     play_audio: bool = False,
39: ) -> Path:
40:     """Execute end-to-end file S2S conversion."""
41:     t_start = time.perf_counter()
```

### Lines 34–41 Explanation:
* Defines the primary entrypoint function `run_pipeline`. Starts a timer `t_start` to measure total glass-to-glass conversion latency.

```python
46:     # 1. Audio Ingestion & Normalization to 16kHz WAV
47:     in_path = Path(input_audio).resolve()
48:     if not in_path.exists():
49:         raise FileNotFoundError(f"Input audio file not found: {in_path}")
50: 
51:     logger.info("[Step 1/5] Ingesting input audio: %s", in_path.name)
52:     wav_16k = convert_to_wav(in_path)
```

### Lines 46–52 (Step 1) Explanation:
* Verifies the file exists and calls `convert_to_wav(in_path)` to ensure the input audio is formatted as 16 kHz mono.

```python
54:     # 2. STT: Faster-Whisper Turbo GPU
55:     logger.info("[Step 2/5] Transcribing speech with Faster-Whisper Turbo (%s)...", device)
56:     stt = STTEngine(device=device)
57:     t_stt0 = time.perf_counter()
58:     transcription = stt.transcribe(wav_16k, language="en", beam_size=1)
59:     stt_duration = time.perf_counter() - t_stt0
60: 
61:     text = transcription["text"].strip()
62:     audio_dur = transcription["duration"]
63:     rtf_stt = stt_duration / max(audio_dur, 0.01)
64: 
65:     logger.info("Transcribed Indian English: \"%s\"", text)
66:     logger.info("STT Latency: %.2fs | Audio Duration: %.2fs | STT RTF: %.3f", stt_duration, audio_dur, rtf_stt)
67: 
68:     # Free STT VRAM and memory before TTS
69:     stt.unload()
```

### Lines 54–69 (Step 2) Explanation:
* Runs Whisper Turbo to transcribe the entire audio file.
* Line 63: Computes Real-Time Factor (`RTF = processing_time / audio_duration`). An RTF of 0.1 means 10 seconds of speech was transcribed in 1 second.
* **Line 69 (`stt.unload()`)**: Critical memory management step. Whisper is deleted and flushed from GPU VRAM before Pocket TTS is loaded, ensuring stability on GPUs with 4GB–6GB VRAM.

```python
71:     if not text:
72:         logger.warning("No speech transcribed. Output audio will not be generated.")
73:         return in_path
```

### Lines 71–73 Explanation:
* If the input audio was empty or only background noise, logs a warning and exits cleanly without error.

```python
75:     # 3. Punctuation & Pause Chunking
76:     logger.info("[Step 3/5] Chunking transcribed text...")
77:     chunks = chunk_text(text, max_words=12)
78:     logger.info("Generated %d clauses for streaming synthesis:", len(chunks))
79:     for i, c in enumerate(chunks, 1):
80:         logger.info("  [%d/%d] \"%s\"", i, len(chunks), c)
```

### Lines 75–81 (Step 3) Explanation:
* Calls `chunk_text` with a 12-word cap, generating natural speaking clauses.

```python
83:     # 4. Pocket TTS American Accent Synthesis
84:     logger.info("[Step 4/5] Initializing Kyutai Pocket TTS with American accent embedding...")
85:     tts = TTSEngine(device=device)
86:     voice_state = tts.get_or_create_voice_state()
87: 
88:     generated_audio_chunks = []
89:     t_tts0 = time.perf_counter()
90:     first_chunk_latency = None
91: 
92:     for idx, clause in enumerate(chunks, 1):
93:         t_c0 = time.perf_counter()
94:         audio_chunk = tts.synthesize_chunk(clause, voice_state=voice_state)
95:         chunk_time = time.perf_counter() - t_c0
96:         if first_chunk_latency is None:
97:             first_chunk_latency = chunk_time
98: 
99:         logger.info("  Clause %d/%d rendered (%d samples, %.2fs)", idx, len(chunks), len(audio_chunk), chunk_time)
100:         if len(audio_chunk) > 0:
101:             generated_audio_chunks.append(audio_chunk)
102: 
103:     total_tts_duration = time.perf_counter() - t_tts0
```

### Lines 83–103 (Step 4) Explanation:
* Initializes `TTSEngine` and loads the cached voice embedding.
* Loops through each clause one by one, synthesizing it via `tts.synthesize_chunk()`.
* Tracks Time-To-First-Byte (`first_chunk_latency`) and stores the audio arrays in `generated_audio_chunks`.

```python
107:     # 5. Concatenation, Normalization, and Export
108:     full_audio = np.concatenate(generated_audio_chunks)
109:     full_audio = TTSEngine.normalize_audio(full_audio, target_peak=0.90)
110: 
111:     # Determine output path
112:     if output_path is None:
113:         outputs_dir = BASE_DIR / "outputs"
114:         outputs_dir.mkdir(parents=True, exist_ok=True)
115:         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
116:         out_file = outputs_dir / f"pipeline_output_{timestamp}.wav"
117:     else:
118:         out_file = Path(output_path).resolve()
119:         out_file.parent.mkdir(parents=True, exist_ok=True)
120: 
121:     logger.info("[Step 5/5] Writing output WAV to: %s", out_file)
122:     sf.write(str(out_file), full_audio, 24000, subtype="PCM_16")
```

### Lines 107–123 (Step 5) Explanation:
* Glues all clause waveforms together using `np.concatenate`.
* Normalizes peak volume to 90% (`0.90`).
* Generates a timestamped filename inside `outputs/` (e.g. `pipeline_output_20261016_104500.wav`) and saves it as a 24 kHz WAV file.

```python
127:     print("\n" + "=" * 60)
128:     print("AUDIO CASCADER — PERFORMANCE SUMMARY")
...
137:     print("=" * 60 + "\n")
138: 
139:     if play_audio:
...
144:             sd.play(full_audio, 24000)
145:             sd.wait()
148:     return out_file
```

### Lines 127–148 Explanation:
* Prints a clean performance summary table showing total run time, STT latency, and output duration.
* If `--play` was passed, plays the converted American audio through your speakers before exiting.

```python
151: def main():
152:     parser = argparse.ArgumentParser(description="Audio Cascader — Indian-to-American S2S Pipeline")
153:     parser.add_argument("--audio", "-a", required=True, help="Path to input audio file (.wav, .m4a, .mp3)")
154:     parser.add_argument("--output", "-o", default=None, help="Path to save output .wav file")
155:     parser.add_argument("--device", "-d", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
156:     parser.add_argument("--play", "-p", action="store_true", help="Play output audio upon completion")
157:     args = parser.parse_args()
158: 
159:     run_pipeline(...)
167: if __name__ == "__main__":
168:     main()
```

### Lines 151–169 Explanation:
* Standard Python CLI argument parsing and execution entrypoint.

---

## 8. `gui.py`

This is the Tkinter desktop graphical interface. It provides the visual controls, handles user events, and runs three parallel background threads for live streaming.

```python
1: """Audio Cascader — Real-Time GUI (Tkinter).
...
16: import tkinter as tk
17: from tkinter import filedialog, messagebox, ttk
```

### Lines 1–17 Explanation:
* Imports Tkinter, Python's native GUI toolkit. `ttk` provides themed, modern widgets (buttons, progress bars, tabs).

```python
42: class AudioCascaderApp(tk.Tk):
43:     """Main desktop application window."""
44: 
45:     def __init__(self):
46:         super().__init__()
47:         self.title("Audio Cascader — Real-Time S2S Voice Converter (Indian -> American)")
48:         self.geometry("960x780")
49:         self.minsize(800, 650)
50:         self.configure(bg="#121214")
```

### Lines 42–50 (`__init__`) Explanation:
* `AudioCascaderApp` inherits from `tk.Tk` (the root window).
* Sets the window title, default dimensions (960x780 pixels), minimum allowed resize dimensions, and a dark background color (`#121214`).

```python
52:         # Engine objects
53:         self.stt_engine: Optional[STTEngine] = None
54:         self.tts_engine: Optional[TTSEngine] = None
55:         self.capture: Optional[AudioCapture] = None
56:         self.segmenter: Optional[StreamingVADSegmenter] = None
57:         self.player: Optional[StreamingAudioPlayer] = None
58: 
59:         # State flags & queues
60:         self.is_live = False
61:         self.live_stop_event = threading.Event()
62:         self.stt_queue: queue.Queue = queue.Queue(maxsize=20)
63:         self.tts_queue: queue.Queue = queue.Queue(maxsize=20)
64:         self.ui_update_queue: queue.Queue = queue.Queue()
```

### Lines 52–65 Explanation (State & Multithread Queues):
* **`self.live_stop_event`**: A threading signal flag. When set, tells all background worker threads to exit their loops cleanly.
* **`self.stt_queue`**: Receives detected speech audio segments from the VAD to send to Whisper.
* **`self.tts_queue`**: Receives transcribed text clauses from Whisper to send to Pocket TTS.
* **`self.ui_update_queue`**: A thread-safe queue where background workers put status messages, transcript logs, and volume meter values to update the UI.

```python
71:         self.after(100, self._process_ui_queue)
72: 
73:         # Pre-load engines in background thread
74:         threading.Thread(target=self._preload_models_background, daemon=True).start()
```

### Lines 71–74 Explanation:
* **Line 71 (`self.after`)**: Schedules `_process_ui_queue` to run every 100 milliseconds on the GUI thread.
* **Line 74**: Starts a background daemon thread (`_preload_models_background`) to load the heavy AI models into GPU memory immediately upon startup without making the window freeze.

```python
76:     def _init_theme(self):
...
107: 
108:     def _build_ui(self):
...
127:         notebook = ttk.Notebook(self)
...
132:         notebook.add(self.tab_live, text="  🎙️ Live Mic Mode  ")
...
137:         notebook.add(self.tab_file, text="  📁 File Mode  ")
...
147:         self.txt_feed = tk.Text(...)
```

### Lines 76–149 Explanation:
* **`_init_theme`**: Sets up custom dark-mode styling (dark cards, neon-accented buttons, modern typography).
* **`_build_ui`**: Creates the application header, live metrics bar, tab navigation (`ttk.Notebook`), and the bottom scrolling text feed (`txt_feed`).

```python
150:     def _build_live_tab(self):
...
159:         self.combo_mic = ttk.Combobox(...)
163:         self.combo_speaker = ttk.Combobox(...)
...
171:         self.prog_mic = ttk.Progressbar(...)
175:         self.prog_out = ttk.Progressbar(...)
...
182:         self.btn_live_toggle = ttk.Button(..., command=self._toggle_live_conversion)
```

### Lines 150–186 Explanation:
* Builds the controls for live streaming: dropdown menus (`Combobox`) for selecting your microphone and speakers, visual progress bars (`Progressbar`) for live volume meters, and the start/stop conversion button.

```python
187:     def _build_file_tab(self):
...
197:         self.ent_filepath = tk.Entry(...)
200:         btn_browse = ttk.Button(..., command=self._browse_file)
207:         self.btn_convert = ttk.Button(..., command=self._start_file_conversion)
210:         self.btn_play_result = ttk.Button(..., command=self._play_output_file)
```

### Lines 187–212 Explanation:
* Builds the file batch mode controls: text entry box for audio file paths, "Browse..." dialog button, "Convert" button, and "Play Output" button.

```python
213:     def _populate_audio_devices(self):
...
241:     def _preload_models_background(self):
...
257:     def _browse_file(self):
...
266:     def _start_file_conversion(self):
...
287:     def _play_output_file(self):
```

### Lines 213–295 Explanation:
* **`_populate_audio_devices`**: Calls `list_audio_devices()` and populates the microphone and speaker dropdown selectors with Windows device names.
* **`_preload_models_background`**: Loads Whisper Turbo and Pocket TTS in the background, updating the status label once the GPU is ready.
* **`_start_file_conversion`**: Spawns a background worker thread that executes `run_pipeline(f_path)` and updates the UI when finished.

```python
296:     def _toggle_live_conversion(self):
297:         if not self.is_live:
298:             self._start_live()
299:         else:
300:             self._stop_live()
```

### Lines 296–301 Explanation:
* Toggles between starting and stopping live microphone conversion.

```python
302:     def _start_live(self):
303:         self.is_live = True
304:         self.live_stop_event.clear()
305:         self.btn_live_toggle.config(text="⏹️ Stop Live Conversion", style="Danger.TButton")
306:         self.lbl_status.config(text="Live conversion active • Listening for Indian English speech...")
...
313:         self.segmenter = StreamingVADSegmenter(speech_threshold=0.5, min_silence_duration_ms=300, max_speech_duration_s=3.5)
314:         self.capture = AudioCapture(device_index=mic_idx, vu_callback=self._on_mic_vu)
315:         self.player = StreamingAudioPlayer(sample_rate=24000, device_index=spk_idx, vu_callback=self._on_out_vu)
316: 
317:         self.player.start()
318:         self.capture.start()
319: 
320:         # Background worker threads
321:         threading.Thread(target=self._live_vad_worker, daemon=True).start()
322:         threading.Thread(target=self._live_stt_worker, daemon=True).start()
323:         threading.Thread(target=self._live_tts_worker, daemon=True).start()
```

### Lines 302–324 (`_start_live`) Explanation:
* Resets the stop event flag and updates button styles to red ("Stop Live Conversion").
* Instantiates `StreamingVADSegmenter`, `AudioCapture` (mic), and `StreamingAudioPlayer` (speaker).
* Starts audio hardware I/O.
* Launches the three pipeline worker threads: VAD worker, STT worker, and TTS worker.

```python
325:     def _stop_live(self):
326:         self.is_live = False
327:         self.live_stop_event.set()
328:         if self.capture:
329:             self.capture.stop()
330:         if self.player:
331:             self.player.stop()
332: 
333:         self.prog_mic["value"] = 0
334:         self.prog_out["value"] = 0
335:         self.btn_live_toggle.config(text="🎙️ Start Live Conversion", style="Primary.TButton")
336:         self.lbl_status.config(text="Live conversion stopped.")
```

### Lines 325–337 (`_stop_live`) Explanation:
* Signals all worker threads to stop (`live_stop_event.set()`), halts microphone and speaker streams, and resets the visual meters.

```python
346:     def _live_vad_worker(self):
347:         while not self.live_stop_event.is_set():
348:             chunk = self.capture.get_chunk(timeout=0.05)
349:             if chunk is None:
350:                 continue
351:             res = self.segmenter.process_chunk(chunk)
352:             if res is not None:
353:                 speech_audio, start_t, end_t = res
354:                 try:
355:                     self.stt_queue.put_nowait((speech_audio, start_t, end_t))
356:                 except queue.Full:
357:                     pass
```

### Lines 346–358 (`_live_vad_worker`) Explanation:
* Constantly pulls audio blocks from the microphone (`capture.get_chunk`).
* Evaluates each chunk with `segmenter.process_chunk(chunk)`.
* When an utterance ends, immediately places the complete speech audio array into `stt_queue`.

```python
359:     def _live_stt_worker(self):
360:         while not self.live_stop_event.is_set():
361:             try:
362:                 audio_seg, start_t, speech_end_t = self.stt_queue.get(timeout=0.05)
363:             except queue.Empty:
364:                 continue
365: 
366:             t0 = time.perf_counter()
367:             res = self.stt_engine.transcribe(audio_seg, language="en", beam_size=1)
368:             stt_latency = (time.perf_counter() - t0) * 1000.0
369:             text = res.get("text", "").strip()
370: 
371:             cleaned = "".join(c for c in text if c.isalnum() or c.isspace()).strip()
372:             if not cleaned or len(cleaned) < 2:
373:                 continue
374: 
375:             # Log transcription in feed
376:             timestamp = datetime.now().strftime("%H:%M:%S")
377:             self.ui_update_queue.put(("feed", f"[{timestamp}] [Indian English Input]: {text} (STT: {stt_latency:.0f}ms)\n"))
378: 
379:             # Send chunks to TTS queue
380:             chunks = chunk_text(text, max_words=10)
381:             for c in chunks:
382:                 try:
383:                     self.tts_queue.put_nowait((c, speech_end_t, stt_latency))
384:                 except queue.Full:
385:                     pass
```

### Lines 359–386 (`_live_stt_worker`) Explanation:
* Waits for audio segments from `stt_queue`.
* Calls `stt_engine.transcribe()`, measuring latency.
* Filters out empty noise or single-character artifacts.
* Posts the transcribed Indian English text to the GUI feed.
* Splits the text into small clauses with `chunk_text(text, max_words=10)` and queues each clause into `tts_queue`.

```python
387:     def _live_tts_worker(self):
388:         voice_state = self.tts_engine.get_or_create_voice_state()
389:         while not self.live_stop_event.is_set():
390:             try:
391:                 clause, speech_end_t, stt_latency = self.tts_queue.get(timeout=0.05)
392:             except queue.Empty:
393:                 continue
394: 
395:             t0 = time.perf_counter()
396:             first_bit = False
397: 
398:             try:
399:                 for chunk_np in self.tts_engine.synthesize_stream(clause, voice_state):
400:                     if not first_bit:
401:                         first_bit = True
402:                         ttfb = (time.perf_counter() - t0) * 1000.0
403:                         total_g2g = (time.perf_counter() - speech_end_t) * 1000.0
404:                         self.ui_update_queue.put(("metrics", (stt_latency, ttfb, total_g2g)))
405: 
406:                     self.player.push_audio(chunk_np)
407: 
408:                 timestamp = datetime.now().strftime("%H:%M:%S")
409:                 self.ui_update_queue.put(("feed", f"[{timestamp}]  -> [American Accent Output]: {clause}\n"))
410:             except Exception as e:
411:                 logger.error("TTS generation error: %s", e)
```

### Lines 387–412 (`_live_tts_worker`) Explanation:
* Waits for text clauses from `tts_queue`.
* Uses `synthesize_stream(clause)`: As each audio slice is generated, pushes it directly to `self.player.push_audio(chunk_np)`.
* Computes real-time latency statistics:
  - `ttfb`: Time-to-first-byte (time to produce the first slice of sound).
  - `total_g2g`: Total glass-to-glass latency from when you stopped talking into the mic until sound came out of the speaker.
* Logs the American Accent output clause in the feed.

```python
413:     def _process_ui_queue(self):
414:         while not self.ui_update_queue.empty():
415:             msg_type, val = self.ui_update_queue.get_nowait()
416:             if msg_type == "status":
417:                 self.lbl_status.config(text=f"Status: {val}")
418:             elif msg_type == "metrics":
419:                 stt_ms, ttfb_ms, g2g_ms = val
420:                 self.lbl_metrics.config(text=f"STT: {stt_ms:.0f} ms | TTFB: {ttfb_ms:.0f} ms | Glass-to-Glass: {g2g_ms:.0f} ms")
421:             elif msg_type == "feed":
422:                 self.txt_feed.insert(tk.END, val)
423:                 self.txt_feed.see(tk.END)
424:             elif msg_type == "vu_mic":
425:                 self.prog_mic["value"] = val
426:                 ...
439:         self.after(50, self._process_ui_queue)
```

### Lines 413–440 (`_process_ui_queue`) Explanation:
* Runs safely on the Tkinter main UI thread.
* Drains all messages posted by background threads and updates labels, progress bars, and scrolling logs.
* Reschedules itself every 50 milliseconds (`self.after(50, ...)`).

```python
442: def main():
443:     app = AudioCascaderApp()
444:     app.mainloop()
445: 
446: if __name__ == "__main__":
447:     main()
```

### Lines 442–449 Explanation:
* Standard application loop. `app.mainloop()` handles mouse clicks, window resizing, and redraw events until the window is closed.

---

## 9. `pocket_tts/` Overview

The `pocket_tts/` directory is a vendored library containing Kyutai's **Pocket TTS** engine. Instead of a cloud API, this neural network runs locally on your PC.

### Key Components inside `pocket_tts/`:
1. **`pocket_tts/models/tts_model.py` (`TTSModel`)**:
   - `load_model()`: Downloads (or reads from cache) the model architecture configuration (`config/b6369a24.yaml`) and safetensors weights.
   - `get_state_for_audio_prompt(path)`: Encodes a reference audio file into FlowLM key-value cache states (our American accent voice embedding).
   - `generate_audio_stream()`: Takes conditioned voice state and tokenized text, predicts acoustic tokens using FlowLM, and decodes them into a 24 kHz audio stream using the Mimi neural audio codec.
2. **`pocket_tts/models/flow_lm.py` (`FlowLMModel`)**:
   - Autoregressive transformer that models speech flow and converts phoneme sequences into discrete audio tokens.
3. **`pocket_tts/models/mimi.py` (`MimiModel`)**:
   - Neural audio codec consisting of a 1D convolutional SEANet encoder and decoder. It takes latent tokens and reconstructs high-fidelity 24 kHz audio waveforms.
