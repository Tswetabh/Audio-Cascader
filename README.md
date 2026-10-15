# Audio Cascader — Indian English → American Accent S2S Converter

Real-time Speech-to-Speech (S2S) voice converter that takes **Indian-English speech in** and renders **natural American-accented speech out** — preserving words, changing only the accent.

Pipeline: **Faster-Whisper Turbo (STT, GPU) → Punctuation Chunker → Kyutai Pocket TTS (GPU) with cached voice state.**

Two front-ends share the same core:

- **File / batch mode** (`pipeline.py`) — convert `.wav` / `.m4a` / `.mp3` → 24 kHz American-accented `.wav`
- **Desktop GUI** (`gui.py` + `run_gui.bat`) — Tkinter app with **File tab + Live-mic tab** (WASAPI + Silero VAD, streaming playback)

---

## Features

- 🎙️ **Live mic conversion** — 16 kHz / 512-sample capture, Silero VAD segmentation (0.5 threshold, 300 ms silence, 3.5 s max speech), 3 worker threads (VAD → STT → TTS), 24 kHz / 1024-sample ring-buffer playback
- 📁 **File conversion** — auto resample to 16 kHz mono, full-file transcribe, per-clause synthesis, peak-normalized 24 kHz PCM_16 output
- 🧠 **GPU-first** — Whisper Turbo (`int8_float16` on CUDA) + Pocket TTS on CUDA, with CPU fallback; STT VRAM freed before TTS in file mode
- 🗣️ **Persistent voice** — American reference (`references/american accent.wav`) is conditioned once into `embeddings/american_accent.pt` (FlowLM KV-cache state) and reloaded in milliseconds on every run
- ✂️ **Punctuation chunker** — splits on `, . ? ! ; :`, caps at 12 words (file) / 10 words (live) for low TTFB
- 📊 **Latency logging** — STT ms + RTF, first-chunk TTFB, glass-to-glass total, per-segment feed in GUI
- 🔊 **Device pickers + VU meters** for mic / speaker

## Architecture

```text
File mode (pipeline.py: run_pipeline):
  Input file ──► convert_to_wav (16 kHz mono)
    ──► STTEngine.transcribe (en, beam=1, full file) ──► stt.unload()
    ──► chunk_text (punctuation, max_words=12)
    ──► TTSEngine.get_or_create_voice_state (.pt cache)
    ──► synthesize_chunk per clause (sequential)
    ──► concat + normalize(0.90) + sf.write 24 kHz WAV

Live mode (gui.py: _live_*_worker):
  Mic 16k/512 ──► Silero VAD ──► stt_queue
    ──► transcribe (beam=1) ──► chunk_text (max_words=10) ──► tts_queue
    ──► synthesize_stream ──► player.push_audio (per tensor, true streaming)
```

> Note: STT is offline per file/segment — the full transcript is chunked post-hoc. Only TTS *output* streaming (`generate_audio_stream`) is incremental.

## Project Structure

```text
Audio Cascader/
├── pipeline.py                  # File-mode S2S: convert → STT → chunk → TTS → wav
├── gui.py                       # Tkinter app: Live tab + File tab
├── run_gui.bat                  # Double-click launcher (.venv\Scripts\python.exe gui.py)
├── requirements.txt             # Pinned deps (see Minimal install below)
├── core/
│   ├── stt_engine.py            # STTEngine — Faster-Whisper Turbo wrapper
│   ├── tts_engine.py            # TTSEngine — Pocket TTS + .pt voice cache, trim + normalize
│   ├── chunker.py               # chunk_text() punctuation splitter
│   ├── audio_convert.py         # PyAV → 16 kHz mono WAV
│   └── audio_stream.py          # AudioCapture / StreamingVADSegmenter / StreamingAudioPlayer
├── pocket_tts/                 # Vendored Kyutai Pocket TTS engine source
│   ├── models/tts_model.py      # load_model / generate_audio_stream / get_state_for_audio_prompt
│   ├── config/b6369a24.yaml     # FlowLM + Mimi 24 kHz config
│   └── ...
├── references/american accent.wav   # Voice prompt (13 s, stereo 48 kHz)
├── embeddings/american_accent.pt     # Cached voice-conditioning state (~49 MB)
├── weights/faster-whisper-turbo/     # large-v3-turbo-ct2 (model.bin ~1.6 GB)
├── test_inputs/                 # Swetabh_Input*.wav, Anoop_voice_line.wav
└── outputs/                     # pipeline_output_*.wav
```

## Requirements

- **OS:** Windows (WASAPI via `sounddevice`)
- **GPU:** NVIDIA RTX 3050 6 GB Laptop tested; CUDA build required for real-time use (CPU works, slower)
- **Python:** 3.12.5 via `.venv` — system Python alone will not work (missing `faster_whisper` / `ctranslate2`)
- Key pinned versions: `torch/torchaudio 2.6.0+cu124`, `ctranslate2 4.8.2`, `faster-whisper 1.2.1`, `safetensors 0.8.0`, `av 19.0.1`, `soundfile 0.12.1`, `sounddevice 0.5.6`, `silero-vad 6.2.3`, `librosa 0.10.2`, `transformers 4.40.2`

## Installation

```bat
cd "C:\47B\Audio Cascader"

:: 1. Create / activate venv (Python 3.12)
python -m venv .venv
.venv\Scripts\activate

:: 2. Install deps (full freeze; a minimal gpu file is recommended for fresh machines)
pip install -r requirements.txt

:: 3. Weights
:: - STT: place faster-whisper-large-v3-turbo-ct2 under weights\faster-whisper-turbo\
:: - TTS: downloaded on first run via HF hub cache (tts_b6369a24.safetensors + tokenizer;
::   voice-cloning weights may require HF auth)
```

Minimal install if the full freeze conflicts:

```bat
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install faster-whisper ctranslate2 soundfile sounddevice silero-vad av librosa scipy safetensors transformers numpy onnxruntime
```

## Usage

### 1. File mode (CLI)

```bat
.venv\Scripts\python.exe pipeline.py --audio test_inputs\Swetabh_Input_16k.wav
.venv\Scripts\python.exe pipeline.py --audio "path\to\input.m4a" --output outputs\out.wav --device cuda --play
```

Options: `--audio/-a` (required, `.wav/.m4a/.mp3`), `--output/-o`, `--device/-d` (`cuda|cpu`), `--play/-p`.

Output: `outputs/pipeline_output_YYYYMMDD_HHMMSS.wav` (24 kHz mono PCM_16) + console performance summary:

```text
STT Time, RTF | First Chunk TTFB | Total TTS Time | Glass-to-Glass Total
```

### 2. GUI (recommended)

```bat
run_gui.bat
:: or
.venv\Scripts\python.exe gui.py
```

- **🎙️ Live Mic Mode:** pick Microphone + Speaker → *Start Live Conversion* → speak Indian English → hear American accent live. Feed shows `[Indian English Input]` + `[American Accent Output]` with STT/TTFB/G2G metrics and VU meters.
- **📁 File Mode:** Browse → *Convert to American Accent* → *Play Output Audio* on success.

Models are pre-loaded in a background thread at startup.

## Core Modules

| Module | What it does |
|---|---|
| `core/stt_engine.py` | `STTEngine.load()` (cuda `int8_float16` → fallback `cpu/int8`), `transcribe(wav/file, language="en", beam_size=1)` → `{text, segments, language, duration, first_word, first_word_time_ms}`, `unload()` frees VRAM |
| `core/tts_engine.py` | `TTSEngine.load()` (`TTSModel.load_model`), `get_or_create_voice_state()` (load `.pt` via `map_location`, else compute from wav + `torch.save`), `synthesize_stream()` (yields float32 tensors), `synthesize_chunk()` (concat + `trim_leading_silence` + `normalize_audio(0.90)`) |
| `core/chunker.py` | `chunk_text(text, max_words)` — regex `[^delims]+[delims]*`, hard word-count split with injected `,` at cut points |
| `core/audio_convert.py` | `convert_to_wav()` — PyAV resample to 16 kHz mono (skips rewrite if already 16 kHz mono), `soundfile`/`librosa` fallback |
| `core/audio_stream.py` | `AudioCapture` (512-block mic), `StreamingVADSegmenter` (pre-roll buffer, max-speech force-cut), `StreamingAudioPlayer` (jitter-buffered 24 kHz playback), `list_audio_devices()` |

## Voice Cloning

- Prompt: `references/american accent.wav` → derived state `embeddings/american_accent.pt` (~49 MB, torch pickle of FlowLM KV-cache, fp32, stored on CPU, `map_location`'d at load).
- Loaded once per run/session (`copy_state=True` per generation preserves the master).
- Missing `.pt` → recompute + re-save. Missing `.wav` → fallback to built-in `"alba"`.
- Single voice only — no picker/registry. Replacing the `.wav` without deleting the `.pt` leaves a stale cache (no hash/mtime check); delete the `.pt` or call `get_or_create_voice_state(force_refresh=True)` to refresh. Corrupt `.pt` currently raises (no auto-recompute).

## Configuration

Hardcoded defaults (no `.env`):

| Setting | Value | Location |
|---|---|---|
| STT weights | `weights/faster-whisper-turbo` | `core/stt_engine.py` |
| STT lang / beam | `en` / `1`, `condition_on_previous_text=False` | `pipeline.py`, `gui.py` |
| Chunk size | `12` file / `10` live | `pipeline.py`, `gui.py` |
| Ref voice / embedding | `references/american accent.wav` / `embeddings/american_accent.pt` | `core/tts_engine.py` |
| VAD | `0.5 / 300 ms / 3.5 s` | `gui.py` |
| Capture / playback | `16 kHz/512` / `24 kHz/1024` | `core/audio_stream.py` |
| TTS defaults | `temp 0.7, lsd_steps 2`, fp32, `NO_CUDA_GRAPH=1` | `pocket_tts/` |

## Sample Data

| File | Format |
|---|---|
| `test_inputs/Swetabh_Input_16k.wav` | 6.5 s, 16 kHz mono |
| `test_inputs/Swetabh_Input.wav` | 6.5 s, 24 kHz mono |
| `test_inputs/Anoop_voice_line.wav` | 14.1 s, 48 kHz stereo |
| `outputs/pipeline_output_*.wav` | 24 kHz mono PCM_16 |

## Known Limitations

- Whole-file/whole-segment STT before any TTS — long files serialize; 4–5 s completion target is unproven on 30–60 s inputs.
- Per-clause AR restarts can cause prosody seams; live short replies (<2 alnum chars, e.g. "Ok") are dropped by the alnum gate in `gui.py`.
- TTS runs fp32 even on GPU; no quantization.
- `requirements.txt` (285 lines) includes unused stacks — prune for clean repro.
- No test suite, no saved transcripts/logs — `outputs/` holds WAVs only.

See `TECHNICAL_HANDOFF.md` for the full audit, bug list, and experiment plan.

## Acknowledgements

- [Kyutai Pocket TTS](https://github.com/kyutai-labs) (vendored under `pocket_tts/`)
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) / CTranslate2 (`large-v3-turbo`)
- [Silero VAD](https://github.com/snakers4/silero-vad)

> Voice provenance/consent for `references/american accent.wav` is undocumented — establish before any public/external use.
