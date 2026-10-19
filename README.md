# Audio Cascader

Local Indian-English speech-to-speech conversion into an American-accented voice, powered by Faster-Whisper Turbo and Kyutai Pocket TTS.

The current application is a **React/Vite web frontend** backed by a local **FastAPI/Python service**. It supports uploaded-file conversion and live microphone conversion through Windows audio devices and VB-CABLE.

## What it does

Audio Cascader preserves the words recognized from the input and renders them with a cached American reference voice:

```text
Audio input
   |
   +-- File upload --+--> decode/resample --> Faster-Whisper STT
   |                 |                         |
   +-- Live device -+--> endpointing ---------+
                                               v
                                       punctuation chunker
                                               v
                                       Pocket TTS synthesis
                                               v
                                   24 kHz PCM WAV / playback
```

The web interface also exposes pipeline diagnostics, device information, latency measurements, queue drops, playback underruns, CPU/GPU samples, and JSON/CSV downloads.

## Main workflows

### File conversion

1. The browser uploads a WAV, M4A, MP3, or another supported audio file to `POST /api/process`.
2. FastAPI stores the upload temporarily and decodes it with PyAV.
3. Audio is converted to mono 16 kHz float32 samples.
4. Faster-Whisper Turbo transcribes the complete file with English language forcing, greedy decoding, and `int8_float16` CUDA inference.
5. The transcript is split into punctuation-aware pieces, capped at 12 words.
6. Pocket TTS renders each piece with the cached American voice state.
7. The pieces are written as a normalized 24 kHz PCM16 WAV in `outputs/`.
8. The web UI displays the transcript, output audio, and stage timings.

File conversion is synchronous from the browser's perspective: the result appears after transcription, synthesis, and WAV writing finish. The first request can be much slower because the models and voice cache may need to load. The current file path synthesizes chunks sequentially, so a 4-6 second completion time is not guaranteed for long files.

### Live speech-to-speech

1. The browser requests the Windows audio-device list from `GET /api/devices`.
2. The user selects an input device, output device, and STT model.
3. `POST /api/start` warms the CUDA STT/TTS models and opens the streams.
4. `sounddevice` captures 16 kHz mono blocks of 512 samples.
5. The RMS endpoint detector calibrates ambient noise, keeps a 100 ms pre-roll, and ends a segment after silence or a maximum segment duration.
6. Whisper runs incremental windows while speech is active and a final pass when the segment ends.
7. Stable transcript text is punctuation-chunked and queued for TTS.
8. Pocket TTS streams 24 kHz audio into a bounded playback buffer.
9. The selected output device plays the converted audio while SSE events update the transcript and diagnostics in the browser.
10. `POST /api/stop` closes the streams and saves the session recording and metrics in `outputs/`.

Current live defaults:

| Setting | Value |
|---|---:|
| Capture sample rate | 16,000 Hz |
| Capture block size | 512 samples |
| Ambient calibration | 2 seconds |
| Silence endpoint | 500 ms |
| Minimum speech | 300 ms |
| Maximum segment | 5 seconds |
| Pre-roll | 100 ms |
| Playback sample rate | 24,000 Hz |
| Playback block size | 512 samples |
| Playback prefill | 150 ms |

## VB-CABLE / virtual microphone setup

VB-CABLE uses reversed-looking names:

| Device | Direction | Meaning |
|---|---|---|
| `CABLE Input` | Render/output | Audio Cascader writes converted audio here |
| `CABLE Output` | Capture/input | Google Meet or another app reads audio here |

### Send converted speech to Google Meet

In Audio Cascader's **Live Speech-to-Speech** tab:

- **Microphone:** your physical microphone, preferably the Windows WASAPI entry.
- **Output:** `CABLE Input (VB-Audio Virtual Cable)` using Windows WASAPI.

In Google Meet or the destination application:

- **Microphone:** `CABLE Output (VB-Audio Virtual Cable)` using Windows WASAPI.

Audio path:

```text
Physical microphone
   -> Audio Cascader input
   -> STT -> chunker -> Pocket TTS
   -> CABLE Input
   -> CABLE Output
   -> Google Meet microphone
```

### Feed Audio Cascader from another application

If another program is sending audio into VB-CABLE, select `CABLE Output (VB-Audio Virtual Cable)` as Audio Cascader's **Microphone**. Choose physical speakers/headphones as the output if you want to hear the converted result locally, or `CABLE Input` if another application should receive it.

Use the **Windows WASAPI** entries rather than MME when available. The backend adds WASAPI auto-convert settings so the 16 kHz capture and 24 kHz playback streams can open against Windows devices using another mix rate.

The repository includes two hardware-path checks:

```powershell
& ".venv\Scripts\python.exe" tools\cable_check.py
& ".venv\Scripts\python.exe" tools\live_cable_smoke.py
```

The first verifies the CABLE loopback objectively. The second exercises the full live path using a clip instead of a physical microphone.

## Requirements

- Windows with working audio drivers and WASAPI.
- Python 3.12.
- NVIDIA CUDA-capable GPU. The current web backend requires CUDA for neural inference; it does not provide a CPU mode.
- Faster-Whisper Turbo CTranslate2 weights under `weights/faster-whisper-turbo/`.
- The tracked voice reference at `voices/american_accent.wav`.
- Node.js and npm for the React frontend.
- VB-AUDIO VB-CABLE for virtual microphone routing.

The tested machine is an NVIDIA RTX 3050 6 GB laptop. First-run model loading and CUDA warm-up can take substantially longer than subsequent requests.

## Installation

From PowerShell:

```powershell
cd "C:\47B\Audio Cascader"

python -m venv .venv
& ".venv\Scripts\Activate.ps1"

& ".venv\Scripts\python.exe" -m pip install -r requirements-gpu.txt `
  --extra-index-url https://download.pytorch.org/whl/cu124

cd web
npm install
```

The Faster-Whisper model files are intentionally ignored by Git because they are large. Place the CTranslate2 model in:

```text
weights/faster-whisper-turbo/
```

Pocket TTS model weights are downloaded through the configured model loader on first use. The derived voice state is cached locally as:

```text
voices/american_accent.pt
voices/manifest.json
```

## Running the application

### Development mode

Use two PowerShell terminals.

Terminal 1 - FastAPI backend:

```powershell
cd "C:\47B\Audio Cascader"
& ".venv\Scripts\python.exe" app.py
```

Terminal 2 - React/Vite frontend:

```powershell
cd "C:\47B\Audio Cascader\web"
npm install
npm run dev
```

Open the Vite URL, normally `http://127.0.0.1:5173`. The Vite configuration proxies `/api/*` and `/health` to FastAPI on port 8000.

### Production-style local mode

```powershell
cd "C:\47B\Audio Cascader\web"
npm install
npm run build

cd ..
& ".venv\Scripts\python.exe" app.py
```

Open `http://127.0.0.1:8000`. FastAPI serves the compiled React files from `web/dist/`.

## Web API

| Route | Purpose |
|---|---|
| `GET /health` | Backend health check |
| `GET /api/status` | GPU, model, live-session, and queue status |
| `GET /api/devices` | Windows input/output device enumeration |
| `GET /api/voice-reference` | Stream the American reference audio |
| `POST /api/process` | Convert an uploaded audio file |
| `POST /api/synthesize` | Synthesize edited transcript text |
| `POST /api/start` | Start live capture and conversion |
| `POST /api/stop` | Stop live capture and finalize recording |
| `GET /api/events` | Server-sent live transcript/output events |
| `GET /api/diagnostics` | Live and latest batch diagnostics |
| `GET /api/diagnostics/export?format=json` | Download JSON diagnostics |
| `GET /api/diagnostics/export?format=csv` | Download CSV segment diagnostics |
| `GET /api/output?path=...` | Safely stream generated WAV output |

Diagnostics are held in memory while the backend is running. They are not automatically written as separate files; JSON/CSV files are created only when the user explicitly downloads them. Conversion outputs and stopped live-session recordings are written to `outputs/`.

## Directory guide

```text
Audio Cascader/
+-- app.py                    # FastAPI backend and current web pipeline
+-- pipeline.py               # Separate legacy CLI file-conversion pipeline
+-- web/
|   +-- src/App.jsx           # React workflows and UI state
|   +-- src/api.js            # Browser-to-FastAPI API layer
|   +-- src/styles.css        # Dark/light themes and responsive layout
|   +-- src/theme.js          # Persisted theme preference helper
|   +-- test/theme.test.mjs   # Frontend theme tests
|   +-- package.json           # React/Vite scripts and dependencies
|   +-- vite.config.js         # Vite build and FastAPI development proxy
+-- core/                     # Shared engine modules used by the legacy CLI
+-- pocket_tts/               # Vendored Kyutai Pocket TTS implementation
+-- models/                   # Adapter/future model integration area
+-- weights/                  # Local STT model weights; ignored by Git
+-- voices/
|   +-- american_accent.wav   # Tracked reference voice
|   +-- american_accent.pt    # Generated voice-conditioning state
|   +-- manifest.json         # Generated reference hash/cache metadata
+-- samples/                  # Example input audio
+-- outputs/                  # Generated WAVs and metrics; ignored by Git
+-- tests/test_app.py         # Backend unit and API tests
+-- tools/
|   +-- cable_check.py        # VB-CABLE loopback verification
|   +-- live_cable_smoke.py   # Full live-path smoke test
+-- docs/                     # Technical handoff, specs, plans, and patches
```

The old desktop GUI launchers have been removed. The supported interface is the React web application.

## Testing

Frontend tests and production build:

```powershell
cd web
npm test
npm run build
```

Backend tests:

```powershell
& ".venv\Scripts\python.exe" -m unittest tests.test_app -v
```

The GPU integration tests run real model inference and require CUDA, local model weights, and the Pocket TTS model cache:

```powershell
& ".venv\Scripts\python.exe" -m unittest tests.test_integration -v
```

## Troubleshooting

### The UI loads but buttons do nothing

The React dev server and FastAPI backend are separate processes. Confirm both are running:

```powershell
Invoke-WebRequest http://127.0.0.1:8000/health
```

It should return `{"status":"ok"}`. In development, use the Vite URL only after starting FastAPI in the other terminal.

### `CUDA is required`

The current web backend is CUDA-only. Verify the project interpreter sees the GPU:

```powershell
& ".venv\Scripts\python.exe" -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
```

### No devices appear

Check Windows privacy permissions, confirm the microphone/speakers work in another application, install VB-CABLE if virtual routing is needed, and use the Windows WASAPI device entries.

### Live start fails with an invalid sample-rate error

Select a WASAPI input/output pair. The backend uses WASAPI auto-convert for Windows shared-mode endpoints. Avoid mixing an MME input with a WASAPI output unless you are deliberately testing that route.

### Conversion is slow

The first request includes model and CUDA warm-up. File conversion also waits for full-file STT and then synthesizes text chunks sequentially. Use the live workflow for first-audio latency, and use the diagnostics stage timings to identify whether decode, STT, TTS, or output writing dominates.

## Voice and usage notes

Pocket TTS conditions its output on `voices/american_accent.wav`; this is a voice-conditioning reference, not an explicit accent-control parameter. Do not publish or distribute the reference or generated voice cache without confirming the speaker's consent and provenance.

## Acknowledgements

- [Kyutai Pocket TTS](https://github.com/kyutai-labs)
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
- [VB-AUDIO VB-CABLE](https://vb-audio.com/Cable/)
