# React Pipeline Website Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the Audio Cascader React website around the existing, correct FastAPI-backed file and live speech-to-speech pipeline.

**Architecture:** Keep `app.py` as the single backend source of truth for audio devices, file conversion, live capture, STT, punctuation chunking, Pocket TTS, voice-state caching, and diagnostics. Rebuild `web/src` as a focused React/Vite client with two equally supported workflows: batch file conversion and live microphone conversion. Vite proxies API calls during development; FastAPI serves `web/dist` in production.

**Tech Stack:** Python 3.12, FastAPI, Faster-Whisper Turbo, Kyutai Pocket TTS, sounddevice/WASAPI, React 19, Vite 7, browser Fetch/EventSource APIs.

**Spec:** `docs/superpowers/specs/2026-10-17-live-diagnostics-design.md`, plus the pipeline contract documented in `README.md` and `docs/TECHNICAL_HANDOFF.md`.

## Global Constraints

- Preserve the pipeline order: audio normalization → Faster-Whisper STT → punctuation chunking → Pocket TTS → 24 kHz output.
- Keep the American reference voice and cached voice-conditioning state as the current backend defaults.
- Keep live capture at 16 kHz/512-sample blocks, VAD endpointing at the existing thresholds, and playback at 24 kHz/1024-sample blocks.
- Keep diagnostics in memory; create JSON/CSV files only when the user explicitly downloads them.
- Do not reintroduce desktop GUI launchers or the obsolete vanilla web frontend.
- The React UI must handle empty sessions, backend-unavailable errors, missing GPU fields, device errors, and mid-segment failures without losing prior results.

## Review Focus

- File upload with WAV, M4A, and MP3 inputs must use `/api/process`, preserve the returned transcript, and play the returned 24 kHz output.
- Edited transcript synthesis must use `/api/synthesize` and must not re-run STT.
- Live start/stop must use selected input/output devices, subscribe to SSE events, and recover cleanly after stop or backend errors.
- A backend restart or missing production build must produce an actionable UI error rather than a blank page.
- The production build must serve through FastAPI without intercepting `/api/*` routes or creating diagnostics artifacts automatically.

---

### Task 1: Restore and verify the backend contract

**Files:**
- Restore: `app.py` from `feature/live-diagnostics`
- Modify: `README.md` to describe React frontend + FastAPI backend, removing deleted desktop-launcher instructions
- Test: `tests/test_app.py`

**Interfaces:**
- Preserve `GET /health`, `/api/status`, `/api/devices`, `/api/diagnostics`, `/api/events`, `/api/output`, and `/api/voice-reference`.
- Preserve `POST /api/process`, `/api/synthesize`, `/api/start`, and `/api/stop`.
- Preserve `create_web_app()` production serving from `web/dist`.

- [ ] Confirm backend imports using `.venv\Scripts\python.exe` and that the restored module exposes `create_web_app`.
- [ ] Run the existing backend unit tests and record any failures before frontend changes.
- [ ] Update stale README commands to use `npm run build` in `web`, then the Python backend entrypoint.
- [ ] Verify `/health` and production-index behavior with the FastAPI test client.

### Task 2: Rebuild the shared React shell and API layer

**Files:**
- Modify: `web/src/App.jsx`
- Modify: `web/src/api.js`
- Modify: `web/src/styles.css`
- Modify: `web/vite.config.js`
- Test: `web` production build

**Interfaces:**
- Normalize API errors into user-readable messages while preserving backend `detail` values.
- Keep one polling loop for status/diagnostics and one reconnect-safe EventSource subscription.
- Represent workflow state explicitly: `idle`, `loading`, `running`, `stopping`, `success`, and `error`.

- [ ] Create the new information hierarchy: app header/status, workflow tabs, primary controls, transcript/output workspace, and expandable diagnostics.
- [ ] Add responsive states for desktop, tablet, and narrow screens without hiding core actions.
- [ ] Add disabled/loading/error states to every network action.
- [ ] Keep API calls in `api.js`; do not embed fetch logic throughout JSX.
- [ ] Run `npm run build` and verify the bundle is generated without warnings that break runtime behavior.

### Task 3: Implement the file-conversion workflow

**Files:**
- Modify: `web/src/App.jsx`
- Modify: `web/src/api.js`
- Modify: `web/src/styles.css`
- Test: browser/manual smoke test plus build

**Interfaces:**
- `processAudio(file)` sends multipart field `upload` to `/api/process`.
- `synthesizeText(text)` sends JSON `{ text }` to `/api/synthesize`.
- `outputUrl(path)` resolves backend output playback URLs.

- [ ] Add drag-and-drop and file-picker input with supported-format guidance.
- [ ] Show local input preview, filename, size, and conversion progress.
- [ ] Render returned transcript as editable text and expose a separate “synthesize edited transcript” action.
- [ ] Render output audio controls, metadata, download action, and clear replacement behavior for a second conversion.
- [ ] Handle empty transcription and backend failures without showing a false success state.

### Task 4: Implement the live microphone workflow

**Files:**
- Modify: `web/src/App.jsx`
- Modify: `web/src/api.js`
- Modify: `web/src/styles.css`
- Test: browser/manual smoke test with `/api/devices`, `/api/start`, `/api/events`, and `/api/stop`

**Interfaces:**
- `getDevices()` populates input/output selectors and identifies host APIs.
- `startLive(input, output, stt)` starts the existing backend live session.
- `subscribeToEvents()` consumes spoken, converted, error, and state events.
- `stopLive()` stops the session and exposes the final recording when present.

- [ ] Add explicit microphone and output selectors with WASAPI preference and visible host API labels.
- [ ] Add start/stop controls with protection against duplicate starts and stops.
- [ ] Show live state, spoken transcript events, converted text events, errors, and final recording playback.
- [ ] Show device warnings for non-WASAPI routes and diagnostics counters for drops/underruns.
- [ ] Ensure changing tabs does not accidentally stop or duplicate an active SSE subscription.

### Task 5: Make diagnostics useful but secondary

**Files:**
- Modify: `web/src/App.jsx`
- Modify: `web/src/styles.css`
- Test: empty-session, active-session, and explicit-download manual checks

**Interfaces:**
- `getDiagnostics()` renders the backend snapshot and derived summary without creating client persistence.
- `exportUrl('json'|'csv')` remains an explicit user-triggered download.

- [ ] Add summary cards for STT latency, TTS first frame, end-to-end latency, queue drops, and playback underruns.
- [ ] Add an ordered per-segment table with text, timing, and error fields.
- [ ] Add CPU/RAM/GPU availability indicators that display unavailable values honestly.
- [ ] Keep diagnostics collapsed or visually secondary until requested.

### Task 6: Verify the full pipeline and finish documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-10-19-react-pipeline-rebuild.md`
- Test: `web` build, Python tests, and local API/browser smoke test

- [ ] Run `npm install` and `npm run build` from `web`.
- [ ] Run `.venv\Scripts\python.exe -m unittest discover -s tests -v`.
- [ ] Run the backend with the project virtual environment and verify `/health`, frontend serving, file conversion request shape, and live start/stop request shape.
- [ ] Verify no desktop launchers are restored and no diagnostics are written without explicit download.
- [ ] Update this plan with completed checks and any runtime limitations observed.
