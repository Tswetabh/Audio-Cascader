# Live Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add accurate in-memory live diagnostics, explicit JSON/CSV downloads, and a functional React/Vite diagnostics frontend without changing the working Meet audio route.

**Architecture:** Instrument the existing `LiveSession` at capture, endpoint, STT, TTS, and playback boundaries. Expose snapshots and explicit exports through FastAPI, then consume them from a React/Vite single-page frontend using the existing SSE event stream and polling. The backend remains the source of truth and never autosaves diagnostics.

**Tech Stack:** Python 3.12, FastAPI, unittest, psutil, optional `nvidia-smi` sampling, React, Vite, browser Fetch/EventSource APIs.

**Spec:** `docs/superpowers/specs/2026-10-17-live-diagnostics-design.md`

## Global Constraints

- Diagnostics are memory-only until the user clicks JSON or CSV download.
- JSON is the complete source of truth; CSV is one row per segment.
- Existing CUDA-only inference and VB-CABLE routing remain unchanged.
- Existing live and file-input flows must remain available.
- The frontend is React/Vite; FastAPI serves its production build.

## Review Focus

- A session stopped before any segment: status and downloads must remain valid and empty, not crash.
- A TTS or device error mid-segment: the segment must retain an error and the UI must show it without losing prior records.
- GPU metrics unavailable: CPU/RAM diagnostics must still work and GPU fields must be explicit `null`/unavailable.
- A session with long text and multiple TTS chunks: segment IDs and timestamps must remain ordered and CSV rows must be stable.
- Downloading twice or downloading while running: downloads must be deterministic snapshots and must not create server files.

### Task 1: Define diagnostics data model and pure summaries

**Files:**
- Modify: `app.py` near `SessionRecorder`/`LiveSession`
- Test: `tests/test_app.py`

**Interfaces:**
- Produce `DiagnosticsSession`, `SegmentDiagnostics`, and `SystemSample` data structures with JSON-safe `to_dict()` methods.
- Produce `summarize_diagnostics(snapshot) -> dict[str, Any]` with count, average, p50, and p95 latency values.

- [ ] Write failing tests for empty sessions, ordered segment serialization, and p50/p95 summaries.
- [ ] Run the targeted tests and verify they fail because the data model/functions do not exist.
- [ ] Implement the minimal data classes and summary function without any filesystem writes.
- [ ] Run the targeted tests and verify they pass.
- [ ] Commit with `feat: define live diagnostics data model`.

### Task 2: Instrument live capture, STT, TTS, playback, and system usage

**Files:**
- Modify: `app.py` in `LiveSession`, callbacks, STT/TTS workers, and status helpers
- Modify: `requirements-gpu.txt` to pin `psutil` if required
- Test: `tests/test_app.py`

**Interfaces:**
- `LiveSession.diagnostics_snapshot() -> dict[str, Any]` returns the complete current in-memory snapshot.
- `LiveSession.clear_diagnostics()` resets only the current session state.
- Capture a system sample at a bounded interval, not on every audio callback.

- [ ] Write failing tests that assert segment timing fields, queue/underrun counters, and graceful missing-GPU fields.
- [ ] Run the tests and verify the new assertions fail.
- [ ] Add monotonic timestamps at endpoint, STT start/end, TTS queue/start/first-frame/end, and playback boundaries; convert durations to milliseconds in the snapshot.
- [ ] Add bounded CPU/RAM sampling with `psutil` and best-effort GPU sampling through `nvidia-smi`/CUDA, returning explicit nulls when unavailable.
- [ ] Keep all sampling and records in memory; do not write diagnostics from the worker or stop path.
- [ ] Run targeted tests and the complete Python suite; verify all pass.
- [ ] Commit with `feat: instrument live diagnostics`.

### Task 3: Add diagnostics API and explicit exports

**Files:**
- Modify: `app.py` in `create_web_app`
- Test: `tests/test_app.py`

**Interfaces:**
- `GET /api/diagnostics` returns the current snapshot and derived summary.
- `GET /api/diagnostics/export?format=json` returns a JSON download.
- `GET /api/diagnostics/export?format=csv` returns a CSV download with one row per segment.

- [ ] Write failing FastAPI client tests for empty snapshots, JSON content disposition, CSV headers, and unsupported formats.
- [ ] Run the targeted tests and verify they fail.
- [ ] Implement snapshot/export routes using in-memory serialization only; use `StreamingResponse` or an in-memory response body and never create an output file.
- [ ] Include a stable session ID and export timestamp in JSON and CSV metadata where the format permits.
- [ ] Run web API tests and the complete Python suite; verify all pass.
- [ ] Commit with `feat: expose diagnostics snapshots and downloads`.

### Task 4: Scaffold the React/Vite frontend shell

**Files:**
- Create: `web/package.json`
- Create: `web/package-lock.json` through npm install
- Create: `web/vite.config.js`
- Create: `web/src/main.jsx`
- Create: `web/src/App.jsx`
- Create: `web/src/styles.css`
- Modify: `run_app.bat` only if a documented build step is needed
- Test: `web` build command

**Interfaces:**
- Production build outputs to `web/dist`.
- `App` consumes typed/normalized status, diagnostics, device, and SSE event data from a small frontend API layer.

- [ ] Add the minimal React/Vite package files and a build script without changing FastAPI yet.
- [ ] Run `npm install` and `npm run build`; verify a production bundle is created.
- [ ] Implement the shell with controls, status banner, diagnostics download buttons, and text panes; keep styling functional and data-dense.
- [ ] Run the build again and verify it passes.
- [ ] Commit with `feat: scaffold react diagnostics frontend`.

### Task 5: Connect React to live SSE/status/diagnostics data

**Files:**
- Modify: `web/src/App.jsx`
- Create: `web/src/api.js`
- Modify: `web/src/styles.css`
- Test: `web` build and manual browser smoke test

**Interfaces:**
- `api.js` owns `/api/status`, `/api/devices`, `/api/diagnostics`, start/stop, export URLs, and EventSource handling.
- React renders metric cards, text events, segment timeline/table, system samples, warnings, and download actions.

- [ ] Implement API normalization for missing/null GPU fields and stale SSE reconnects.
- [ ] Render per-segment timing rows with clear labels for capture, STT, TTS, playback, and total latency.
- [ ] Render system usage and audio-device configuration warnings, especially MME versus WASAPI.
- [ ] Add explicit JSON/CSV download buttons that call the backend export endpoints and do not autosave.
- [ ] Build and manually verify start, stop, SSE events, status refresh, empty-session state, and downloads.
- [ ] Commit with `feat: connect live diagnostics dashboard`.

### Task 6: Serve the React build and finish verification

**Files:**
- Modify: `app.py` static/frontend serving in `create_web_app`
- Modify: `README.md` with build/run instructions
- Test: `tests/test_app.py`, `npm run build`, live HTTP smoke test

**Interfaces:**
- `/` serves `web/dist/index.html` when the production build exists.
- `/assets/*` serves Vite assets.
- A clear error or development fallback explains when the React build is missing.

- [ ] Write a web test for the production frontend path and the missing-build behavior.
- [ ] Configure FastAPI static mounting and index serving without breaking `/api/*` routes.
- [ ] Document `npm install`, `npm run build`, and `run_app.bat` usage.
- [ ] Run `npm run build`, all Python unit/integration tests, and the cable/live smoke checks.
- [ ] Manually verify a Meet run: WASAPI headset input, CABLE Input output, diagnostics update, and explicit downloads.
- [ ] Commit with `feat: serve react diagnostics dashboard`.

## Final Verification

- Run: `.venv\Scripts\python.exe -m unittest discover -s tests -v`
- Run: `npm run build` from `web`
- Run: `.venv\Scripts\python.exe tools\cable_check.py`
- Run: `.venv\Scripts\python.exe tools\live_cable_smoke.py`
- Confirm: no diagnostics files appear in `outputs/` unless a user explicitly downloads them.
