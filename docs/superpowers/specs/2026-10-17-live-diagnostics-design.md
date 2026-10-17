# Live Diagnostics Design

## Intent

Make live Meet testing measurable before spending time on visual polish. The
operator must be able to tell whether a problem comes from microphone capture,
endpointing, STT, TTS, playback, GPU pressure, or device configuration.

## Goals

- Record accurate per-segment timing from captured audio through playback.
- Show current counters and system usage while a session is running.
- Preserve raw diagnostics in memory for the current session only.
- Provide explicit JSON and CSV downloads; never autosave diagnostics.
- Replace the single-file vanilla frontend with a functional React frontend.
- Make MME/WASAPI and device choices visible, including warnings for likely
  problematic selections.

## Non-goals

- This version does not redesign STT or TTS quality beyond exposing evidence.
- This version does not add cloud telemetry, accounts, databases, or background
  uploads.
- This version does not automatically write diagnostics or transcripts to
  `outputs/`.
- Visual polish is secondary to correctness and inspectable data.

## Runtime data contract

The backend keeps one in-memory `DiagnosticsSession` while a live session is
running. It contains:

- session metadata: ID, UTC start/end, model, input/output device descriptors,
  host APIs, sample rates, block sizes;
- counters: captured blocks, input queue drops, TTS queue drops, playback drops,
  playback underruns, segment count, converted chunk count, errors;
- per-segment records: capture start/end, duration, endpoint reason, RMS/peak,
  STT window duration, STT latency, final text, TTS enqueue/start/first-frame/
  finish timestamps, generated audio duration, TTS real-time factor, playback
  start/end and error fields;
- periodic system samples: CPU percent, process RSS, GPU utilization, VRAM used
  and total, and GPU temperature when available;
- derived summary: averages and p50/p95 for STT, TTS first-frame, end-to-end,
  and playback start latency.

Diagnostics remain memory-only. `GET /api/diagnostics` returns the current
snapshot. `GET /api/diagnostics/export?format=json|csv` returns an explicit
download from the current session. CSV contains one row per segment; JSON is
the complete source of truth.

## Frontend

Use React with Vite and serve the production build from FastAPI. The first UI
is data-dense rather than decorative:

- controls and device/model configuration;
- live status and warning banner;
- metric cards for STT, TTS, end-to-end, queue, and underrun values;
- spoken/final and converted text panels;
- per-segment timing table and system usage table;
- JSON and CSV download buttons enabled only when diagnostics exist.

Use the existing SSE stream for event updates and periodic status/diagnostics
fetches. The React client must not create a second persistence layer.

## Constraints

- CUDA-only neural inference remains unchanged.
- Existing VB-CABLE routing remains unchanged.
- Existing live audio and file-input flows remain available.
- No diagnostics file is created unless the user clicks a download button.
- Existing Python unit and GPU integration tests remain green.
