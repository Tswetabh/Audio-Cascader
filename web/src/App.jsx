import React, { useState } from 'react';

const emptyDiagnostics = {
  segments: [],
  system_samples: [],
  counters: {},
  summary: {},
};

function Metric({ label, value, unit = '' }) {
  return (
    <article className="metric">
      <span>{label}</span>
      <strong>{value ?? '—'}{value != null ? unit : ''}</strong>
    </article>
  );
}

export default function App() {
  const [diagnostics] = useState(emptyDiagnostics);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">LOCAL AUDIO PIPELINE</p>
          <h1>Audio Cascader</h1>
          <p className="muted">Diagnostics-first live voice conversion</p>
        </div>
        <span className="status-pill">READY</span>
      </header>

      <section className="control-panel panel">
        <div className="panel-heading">
          <div><p className="eyebrow">SESSION CONTROL</p><h2>Live route</h2></div>
          <div className="actions"><button>Start live</button><button className="secondary">Stop</button></div>
        </div>
        <div className="control-grid">
          <label>Microphone<select><option>Loading devices…</option></select></label>
          <label>Output<select><option>Loading devices…</option></select></label>
          <label>STT model<select><option>turbo</option><option>tiny</option></select></label>
        </div>
      </section>

      <section className="metrics-grid">
        <Metric label="STT latency" value={diagnostics.summary.stt_latency_ms?.average} unit=" ms" />
        <Metric label="TTS first frame" value={diagnostics.summary.tts_first_frame_latency_ms?.average} unit=" ms" />
        <Metric label="End-to-end" value={diagnostics.summary.end_to_end_latency_ms?.average} unit=" ms" />
        <Metric label="Input drops" value={diagnostics.counters.input_queue_drops ?? 0} />
        <Metric label="Playback underruns" value={diagnostics.counters.playback_underruns ?? 0} />
      </section>

      <section className="columns">
        <article className="panel transcript"><div className="panel-heading"><h2>Spoken / final STT</h2><span className="muted">0 segments</span></div><div className="empty-state">Waiting for live speech…</div></article>
        <article className="panel transcript converted"><div className="panel-heading"><h2>Converted / sent to TTS</h2><span className="muted">0 chunks</span></div><div className="empty-state">Waiting for converted text…</div></article>
      </section>

      <section className="panel"><div className="panel-heading"><h2>Segment timeline</h2><div className="actions"><button className="secondary">Download JSON</button><button className="secondary">Download CSV</button></div></div><div className="empty-state">No segment diagnostics yet.</div></section>
      <section className="panel"><div className="panel-heading"><h2>System usage</h2><span className="muted">No samples yet</span></div><div className="empty-state">GPU, CPU, memory, and device telemetry will appear here.</div></section>
    </main>
  );
}
