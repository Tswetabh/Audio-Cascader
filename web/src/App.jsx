import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  exportUrl,
  getDevices,
  getDiagnostics,
  getStatus,
  startLive,
  stopLive,
  subscribeToEvents,
} from './api.js';

const emptyDiagnostics = { segments: [], system_samples: [], counters: {}, summary: {} };

function Metric({ label, value, unit = '', tone = '' }) {
  return <article className={`metric ${tone}`}><span>{label}</span><strong>{value == null ? '—' : `${value}${unit}`}</strong></article>;
}

function average(summary, key) {
  return summary?.[key]?.average == null ? null : summary[key].average;
}

function download(format) {
  const link = document.createElement('a');
  link.href = exportUrl(format);
  link.download = `audio-cascader-diagnostics.${format}`;
  document.body.appendChild(link);
  link.click();
  link.remove();
}

export default function App() {
  const [status, setStatus] = useState({ state: 'ready', stt_model: 'turbo', tts_model: 'pocket_tts' });
  const [diagnostics, setDiagnostics] = useState(emptyDiagnostics);
  const [devices, setDevices] = useState([]);
  const [input, setInput] = useState('');
  const [output, setOutput] = useState('');
  const [stt, setStt] = useState('turbo');
  const [spoken, setSpoken] = useState([]);
  const [converted, setConverted] = useState([]);
  const [error, setError] = useState('');

  const refresh = useCallback(async () => {
    try {
      const [nextStatus, nextDiagnostics] = await Promise.all([getStatus(), getDiagnostics()]);
      setStatus(nextStatus);
      setDiagnostics(nextDiagnostics);
    } catch (cause) {
      setError(cause.message);
    }
  }, []);

  useEffect(() => {
    getDevices().then((nextDevices) => {
      setDevices(nextDevices);
      const inputs = nextDevices.filter((device) => device.inputs);
      const outputs = nextDevices.filter((device) => device.outputs);
      const preferredInput = inputs.find((device) => device.host_api === 'Windows WASAPI') || inputs[0];
      const preferredOutput = outputs.find((device) => device.name.includes('CABLE Input') && device.host_api === 'Windows WASAPI') || outputs[0];
      if (preferredInput) setInput(String(preferredInput.index));
      if (preferredOutput) setOutput(String(preferredOutput.index));
    }).catch((cause) => setError(cause.message));
    refresh();
    const timer = setInterval(refresh, 1000);
    const unsubscribe = subscribeToEvents((event) => {
      if (event.kind === 'spoken' && event.text) setSpoken((items) => [...items, event.text]);
      if (event.kind === 'converted' && event.text) setConverted((items) => [...items, event.text]);
      if (event.kind === 'error') setError(event.error || 'Live worker error');
      if (event.kind === 'state') refresh();
    }, () => undefined);
    return () => { clearInterval(timer); unsubscribe(); };
  }, [refresh]);

  const inputDevice = devices.find((device) => String(device.index) === input);
  const outputDevice = devices.find((device) => String(device.index) === output);
  const warnings = useMemo(() => {
    const values = [];
    if (inputDevice && !inputDevice.host_api.includes('WASAPI')) values.push('Input is using MME; WASAPI is preferred for live capture.');
    if (diagnostics.counters?.input_queue_drops) values.push(`${diagnostics.counters.input_queue_drops} input blocks were dropped.`);
    if (diagnostics.counters?.playback_underruns) values.push(`${diagnostics.counters.playback_underruns} playback underruns detected.`);
    return values;
  }, [diagnostics, inputDevice]);

  async function handleStart() {
    setError(''); setSpoken([]); setConverted([]);
    try { await startLive(input, output, stt); await refresh(); } catch (cause) { setError(cause.message); }
  }

  async function handleStop() {
    try { await stopLive(); await refresh(); } catch (cause) { setError(cause.message); }
  }

  const counters = diagnostics.counters || {};
  const latestSample = diagnostics.system_samples?.at(-1);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div><p className="eyebrow">LOCAL AUDIO PIPELINE</p><h1>Audio Cascader</h1><p className="muted">Diagnostics-first live voice conversion</p></div>
        <span className={`status-pill ${status.state}`}>{String(status.state || 'unknown').toUpperCase()}</span>
      </header>

      {error && <div className="alert error">{error}</div>}
      {warnings.map((warning) => <div className="alert warning" key={warning}>{warning}</div>)}

      <section className="control-panel panel">
        <div className="panel-heading"><div><p className="eyebrow">SESSION CONTROL</p><h2>Live route</h2></div><div className="actions"><button onClick={handleStart}>Start live</button><button className="secondary" onClick={handleStop}>Stop</button></div></div>
        <div className="control-grid">
          <label>Microphone<select value={input} onChange={(event) => setInput(event.target.value)}>{devices.filter((device) => device.inputs).map((device) => <option value={device.index} key={`in-${device.index}`}>{device.index}: {device.name} [{device.host_api}]</option>)}</select></label>
          <label>Output<select value={output} onChange={(event) => setOutput(event.target.value)}>{devices.filter((device) => device.outputs).map((device) => <option value={device.index} key={`out-${device.index}`}>{device.index}: {device.name} [{device.host_api}]</option>)}</select></label>
          <label>STT model<select value={stt} onChange={(event) => setStt(event.target.value)}><option value="turbo">turbo (best)</option><option value="tiny">tiny (fast fallback)</option></select></label>
        </div>
      </section>

      <section className="metrics-grid">
        <Metric label="STT latency · avg" value={average(diagnostics.summary, 'stt_latency_ms')} unit=" ms" />
        <Metric label="STT latency · p95" value={diagnostics.summary?.stt_latency_ms?.p95} unit=" ms" />
        <Metric label="TTS first frame" value={average(diagnostics.summary, 'tts_first_frame_latency_ms')} unit=" ms" />
        <Metric label="End-to-end · p95" value={diagnostics.summary?.end_to_end_latency_ms?.p95} unit=" ms" />
        <Metric label="Input drops" value={counters.input_queue_drops ?? 0} tone={counters.input_queue_drops ? 'bad' : ''} />
        <Metric label="Playback underruns" value={counters.playback_underruns ?? 0} tone={counters.playback_underruns ? 'bad' : ''} />
      </section>

      <section className="columns">
        <article className="panel transcript"><div className="panel-heading"><h2>Spoken / final STT</h2><span className="muted">{spoken.length} events</span></div><div className="lines">{spoken.length ? spoken.map((text, index) => <p key={`${index}-${text}`}>{text}</p>) : <div className="empty-state">Waiting for live speech…</div>}</div></article>
        <article className="panel transcript converted"><div className="panel-heading"><h2>Converted / sent to TTS</h2><span className="muted">{converted.length} chunks</span></div><div className="lines">{converted.length ? converted.map((text, index) => <p key={`${index}-${text}`}>{text}</p>) : <div className="empty-state">Waiting for converted text…</div>}</div></article>
      </section>

      <section className="panel"><div className="panel-heading"><h2>Segment timeline</h2><div className="actions"><button className="secondary" onClick={() => download('json')}>Download JSON</button><button className="secondary" onClick={() => download('csv')}>Download CSV</button></div></div><div className="table-wrap"><table><thead><tr><th>ID</th><th>Audio</th><th>STT</th><th>TTS first</th><th>End-to-end</th><th>Text</th></tr></thead><tbody>{diagnostics.segments?.map((segment) => <tr key={segment.segment_id}><td>{segment.segment_id}</td><td>{Number(segment.audio_duration_s || 0).toFixed(2)} s</td><td>{segment.stt?.latency_ms ?? '—'} ms</td><td>{segment.timings?.tts_first_frame_latency_ms ?? '—'} ms</td><td>{segment.timings?.end_to_end_latency_ms ?? '—'} ms</td><td>{segment.stt?.text || '—'}</td></tr>)}</tbody></table></div>{!diagnostics.segments?.length && <div className="empty-state">No segment diagnostics yet.</div>}</section>

      <section className="panel"><div className="panel-heading"><h2>System and route</h2><span className="muted">{latestSample ? 'latest sample' : 'No samples yet'}</span></div><div className="system-grid"><Metric label="CPU" value={latestSample?.cpu_percent} unit="%" /><Metric label="Process RAM" value={latestSample?.process_rss_mb} unit=" MB" /><Metric label="GPU utilization" value={latestSample?.gpu_utilization_percent} unit="%" /><Metric label="GPU VRAM" value={latestSample?.gpu_vram_used_mb} unit=" MB" /><Metric label="TTS queue" value={counters.tts_queue_depth ?? 0} /><Metric label="Backlog" value={counters.playback_backlog_s} unit=" s" /></div><p className="route-note">Input: {inputDevice ? `${inputDevice.name} [${inputDevice.host_api}]` : '—'} · Output: {outputDevice ? `${outputDevice.name} [${outputDevice.host_api}]` : '—'} · GPU: {status.gpu || '—'}</p></section>
    </main>
  );
}
