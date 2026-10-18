import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  exportUrl,
  getDevices,
  getDiagnostics,
  getStatus,
  outputUrl,
  processAudio,
  startLive,
  stopLive,
  subscribeToEvents,
  synthesizeText,
  voiceReferenceUrl,
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

function DeviceSelect({ label, devices, value, onChange, inputOnly = false }) {
  return <label>{label}<select value={value} onChange={(event) => onChange(event.target.value)}>
    {devices.filter((device) => inputOnly ? device.inputs : device.outputs).map((device) => (
      <option value={device.index} key={`${inputOnly ? 'in' : 'out'}-${device.index}`}>
        {device.index}: {device.name} [{device.host_api}]
      </option>
    ))}
  </select></label>;
}

export default function App() {
  const [mode, setMode] = useState('batch');
  const [status, setStatus] = useState({ state: 'ready', stt_model: 'turbo', tts_model: 'pocket_tts' });
  const [diagnostics, setDiagnostics] = useState(emptyDiagnostics);
  const [devices, setDevices] = useState([]);
  const [input, setInput] = useState('');
  const [output, setOutput] = useState('');
  const [stt, setStt] = useState('turbo');
  const [spoken, setSpoken] = useState([]);
  const [converted, setConverted] = useState([]);
  const [transcript, setTranscript] = useState('');
  const [inputFile, setInputFile] = useState(null);
  const [inputPreview, setInputPreview] = useState('');
  const [outputAudio, setOutputAudio] = useState('');
  const [outputMeta, setOutputMeta] = useState(null);
  const [busy, setBusy] = useState(false);
  const [showDiagnostics, setShowDiagnostics] = useState(true);
  const [showAdvanced, setShowAdvanced] = useState(false);
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
      if (event.kind === 'spoken' && event.text) {
        setSpoken((items) => [...items, event.text]);
        setTranscript(event.text);
      }
      if (event.kind === 'converted' && event.text) setConverted((items) => [...items, event.text]);
      if (event.kind === 'error') setError(event.error || 'Live worker error');
      if (event.kind === 'state') refresh();
    }, () => undefined);
    return () => { clearInterval(timer); unsubscribe(); };
  }, [refresh]);

  useEffect(() => () => { if (inputPreview) URL.revokeObjectURL(inputPreview); }, [inputPreview]);

  const inputDevice = devices.find((device) => String(device.index) === input);
  const outputDevice = devices.find((device) => String(device.index) === output);
  const warnings = useMemo(() => {
    const values = [];
    if (inputDevice && !inputDevice.host_api.includes('WASAPI')) values.push('Input is using MME; WASAPI is preferred for live capture.');
    if (diagnostics.counters?.input_queue_drops) values.push(`${diagnostics.counters.input_queue_drops} input blocks were dropped.`);
    if (diagnostics.counters?.playback_underruns) values.push(`${diagnostics.counters.playback_underruns} playback underruns detected.`);
    return values;
  }, [diagnostics, inputDevice]);

  function selectFile(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    if (inputPreview) URL.revokeObjectURL(inputPreview);
    setInputFile(file);
    setInputPreview(URL.createObjectURL(file));
    setError('');
  }

  async function handleConvert() {
    if (!inputFile) { setError('Choose an audio file first.'); return; }
    setBusy(true); setError('');
    try {
      const result = await processAudio(inputFile);
      setTranscript(result.transcript?.text || '');
      setOutputMeta(result);
      setOutputAudio(outputUrl(result.output));
    } catch (cause) { setError(cause.message); }
    finally { setBusy(false); }
  }

  async function handleSynthesize() {
    if (!transcript.trim()) { setError('Enter or transcribe text before synthesizing.'); return; }
    setBusy(true); setError('');
    try {
      const result = await synthesizeText(transcript);
      setOutputMeta(result);
      setOutputAudio(outputUrl(result.output));
    } catch (cause) { setError(cause.message); }
    finally { setBusy(false); }
  }

  async function handleStart() {
    setError(''); setSpoken([]); setConverted([]); setTranscript('');
    try { await startLive(input, output, stt); await refresh(); } catch (cause) { setError(cause.message); }
  }

  async function handleStop() {
    try {
      const result = await stopLive();
      if (result.recording?.wav) setOutputAudio(outputUrl(result.recording.wav));
      await refresh();
    } catch (cause) { setError(cause.message); }
  }

  const counters = diagnostics.counters || {};
  const latestSample = diagnostics.system_samples?.at(-1);
  const live = status.state === 'running';

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand"><div className="brand-mark">AC</div><div><p className="eyebrow">S2S VOICE & ACCENT STUDIO</p><h1>Audio Cascader</h1><p className="muted">Speech-to-speech workspace</p></div></div>
        <div className="header-status"><span className={`status-dot ${live ? 'live' : 'ready'}`} />{live ? 'Live capture active' : 'Ready'}<span className="cuda-badge">{status.gpu ? 'CUDA ready' : 'CUDA unavailable'}</span><button className="icon-button" title="Settings and diagnostics" onClick={() => setShowAdvanced((value) => !value)}>⚙</button></div>
      </header>

      {error && <div className="alert error">{error}</div>}
      {warnings.map((warning) => <div className="alert warning" key={warning}>{warning}</div>)}

      <nav className="mode-tabs" aria-label="Conversion mode">
        <button className={mode === 'batch' ? 'mode-tab active' : 'mode-tab'} onClick={() => setMode('batch')}>Batch Conversion</button>
        <button className={mode === 'live' ? 'mode-tab active' : 'mode-tab'} onClick={() => setMode('live')}>Live Speech-to-Speech</button>
      </nav>

      <section className="workspace">
        <aside className="panel left-column">
          <div className="section-heading"><div><p className="eyebrow">INPUT & CONFIGURATION</p><h2>{mode === 'batch' ? 'Source audio' : 'Live route'}</h2></div><span className="small-status">{mode === 'batch' ? 'Batch' : live ? 'Listening' : 'Idle'}</span></div>

          {mode === 'batch' ? <>
            <label className="field-label">Input audio<input type="file" accept="audio/*" onChange={selectFile} /></label>
            {inputPreview ? <div className="audio-preview"><span>{inputFile?.name}</span><audio controls src={inputPreview} /></div> : <div className="empty-state compact">Choose a WAV, M4A, MP3, or other supported audio file.</div>}
          </> : <>
            <DeviceSelect label="Microphone" devices={devices} value={input} onChange={setInput} inputOnly />
            <DeviceSelect label="Output" devices={devices} value={output} onChange={setOutput} />
          </>}

          <div className="field-row">
            <label className="field-label">STT engine<select value={stt} onChange={(event) => setStt(event.target.value)}><option value="turbo">Faster-Whisper turbo · best</option><option value="tiny">Faster-Whisper tiny · fast fallback</option></select></label>
            <label className="field-label">Inference<select defaultValue="cuda"><option value="cuda">CUDA</option><option value="auto">Auto</option></select></label>
          </div>

          <div className="voice-block">
            <div className="section-heading compact-heading"><div><p className="eyebrow">TARGET VOICE</p><h2>American accent</h2></div><span className="small-status success">Cached</span></div>
            <p className="muted">Current web pipeline uses the cached American reference voice.</p>
            <audio controls src="/api/voice-reference" className="full-audio" />
          </div>

          <details open={showAdvanced} onToggle={(event) => setShowAdvanced(event.currentTarget.open)} className="advanced">
            <summary>Advanced settings <span className="muted">model and diagnostic controls</span></summary>
            <div className="advanced-body"><label className="field-label">Diagnostics refresh<select defaultValue="1000"><option value="1000">1 second</option><option value="2000">2 seconds</option></select></label><p className="hint">Inference remains on the existing asynchronous workers. These controls do not change model weights or audio routing.</p></div>
          </details>

          {mode === 'batch' ? <button className="primary-action" disabled={busy || !inputFile} onClick={handleConvert}>{busy ? 'Converting…' : 'Convert Speech'}</button> : <div className="live-actions"><button className="primary-action" disabled={live} onClick={handleStart}>Start live</button><button className="secondary-action" disabled={!live} onClick={handleStop}>Stop live</button></div>}
          {mode === 'batch' && <button className="secondary-action" disabled={busy || !transcript.trim()} onClick={handleSynthesize}>Synthesize edited transcript</button>}
        </aside>

        <section className="right-column">
          <article className="panel transcript-panel"><div className="section-heading"><div><p className="eyebrow">TRANSCRIPTION</p><h2>Editable transcript</h2></div><span className="small-status">{spoken.length} live events</span></div><textarea value={transcript} onChange={(event) => setTranscript(event.target.value)} placeholder="Transcribed speech will appear here. Edit it, then synthesize the revised text." /><div className="transcript-history">{spoken.length ? spoken.map((text, index) => <span key={`${index}-${text}`}>{text}</span>) : <span className="muted">No live transcript yet.</span>}</div></article>
          <article className="panel output-panel"><div className="section-heading"><div><p className="eyebrow">GENERATED AUDIO</p><h2>Output result</h2></div><span className={`small-status ${outputAudio ? 'success' : ''}`}>{outputAudio ? 'Ready to play' : 'Awaiting conversion'}</span></div>{outputAudio ? <><audio controls src={outputAudio} className="full-audio output-audio" /><div className="output-meta"><span>{outputMeta?.output?.split(/[\\/]/).pop() || 'Live recording'}</span><span>{outputMeta?.duration_s ? `${outputMeta.duration_s}s` : 'Browser playback'}</span></div></> : <div className="empty-state">Convert speech to preview the generated voice here.</div>}</article>
          <article className="panel converted-panel"><div className="section-heading"><div><p className="eyebrow">LIVE OUTPUT</p><h2>Converted speech</h2></div><span className="small-status">{converted.length} chunks</span></div><div className="lines">{converted.length ? converted.map((text, index) => <p key={`${index}-${text}`}>{text}</p>) : <div className="empty-state compact">Live converted text will appear here.</div>}</div></article>
        </section>
      </section>

      <details className="diagnostics panel" open={showDiagnostics} onToggle={(event) => setShowDiagnostics(event.currentTarget.open)}><summary><span><p className="eyebrow">DIAGNOSTICS</p><h2>Pipeline telemetry & benchmark data</h2></span><span className="summary-actions"><button className="secondary-action" onClick={(event) => { event.preventDefault(); download('json'); }}>Download JSON</button><button className="secondary-action" onClick={(event) => { event.preventDefault(); download('csv'); }}>Download CSV</button></span></summary><div className="diagnostic-content"><section className="metrics-grid"><Metric label="STT latency · avg" value={average(diagnostics.summary, 'stt_latency_ms')} unit=" ms" /><Metric label="STT latency · p95" value={diagnostics.summary?.stt_latency_ms?.p95} unit=" ms" /><Metric label="TTS first frame" value={average(diagnostics.summary, 'tts_first_frame_latency_ms')} unit=" ms" /><Metric label="End-to-end · p95" value={diagnostics.summary?.end_to_end_latency_ms?.p95} unit=" ms" /><Metric label="Input drops" value={counters.input_queue_drops ?? 0} tone={counters.input_queue_drops ? 'bad' : ''} /><Metric label="Playback underruns" value={counters.playback_underruns ?? 0} tone={counters.playback_underruns ? 'bad' : ''} /></section><div className="table-wrap"><table><thead><tr><th>ID</th><th>Audio</th><th>STT</th><th>TTS first</th><th>End-to-end</th><th>Text</th></tr></thead><tbody>{diagnostics.segments?.map((segment) => <tr key={segment.segment_id}><td>{segment.segment_id}</td><td>{Number(segment.audio_duration_s || 0).toFixed(2)} s</td><td>{segment.stt?.latency_ms ?? '—'} ms</td><td>{segment.timings?.tts_first_frame_latency_ms ?? '—'} ms</td><td>{segment.timings?.end_to_end_latency_ms ?? '—'} ms</td><td>{segment.stt?.text || '—'}</td></tr>)}</tbody></table></div><div className="system-grid"><Metric label="CPU" value={latestSample?.cpu_percent} unit="%" /><Metric label="Process RAM" value={latestSample?.process_rss_mb} unit=" MB" /><Metric label="GPU utilization" value={latestSample?.gpu_utilization_percent} unit="%" /><Metric label="GPU VRAM" value={latestSample?.gpu_vram_used_mb} unit=" MB" /><Metric label="TTS queue" value={counters.tts_queue_depth ?? 0} /><Metric label="Backlog" value={counters.playback_backlog_s} unit=" s" /></div><p className="route-note">Input: {inputDevice ? `${inputDevice.name} [${inputDevice.host_api}]` : '—'} · Output: {outputDevice ? `${outputDevice.name} [${outputDevice.host_api}]` : '—'} · GPU: {status.gpu || '—'}</p></div></details>
    </main>
  );
}
