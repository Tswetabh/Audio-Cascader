async function request(path, options) {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `Request failed: ${response.status}`);
  return body;
}

export const getStatus = () => request('/api/status');
export const getDevices = () => request('/api/devices');
export const getDiagnostics = () => request('/api/diagnostics');
export const startLive = (input, output, stt) => request(
  `/api/start?input_device=${encodeURIComponent(input)}&output_device=${encodeURIComponent(output)}&stt_model=${encodeURIComponent(stt)}`,
  { method: 'POST' },
);
export const stopLive = () => request('/api/stop', { method: 'POST' });
export const exportUrl = (format) => `/api/diagnostics/export?format=${format}`;
export const subscribeToEvents = (onEvent, onError) => {
  const source = new EventSource('/api/events');
  source.onmessage = (event) => onEvent(JSON.parse(event.data));
  source.onerror = onError;
  return () => source.close();
};
