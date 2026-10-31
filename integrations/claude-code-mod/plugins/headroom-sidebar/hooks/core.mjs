// Pure protocol and presentation helpers. No runtime imports or ambient I/O.
export const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
export const safeText = (s, limit = 160) => String(s ?? '').slice(0, limit)
  .replace(/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]/g, '\ufffd');
export const label = (s, limit = 100) => safeText(s, limit).replace(/[\r\n\t]/g, ' ');
export const finite = n => typeof n === 'number' && Number.isFinite(n) && Math.abs(n) <= Number.MAX_SAFE_INTEGER;
export const count = n => finite(n) ? (Math.abs(n) >= 1e6 ? `${+(n / 1e6).toFixed(2)}M` : Math.abs(n) >= 1e3 ? `${+(n / 1e3).toFixed(1)}k` : String(n)) : '—';
export const percent = n => finite(n) ? `${n.toFixed(1)}%` : '—';
export const ms = n => finite(n) ? `${n.toFixed(1)} ms` : '—';
export function localUrl(raw) {
  // Validate the spelling too: URL normalizes octal/integer/hex IPv4 aliases.
  if (typeof raw !== 'string' || !/^http:\/\/(127\.0\.0\.1|localhost|\[::1\])(?::[0-9]{1,5})?\/?$/.test(raw))
    throw new Error('Use HEADROOM_MOD_URL=http://127.0.0.1:8787; remote origins and paths are not supported.');
  const u = new URL(raw);
  const port = raw.match(/\]?:([0-9]+)\/?$/)?.[1] ?? '8787';
  if (+port < 1 || +port > 65535) throw new Error('Invalid local proxy port.');
  return `http://${u.hostname === '[::1]' ? '[::1]' : '127.0.0.1'}:${port}`;
}
export function parseResponse(response, sid, requestId) {
  if (!response?.ok) throw new Error(response?.status === 404
    ? 'Companion or retained request unavailable (HTTP 404). Check headroom-mod doctor.'
    : `Headroom companion returned HTTP ${response?.status ?? 'unknown'}.`);
  if (typeof response.text !== 'string' || response.text.length > 400_000)
    throw new Error('Headroom returned an oversized or invalid response.');
  let data;
  try { data = JSON.parse(response.text); } catch { throw new Error('Headroom returned invalid JSON.'); }
  if (!data || data.schema_version !== 1 || data.session_id !== sid || !UUID.test(data.epoch ?? ''))
    throw new Error('Unsupported protocol or mismatched conversation. No data was displayed.');
  if (requestId !== undefined) {
    if (data.request_id !== requestId || typeof data.available !== 'boolean') throw new Error('Mismatched inspection response.');
  } else if (!Array.isArray(data.requests) || data.requests.length > 100 || !data.totals ||
             data.basis !== 'retained_request_totals_not_unique_context_or_lifetime') {
    throw new Error('Invalid Headroom metrics response.');
  }
  const meta = r => {
    if (!r || typeof r !== 'object' || typeof r.request_id !== 'string') throw new Error('Invalid request metadata.');
    const out = { request_id: label(r.request_id, 128), inspectable_id: r.inspectable_id === true && /^[A-Za-z0-9_.:-]{1,128}$/.test(r.request_id),
      timestamp: label(r.timestamp, 64), provider: label(r.provider, 80), model: label(r.model, 120),
      accounting: ['complete', 'missing', 'inconsistent'].includes(r.accounting) ? r.accounting : 'missing',
      failed: r.failed === true, has_messages: r.has_messages === true,
      transforms: Array.isArray(r.transforms) ? r.transforms.slice(0, 16).map(v => label(v, 100)) : [] };
    for (const key of ['before', 'after', 'saved', 'percent', 'overhead_ms', 'latency_ms', 'cache_read', 'cache_write', 'uncached_input', 'output'])
      out[key] = finite(r[key]) ? r[key] : null;
    return out;
  };
  const base = { schema_version: 1, session_id: sid, epoch: data.epoch };
  if (requestId !== undefined) {
    if (!data.available) return { ...base, request_id: requestId, available: false, reason: safeText(data.reason, 300) };
    if (typeof data.text !== 'string' || data.text.length > 1500 || !['original', 'compressed', 'diff'].includes(data.side) ||
        !Number.isInteger(data.page) || !Number.isInteger(data.pages) || data.pages < 1 || data.page < 0 || data.page >= data.pages ||
        !Number.isInteger(data.message) || !Number.isInteger(data.message_count) || data.message < 0 || data.message_count < 1 || data.message >= data.message_count)
      throw new Error('Invalid inspection page.');
    return { ...base, request_id: requestId, available: true, side: data.side, text: safeText(data.text, 1500),
      page: data.page, pages: data.pages, message: data.message, message_count: data.message_count, truncated: data.truncated === true };
  }
  const totals = {};
  for (const key of ['requests', 'accounted_requests', 'failed_requests', 'unaccounted_requests', 'before', 'after', 'saved', 'percent',
    'output', 'output_accounted_requests', 'average_overhead_ms', 'cache_read', 'cache_write', 'cache_read_percent', 'cache_accounted_requests'])
    totals[key] = finite(data.totals[key]) ? data.totals[key] : null;
  if (!Number.isInteger(totals.requests) || totals.requests < 0) throw new Error('Invalid retained-request count.');
  totals.transforms = Object.fromEntries(Object.entries(data.totals.transforms ?? {}).slice(0, 16)
    .filter(([, n]) => Number.isInteger(n) && n >= 0).map(([k, n]) => [label(k, 100), n]));
  return { ...base, totals, latest: data.latest === null ? null : meta(data.latest),
    compression_enabled: typeof data.compression_enabled === 'boolean' ? data.compression_enabled : null,
    requests: data.requests.map(meta), log_full_messages: data.log_full_messages === true,
    retention: { window_full: data.retention?.window_full === true }, basis: data.basis };
}
export function initialState(owner = 1) {
  return { owner, sessionId: '', baseUrl: '', open: false, band: false, tab: 'overview',
    compressionEnabled: null, timeWindow: 'all', controlNotice: '', controlPending: false, showDetails: false,
    connection: 'setup', notice: 'Launch with headroom-mod run to correlate this conversation.',
    updatedAt: null, context: null, summary: null, detail: null, selection: null,
    listPage: 0, changedOnly: false, textPage: 0 };
}
export function trend(rows) {
  const values = rows.slice(0, 16).reverse().map(r => finite(r.saved) ? Math.max(0, r.saved) : 0);
  const top = Math.max(1, ...values);
  return values.map(n => '▁▂▃▄▅▆▇█'[Math.min(7, Math.floor(n / top * 7))]).join('');
}
export function parseControl(response, sid) {
  if (!response?.ok) throw new Error(`Headroom control returned HTTP ${response?.status ?? 'unknown'}.`);
  if (typeof response.text !== 'string' || response.text.length > 4000) throw new Error('Invalid control response.');
  let data;
  try { data = JSON.parse(response.text); } catch { throw new Error('Invalid control JSON.'); }
  if (!data || data.schema_version !== 1 || data.session_id !== sid || !UUID.test(data.epoch ?? '') || typeof data.compression_enabled !== 'boolean')
    throw new Error('Unsupported control response or mismatched conversation.');
  return { epoch: data.epoch, compressionEnabled: data.compression_enabled };
}
export function progressBar(value, columns = 48) {
  if (!finite(value)) return null;
  const width = Math.max(8, Math.min(24, Math.floor(columns) - 14));
  const filled = Math.round(Math.max(0, Math.min(100, value)) / 100 * width);
  return `${'█'.repeat(filled)}${'░'.repeat(width - filled)} ${percent(value)}`;
}
// Text-only pagination is terminal-width aware. No Markdown/ANSI interpretation.
export function textPages(raw, columns, rows) {
  const width = Math.max(16, Math.min(100, columns - 3));
  const height = Math.max(3, rows - 12);
  const lines = safeText(raw, 5000).split('\n').flatMap(line => {
    const points = Array.from(line.replace(/\t/g, '  '));
    if (!points.length) return [''];
    const chunks = [];
    // Half-width limit remains safe for double-width CJK glyphs and emoji.
    const n = /[^\x20-\x7e]/.test(line) ? Math.max(8, Math.floor(width / 2)) : width;
    for (let i = 0; i < points.length; i += n) chunks.push(points.slice(i, i + n).join(''));
    return chunks;
  });
  const pages = [];
  for (let i = 0; i < lines.length; i += height) pages.push(lines.slice(i, i + height).join('\n'));
  return pages.length ? pages : [''];
}
