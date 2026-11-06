// Offline protocol fixture: runs shipped hooks but is NOT the Claude engine.
import { register } from '../plugins/headroom-sidebar/hooks/headroom.mjs';
export const SID = '11111111-1111-4111-8111-111111111111';
export const OTHER = '22222222-2222-4222-8222-222222222222';
export const EPOCH = '33333333-3333-4333-8333-333333333333';
export const row = (id = 'req-1') => ({ request_id: id, timestamp: '2026-10-05T20:00:00Z', model: 'claude-fixture', provider: 'anthropic',
  before: 1000, after: 600, saved: 400, percent: 40, overhead_ms: 8, accounting: 'complete', failed: false,
  cache_read: 200, cache_write: 100, uncached_input: 300, inspectable_id: true, has_messages: true, transforms: ['smart_crusher'] });
export const summary = (sid = SID, epoch = EPOCH) => ({ schema_version: 1, session_id: sid, epoch,
  basis: 'retained_request_totals_not_unique_context_or_lifetime', log_full_messages: true,
  totals: { requests: 1, accounted_requests: 1, failed_requests: 0, unaccounted_requests: 0,
    before: 1000, after: 600, saved: 400, percent: 40, average_overhead_ms: 8, cache_read: 200, cache_write: 100, cache_read_percent: 33.33, transforms: { smart_crusher: 1 } },
  latest: row(), requests: [row()], retention: { window_full: false } });
export const detail = (sid = SID, epoch = EPOCH) => ({ schema_version: 1, session_id: sid, epoch, request_id: 'req-1', available: true,
  side: 'compressed', message: 0, message_count: 1, page: 0, pages: 1, truncated: false, text: 'compressed content' });
export const response = data => ({ status: 200, ok: true, headers: {}, text: JSON.stringify(data) });
export async function settle() { for (let i = 0; i < 80; i++) await Promise.resolve(); }

export function host(options = {}) {
  const hooks = [], states = new Map(), tasks = [], calls = [];
  let time = 0;
  const env = { HEADROOM_MOD_SESSION_ID: SID, HEADROOM_MOD_URL: 'http://127.0.0.1:8787', ...options.env };
  const ctl = { sid: SID, transport: async url => response(url.includes('/requests/') ? detail() : summary()), ...options };
  function on(name, pattern, fn) { if (typeof pattern === 'function') [fn, pattern] = [pattern, null]; hooks.push({ name, pattern, fn }); }
  register(on);
  function match(p, e) { return !p || Object.entries(p).every(([k, v]) => Array.isArray(v) ? v.includes(e[k]) : v === e[k]); }
  async function dispatch(name, e = {}, bottom = async () => ({})) {
    const chain = hooks.filter(h => h.name === name && match(h.pattern, e));
    async function step(i, input) { return i >= chain.length ? bottom(input) : chain[i].fn($, input, next => step(i + 1, next)); }
    return step(0, e);
  }
  const timer = (ms, fn, repeat = false) => { const t = { at: time + ms, ms, fn, repeat, cancelled: false }; tasks.push(t); return { cancel() { t.cancelled = true; } }; };
  const $ = {
    state: { get: async ref => ({ value: structuredClone(states.get(ref.key)) }), set: async (ref, value) => { states.set(ref.key, structuredClone(value)); } },
    env: { get: async name => env[name] },
    session: { id: async () => ctl.sid, usage: async () => ({ context: { tokens: 600, window: 200000, percent: 0.3 } }) },
    clock: { now: async () => time, after: (ms, fn) => timer(ms, fn), every: (ms, fn) => timer(ms, fn, true) },
    command: { register: async spec => { calls.push(['command.register', spec]); return { command: spec.name }; } },
    http: { fetch: async (url, init) => { calls.push(['http.fetch', url, init]); return ctl.transport(url, init); } },
    ui: {
      resolve: () => Object.fromEntries(['Box', 'Text', 'Button'].map(type => [type, props => ({ type, ...props })])),
      open: async args => { calls.push(['ui.open', args]); return { isPlaced: ctl.placed !== false }; },
      close: async args => dispatch('ui.close', { ...args, origin: { kind: 'plugin' } }),
    },
  };
  async function advance(ms) {
    const end = time + ms;
    let steps = 0;
    while (true) {
      const t = tasks.filter(t => !t.cancelled && t.at <= end).sort((a, b) => a.at - b.at)[0];
      if (!t) break;
      if (++steps > 1000) throw new Error('Runaway fixture timer');
      time = t.at;
      if (t.repeat) t.at += t.ms; else t.cancelled = true;
      t.fn();
      await settle();
    }
    time = end;
    await settle();
  }
  return { $, ctl, env, calls, dispatch, advance, state: () => structuredClone(states.get('model')),
    start: async () => { await dispatch('session.start', { cwd: '/fixture', surface: 'terminal', isInteractive: true }); await advance(2); },
    pane: async (surface = 'terminal', columns = 48, rows = 30) => dispatch('ui.render', { component: 'Pane', requestId: 'headroom-sidebar', surface, props: { bodyColumns: columns, scroll: { bodyRows: rows } } }),
  };
}
export function flatten(tree) { if (!tree) return []; return [tree, ...(Array.isArray(tree.children) ? tree.children.flatMap(flatten) : typeof tree.children === 'object' ? flatten(tree.children) : [])]; }
export const words = tree => flatten(tree).filter(n => n.type === 'Text').map(n => n.children).join('\n');
export async function press(h, key) { const b = flatten(await h.pane()).find(n => n.type === 'Button' && n.key === key); if (!b) throw new Error(`No Button ${key}`); await b.onPress(); await settle(); }
