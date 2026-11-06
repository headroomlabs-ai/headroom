import { UUID, initialState, localUrl, parseResponse, parseControl, safeText, count, percent } from './core.mjs';
import { renderPane } from './render.mjs';

const REF = { plugin: 'headroom-sidebar', key: 'model' };
const PANE = 'headroom-sidebar';
const PREFIX = '/headroom-mod/v1/sessions/';

const read = async $ => (await $.state.get(REF)).value ?? initialState();
const patch = async ($, ctx, changes) => {
  const s = await read($);
  if (s.owner === ctx.owner) await $.state.set(REF, { ...s, ...changes });
};
const quiet = promise => { void promise.catch(() => undefined); };

async function recoverState($, ctx) {
  if ((await $.state.get(REF)).value != null) return;
  // Native /clear and session switching can discard state without re-registering hooks.
  // Only current command events recover it; stale timers never adopt a new session.
  ctx.timer?.cancel(); ctx.scheduled?.cancel(); ctx.inspectRevision++;
  ctx.owner++;
  const sid = await $.env.get('HEADROOM_MOD_SESSION_ID');
  const rawUrl = await $.env.get('HEADROOM_MOD_URL');
  let link = {};
  if (UUID.test(sid ?? '') && rawUrl) {
    try { link = { sessionId: sid, baseUrl: localUrl(rawUrl) }; }
    catch { /* Unsupported origins remain unlinked. */ }
  }
  await $.state.set(REF, { ...initialState(ctx.owner), open: true,
    ...link,
    notice: 'Conversation changed. Relaunch with headroom-mod run to link this conversation.' });
  ctx.timer = $.clock.every(5000, () => quiet(refresh($, ctx)));
}

async function checked($, ctx) {
  const s = await read($);
  if (s.owner !== ctx.owner) { ctx.timer?.cancel(); return null; }
  const actual = await $.session.id();
  if (!UUID.test(s.sessionId) || actual !== s.sessionId) {
    ctx.inspectRevision++;
    await patch($, ctx, { connection: 'setup', summary: null, detail: null, selection: null, compressionEnabled: null, controlNotice: '',
      notice: UUID.test(s.sessionId ?? '')
        ? 'Conversation changed since launch. Resume it through Headroom to restore session linking.'
        : 'Claude started without Headroom session linking. The pane cannot attribute requests to this conversation.',
      resumeId: UUID.test(actual ?? '') ? actual : null });
    return null;
  }
  return s;
}

async function getJson($, ctx, url, sid, requestId) {
  return parseResponse(await getResponse($, ctx, url), sid, requestId);
}

async function getResponse($, ctx, url, init = {}) {
  if (ctx.network) throw new Error('A previous Headroom read is still pending; no overlapping request was started.');
  // The published HttpInit has no abort/timeout member. Do not invent one.
  // A UI deadline does not release the single-flight lock until the HOST read settles.
  const operation = $.http.fetch(url, { method: 'GET', headers: { Accept: 'application/json' }, ...init });
  ctx.network = operation;
  void operation.then(() => { if (ctx.network === operation) ctx.network = null; }, () => { if (ctx.network === operation) ctx.network = null; });
  let deadline;
  const timeout = new Promise((_, reject) => {
    deadline = $.clock.after(4000, () => reject(new Error('Headroom read timed out. A pending host read is not retried concurrently.')));
  });
  try { return await Promise.race([operation, timeout]); }
  finally { deadline?.cancel(); }
}

async function waitForRead($, ctx) {
  const pending = ctx.network;
  if (!pending) return;
  let deadline;
  const timeout = new Promise((_, reject) => {
    deadline = $.clock.after(4000, () => reject(new Error('A pending Headroom read did not finish. Retry when it settles.')));
  });
  try { await Promise.race([pending.catch(() => undefined), timeout]); }
  finally { deadline?.cancel(); }
}

async function refresh($, ctx) {
  if (ctx.busy || ctx.network || ctx.controlling) return;
  ctx.busy = true;
  const startedOwner = ctx.owner;
  try {
    const start = await checked($, ctx);
    if (!start || !start.open) return;
    const revision = ctx.metricsRevision;
    const usage = await $.session.usage(); // no breakdown: never a token-count/model call
    if (ctx.network) return; // Inspection may have started while usage was awaited.
    const data = await getJson($, ctx, `${start.baseUrl}${PREFIX}${start.sessionId}?limit=100&window=${start.timeWindow}`, start.sessionId);
    const current = await checked($, ctx);
    if (!current?.open || current.owner !== start.owner || revision !== ctx.metricsRevision) return;
    const restarted = current.summary && current.summary.epoch !== data.epoch;
    if (restarted) ctx.inspectRevision++;
    await patch($, ctx, { summary: data, context: usage?.context ?? null, connection: 'live',
      compressionEnabled: data.compression_enabled,
      updatedAt: await $.clock.now(),
      notice: restarted ? 'Proxy restarted. Controls and stats reflect its new retained history.' : 'Local · refreshes every 5 seconds while open',
      ...(restarted ? { detail: null, selection: null, tab: 'overview', controlNotice: '' } : {}) });
  } catch (error) {
    if (ctx.owner === startedOwner) await patch($, ctx, { connection: 'offline', detail: null,
      notice: `${safeText(error?.message, 230)} Last metrics, if shown, are stale.` });
  } finally { ctx.busy = false; }
}

function requestRefresh($, ctx) {
  ctx.scheduled?.cancel();
  ctx.scheduled = $.clock.after(1, () => quiet(refresh($, ctx)));
}

async function changeControl($, ctx, kind) {
  if (ctx.controlling) return;
  ctx.controlling = true;
  const owner = ctx.owner;
  try {
    await waitForRead($, ctx);
    if (ctx.owner !== owner) return;
    const start = await checked($, ctx);
    if (!start?.open || start.compressionEnabled === null) return;
    await patch($, ctx, { controlPending: true, controlNotice: kind === 'reset' ? 'Resetting stats…' : 'Changing compression…' });
    const body = kind === 'compression' ? { enabled: !start.compressionEnabled } : {};
    const result = parseControl(await getResponse($, ctx, `${start.baseUrl}${PREFIX}${start.sessionId}/${kind}`, {
      method: 'POST', headers: { Accept: 'application/json', 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    }), start.sessionId);
    const current = await checked($, ctx);
    if (!current?.open || current.owner !== start.owner) return;
    ctx.metricsRevision++; ctx.inspectRevision++;
    await patch($, ctx, { compressionEnabled: result.compressionEnabled, detail: null, selection: null, listPage: 0,
      ...(kind === 'reset' || result.epoch !== current.summary?.epoch ? { summary: null, tab: 'overview' } : {}),
      controlNotice: kind === 'reset' ? 'Stats reset. Request history is retained; totals start from now.'
        : result.compressionEnabled ? 'Compression ON · applies to the next request.' : 'Compression PAUSED · next requests pass through unchanged.',
    });
  } catch (error) {
    if (ctx.owner === owner) await patch($, ctx, { controlNotice: `Control failed: ${safeText(error?.message, 230)} Refresh to confirm proxy state before retrying.` });
  } finally {
    ctx.controlling = false;
    if (ctx.owner === owner) {
      await patch($, ctx, { controlPending: false });
      requestRefresh($, ctx);
    }
  }
}

async function open($, ctx, focus = true) {
  await recoverState($, ctx);
  await patch($, ctx, { open: true, detail: null, selection: null, tab: 'overview' });
  const placed = await $.ui.open({ id: PANE, title: 'Headroom', columns: 48, ...(focus ? { focus: true } : {}) });
  await patch($, ctx, { band: placed.isPlaced === false });
  requestRefresh($, ctx);
}
async function close($, ctx) {
  ctx.inspectRevision++;
  await patch($, ctx, { open: false, band: false, detail: null, selection: null });
  await $.ui.close({ id: PANE });
}

async function inspectRequest($, ctx, selection, lastScreen = false) {
  const rev = ++ctx.inspectRevision;
  const start = await checked($, ctx);
  if (!start?.open) return;
  if (!start.summary?.requests.some(r => r.request_id === selection.requestId && r.inspectable_id) && start.selection?.requestId !== selection.requestId) return;
  await patch($, ctx, { selection, detail: null, tab: 'inspect', textPage: 0 });
  try {
    const q = `?side=${selection.side}&message=${selection.message}&page=${selection.page}`;
    const data = await getJson($, ctx, `${start.baseUrl}${PREFIX}${start.sessionId}/requests/${encodeURIComponent(selection.requestId)}${q}`, start.sessionId, selection.requestId);
    const current = await checked($, ctx);
    if (!current?.open || rev !== ctx.inspectRevision || current.owner !== start.owner) return;
    if (data.epoch !== current.summary?.epoch) throw new Error('Proxy restarted; refresh before inspecting.');
    // Rendering knows the terminal size and clamps this sentinel to the last
    // screen. Prev/Next then continue from that clamped position.
    await patch($, ctx, { detail: data, textPage: lastScreen ? Number.MAX_SAFE_INTEGER : 0 });
  } catch (error) {
    if (rev === ctx.inspectRevision) await patch($, ctx, { detail: { available: false, reason: safeText(error?.message, 280) } });
  }
}

function actions($, ctx) {
  return {
    details: async () => {
      const s = await read($);
      await patch($, ctx, { showDetails: !s.showDetails });
    },
    compression: () => changeControl($, ctx, 'compression'),
    reset: () => changeControl($, ctx, 'reset'),
    window: async timeWindow => {
      if (!['all', '15m', '1h', '24h'].includes(timeWindow)) return;
      ctx.metricsRevision++; ctx.inspectRevision++;
      await patch($, ctx, { timeWindow, summary: null, detail: null, selection: null, listPage: 0, tab: 'overview' });
      requestRefresh($, ctx);
    },
    start: async () => {
      if (ctx.starting) return;
      ctx.starting = true;
      const startOwner = ctx.owner;
      try {
        const s = await read($);
        const owner = s.owner;
        const base = localUrl(s.baseUrl || 'http://127.0.0.1:8787');
        await patch($, ctx, { startNotice: 'Starting Headroom… Approve the launcher command if Claude asks.' });
      const launch = await $.tool.call({ tool: 'Bash', command: `headroom-mod start --proxy-url '${base}'`, timeout: 120000 });
      if (launch?.deny || launch?.isError) throw new Error(launch.deny || launch.text || 'The launcher command failed.');
      if (ctx.owner !== owner) return;
      await waitForRead($, ctx);
      if (ctx.owner !== owner) return;
      const health = await getResponse($, ctx, `${base}/headroom-mod/v1/health`);
        if (!health?.ok) throw new Error('The Headroom companion is unavailable. Check the launcher output and retry.');
        const actual = await $.session.id();
        if (ctx.owner !== owner) return;
        const resume = UUID.test(actual ?? '') ? ` --resume ${actual}` : '';
        await patch($, ctx, { startNotice: `Exit Claude and restart with: headroom-mod run --proxy-url '${base}'${resume}`, resumeId: UUID.test(actual ?? '') ? actual : null });
        requestRefresh($, ctx);
      } catch (error) {
        if (ctx.owner === startOwner) await patch($, ctx, { startNotice: `Headroom startup failed: ${safeText(error?.message, 230)}` });
      } finally { ctx.starting = false; }
    },
    refresh: () => quiet(refresh($, ctx)), close: () => quiet(close($, ctx)),
    tab: tab => { ctx.inspectRevision++; return patch($, ctx, { tab, detail: null, selection: null }); },
    filter: async () => { const s = await read($); await patch($, ctx, { changedOnly: !s.changedOnly, listPage: 0 }); },
    listPage: listPage => patch($, ctx, { listPage }),
    inspect: requestId => inspectRequest($, ctx, { requestId, side: 'compressed', message: 0, page: 0 }),
    side: async side => { const s = await read($); if (s.selection) await inspectRequest($, ctx, { ...s.selection, side, message: 0, page: 0 }); },
    message: async message => { const s = await read($); if (s.selection) await inspectRequest($, ctx, { ...s.selection, message, page: 0 }); },
    moveText: async (delta, screens) => {
      const s = await read($);
      if (!s.detail?.available || !s.selection) return;
      const next = Math.min(s.textPage, screens - 1) + delta;
      if (next >= 0 && next < screens) { await patch($, ctx, { textPage: next }); return; }
      const page = s.detail.page + delta;
      if (page >= 0 && page < s.detail.pages) await inspectRequest($, ctx, { ...s.selection, page }, delta < 0);
    },
  };
}


/** @type {import('claude-code').Register} */
export function register(on) {
  const ctx = { owner: 0, timer: null, scheduled: null, network: null, busy: false, controlling: false, inspectRevision: 0, metricsRevision: 0 };
  on('session.start', async ($, e, next) => {
    const result = await next(e);
    try {
      ctx.timer?.cancel(); ctx.scheduled?.cancel(); ctx.inspectRevision++;
      const previous = await read($);
      ctx.owner = previous.owner + 1;
      let s = initialState(ctx.owner);
      await $.state.set(REF, s);
      await $.command.register({ name: 'headroom-sidebar', description: 'Show Headroom compression, retained requests, and message inspection' });
      const sid = await $.env.get('HEADROOM_MOD_SESSION_ID');
      const rawUrl = await $.env.get('HEADROOM_MOD_URL');
      if (UUID.test(sid ?? '') && rawUrl) {
        try { s = { ...s, sessionId: sid, baseUrl: localUrl(rawUrl), notice: 'Connecting to local Headroom…' }; }
        catch (error) { s.notice = safeText(error?.message, 280); }
      }
      await $.state.set(REF, s);
      if (e.isInteractive) {
        await open($, ctx, false);
        ctx.timer = $.clock.every(5000, () => quiet(refresh($, ctx)));
      }
    } catch (error) {
      // Observability must not turn a successful session start into a failed one.
      await patch($, ctx, { connection: 'setup', notice: safeText(error?.message, 280) }).catch(() => undefined);
    }
    return result;
  });
  on('turn.complete', async ($, e, next) => {
    const result = await next(e);
    try { if (!e.agentId) requestRefresh($, ctx); } catch { /* telemetry policy cannot fail a completed turn */ }
    return result;
  });
  on('session.end', async ($, e, next) => {
    try { return await next(e); } finally {
      try {
        ctx.timer?.cancel(); ctx.scheduled?.cancel(); ctx.inspectRevision++;
        await patch($, ctx, { open: false, summary: null, detail: null, selection: null });
      } catch { /* cleanup must not prevent normal session shutdown */ }
    }
  });
  on('command.run', { command: 'headroom-sidebar' }, async ($, e) => {
    if (e.args?.trim() === 'close') { await close($, ctx); return { text: 'Headroom closed.' }; }
    await open($, ctx);
    return { text: 'Headroom opened. 1 Overview · 2 Requests · r Refresh · s Start Headroom when offline · PgUp/PgDn scroll. Use the pane close control to close.' };
  });
  on('command.run', { command: ['clear', 'resume'] }, async ($, e, next) => {
    const result = await next(e);
    await recoverState($, ctx).catch(() => undefined);
    await checked($, ctx).catch(() => undefined);
    return result;
  });
  on('ui.close', { id: PANE }, async ($, e, next) => {
    const result = await next(e);
    if (result.deny === undefined) {
      ctx.inspectRevision++;
      await patch($, ctx, { open: false, band: false, detail: null, selection: null }).catch(() => undefined);
    }
    return result;
  });
  on('ui.render', { component: 'Pane' }, async ($, e, next) => {
    if (e.requestId !== PANE) return next(e);
    const s = await read($);
    if (!s.open) return next(e);
    return renderPane($.ui.resolve(e), s, actions($, ctx), e.props.bodyColumns ?? 48, e.props.scroll?.bodyRows ?? 30);
  });
  on('ui.render', { component: 'AbovePrompt' }, async ($, e, next) => {
    const s = await read($);
    if (!s.open || !s.band || e.props.hasSurvey) return next(e);
    const { Box, Text } = $.ui.resolve(e);
    const existing = await next(e);
    return Box({ flexDirection: 'column', children: [existing,
      Text({ children: `Headroom ${s.connection}${s.compressionEnabled === false ? ' · PAUSED' : ''} · ${count(s.summary?.latest?.saved)} tokens removed · ${percent(s.summary?.latest?.percent)}. Widen the window and run /headroom-sidebar for the inspector.` })] });
  });
}
