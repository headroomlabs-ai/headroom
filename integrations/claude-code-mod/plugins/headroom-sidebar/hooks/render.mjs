import { count, percent, ms, label, safeText, trend, textPages, progressBar } from './core.mjs';

export function renderPane(ui, state, actions, columns = 48, rows = 30) {
  const { Box, Text, Button } = ui;
  const line = (value, extra = {}) => Text({ ...extra, children: value });
  const button = (key, title, fn, hotkey) => Button({ key, label: title, onPress: fn, ...(hotkey ? { hotkey } : {}) });
  const group = (children, extra = {}) => Box({ flexDirection: 'row', gap: 1, ...extra, children });
  const section = (key, title, body, extra = {}) => Box({ key, flexDirection: 'column', marginTop: 1, ...extra,
    children: [line(title, { dimColor: true }), ...body] });
  const rule = () => line('─'.repeat(Math.max(12, Math.min(48, columns - 4))), { dimColor: true });
  const periodicNotice = state.connection === 'live' && state.notice?.startsWith('Local ·');
  const windowName = { all: 'All retained', '15m': 'Last 15 minutes', '1h': 'Last hour', '24h': 'Last 24 hours' }[state.timeWindow] ?? 'All retained';
  const children = [group([line('HEADROOM', { bold: true }),
    line(state.connection.toUpperCase(), { color: state.connection === 'live' ? 'green' : 'yellow' })], { justifyContent: 'space-between' }),
    group([button('overview', state.tab === 'overview' ? 'Overview ●' : 'Overview', () => actions.tab('overview'), '1'),
      button('requests', state.tab === 'requests' ? 'Requests ●' : 'Requests', () => actions.tab('requests'), '2'),
      button('refresh', 'Refresh', actions.refresh, 'r')], { marginTop: 1 })];
  if (state.connection !== 'live') children.push(button('start', 'Start Headroom', actions.start, 's'));
  if (state.startNotice) children.push(line(safeText(state.startNotice, 500)));
  if (state.notice && !periodicNotice) children.push(section('notice', 'STATUS', [line(safeText(state.notice, 300))]));
  if (state.sessionId && state.connection !== 'setup') {
    const controls = [line(`COMPRESSION ${state.compressionEnabled === false ? 'PAUSED' : state.compressionEnabled === true ? 'ON' : 'UNKNOWN'}`,
      { color: state.compressionEnabled === false ? 'yellow' : 'green', bold: true })];
    if (state.compressionEnabled !== null) {
      controls.push(group([
        button('compression', state.controlPending ? 'Working…' : state.compressionEnabled ? 'Pause' : 'Resume', actions.compression, 'p'),
        button('reset', 'Reset stats', actions.reset),
      ]), group([line('Window', { dimColor: true }), ...['15m', '1h', '24h', 'all'].map(w =>
        button(`window-${w}`, `${w === 'all' ? 'All' : w}${state.timeWindow === w ? ' ●' : ''}`, () => actions.window(w)))], { marginTop: 1 }));
    } else controls.push(line('Upgrade the companion to 0.1.1+ for conversation controls.', { dimColor: true }));
    children.push(Box({ key: 'controls', flexDirection: 'column', marginTop: 1, children: controls }));
  }
  if (state.controlNotice) children.push(Box({ flexDirection: 'column', marginTop: 1,
    children: [line(safeText(state.controlNotice, 300), { color: state.controlNotice.includes('failed') ? 'red' : state.compressionEnabled === false ? 'yellow' : 'green' })] }));
  const data = state.summary;
  if (!data) {
    children.push(line(state.tab === 'requests' ? 'REQUESTS' : 'OVERVIEW', { bold: true }),
      line(state.connection === 'setup'
        ? (state.tab === 'requests' ? 'Request history requires a linked conversation.' : 'Compression metrics require a linked conversation.')
        : 'Telemetry is unavailable until the companion responds.'));
    if (state.connection === 'setup' && state.resumeId) {
      children.push(line('Exit Claude, then relaunch this conversation through the launcher:'),
        line(`headroom-mod run --resume ${state.resumeId}`),
        line('Add --proxy-url with your companion origin before --resume if it uses a custom port.', { dimColor: true }));
    } else children.push(line('Check your companion with headroom-mod doctor, then use Refresh.'));
    children.push(line('1 Overview · 2 Requests · r Refresh · q Close · Tab/Enter selects controls', { dimColor: true }),
      line('Controls require a linked companion. Message capture remains a separate proxy opt-in.', { dimColor: true }));
    return Box({ flexDirection: 'column', paddingX: 1, paddingY: 1, children });
  }
  if (state.tab === 'overview') {
    const latest = data.latest, totals = data.totals;
    const savings = totals.requests === 0 ? 0 : totals.saved;
    const reduction = totals.requests === 0 ? 0 : totals.percent;
    const savingsColor = typeof savings !== 'number' ? 'gray' : savings < 0 ? 'red' : 'green';
    children.push(section('savings', 'SAVINGS', [
      line(`${count(savings)} tokens removed`, { bold: true, color: savingsColor }),
      line(progressBar(reduction, columns - 6) ?? 'Reduction unavailable', { color: savingsColor }),
      line(`${totals.requests} requests · ${windowName}`, { dimColor: true }),
    ], { borderStyle: 'round', borderColor: 'gray', paddingX: 1, paddingY: 1 }));
    if (totals.failed_requests) children.push(line(`${totals.failed_requests} failed requests`, { color: 'yellow' }));
    if (totals.unaccounted_requests) children.push(line(`${totals.unaccounted_requests} requests excluded from savings (failed or missing accounting).`, { color: 'yellow' }));
    const latestLines = [];
    if (latest) {
      latestLines.push(line(`${count(latest.before)} → ${count(latest.after)} tokens`, { bold: true }),
        line(`${count(latest.saved)} removed · ${percent(latest.percent)} reduction`, { dimColor: true }));
      if (latest.failed) latestLines.push(line('Latest request failed; excluded from savings totals.', { color: 'yellow' }));
      if (latest.accounting !== 'complete') latestLines.push(line('Token accounting unavailable/inconsistent.', { color: 'yellow' }));
    } else latestLines.push(line('No requests in this window.', { dimColor: true }));
    children.push(section('latest', 'LATEST REQUEST', latestLines));
    children.push(section('context', 'CLAUDE CONTEXT', [
      line(`${count(state.context?.tokens)} / ${count(state.context?.window)} tokens`, { bold: true }),
      line(progressBar(state.context?.percent, columns) ?? 'Context usage unavailable', { color: state.context?.percent >= 85 ? 'yellow' : 'cyan' }),
    ]));
    if (data.retention?.window_full) children.push(section('retention-warning', 'HISTORY', [line('Older requests may be missing: retention is full.', { color: 'yellow' })]));
    children.push(Box({ marginTop: 1, children: button('details', state.showDetails ? 'Less details ▴' : 'More details ▾', actions.details, 'd') }));
    if (state.showDetails) {
      const details = [line(`${ms(totals.average_overhead_ms)} mean compression overhead`),
        line(`Cache reads ${count(totals.cache_read)} · writes ${count(totals.cache_write)}`),
        line(`${percent(totals.cache_read_percent)} provider cache-read share`),
        line(`Saved/request  ${trend(data.requests)}`)];
      if (latest) details.push(line(`${label(latest.model, 40)} · ${ms(latest.overhead_ms)} latest`));
      const transforms = Object.entries(totals.transforms ?? {}).slice(0, 4);
      if (transforms.length) details.push(line('Transforms (request counts)', { dimColor: true }),
        ...transforms.map(([name, n]) => line(`${label(name, 36)}  ${n}`)));
      details.push(line(`Conversation ${state.sessionId.slice(0, 8)}`, { dimColor: true }),
        line('Includes inherited child requests. Request totals are not unique context or bill savings.', { dimColor: true }));
      children.push(section('details-body', 'PERFORMANCE & ACCOUNTING', details));
    }
  } else if (state.tab === 'requests') {
    const items = state.changedOnly ? data.requests.filter(r => r.saved !== null && r.saved !== 0) : data.requests;
    const perPage = Math.max(1, Math.min(5, Math.floor((rows - 12) / 3)));
    const pages = Math.max(1, Math.ceil(items.length / perPage));
    const page = Math.min(state.listPage, pages - 1);
    children.push(group([button('filter', state.changedOnly ? 'Changed only' : 'All requests', actions.filter),
      button('older', 'Older', () => actions.listPage(Math.min(page + 1, pages - 1))),
      button('newer', 'Newer', () => actions.listPage(Math.max(0, page - 1)))]),
      line(`Page ${page + 1}/${pages} · newest ${data.requests.length} retained requests`, { dimColor: true }));
    for (const r of items.slice(page * perPage, (page + 1) * perPage)) {
      const request = [line(`${count(r.before)} → ${count(r.after)} · ${percent(r.percent)}${r.failed ? ' · FAILED' : ''}`, { bold: true }),
        line(`${label(r.timestamp, 19)}  ${label(r.model, 28)}`, { dimColor: true })];
      if (r.inspectable_id) request.push(button(`inspect-${r.request_id}`, r.has_messages ? 'Review messages' : 'Capture status', () => actions.inspect(r.request_id)));
      children.push(Box({ key: `request-${r.request_id}`, flexDirection: 'column', marginTop: 1, paddingX: 1, borderStyle: 'single', borderColor: 'gray', children: request }));
    }
    if (!items.length) children.push(line('No matching requests in the retained window.'));
    if (!data.log_full_messages) children.push(line('Message capture is off. Metrics still work. --log-messages is an explicit proxy opt-in.'));
  } else {
    const d = state.detail, sel = state.selection;
    children.push(line(`REQUEST ${label(sel?.requestId, 40)}`, { bold: true }),
      group(['original', 'compressed', 'diff'].map(side => button(side, side === 'diff' ? 'Request diff' : side, () => actions.side(side)))));
    if (!d) children.push(line('Loading selected preview…'));
    else if (!d.available) children.push(line(safeText(d.reason, 300)));
    else {
      const display = textPages(d.text, columns, rows);
      const textPage = Math.min(state.textPage, display.length - 1);
      children.push(line(`${d.side} · message ${d.message + 1}/${d.message_count} · chunk ${d.page + 1}/${d.pages}`, { dimColor: true }),
        group([button('message-prev', 'Prev msg', () => actions.message(Math.max(0, d.message - 1))),
          button('message-next', 'Next msg', () => actions.message(Math.min(d.message_count - 1, d.message + 1))),
          button('text-prev', 'Prev', () => actions.moveText(-1, display.length)),
          button('text-next', 'Next', () => actions.moveText(1, display.length))]),
        line(display[textPage]), line(`Screen ${textPage + 1}/${display.length}${d.truncated ? ' · TRUNCATED PREVIEW' : ''}`, { dimColor: true }));
      children.push(line('Original/compressed indices are independent. Diff is request-level.', { dimColor: true }));
    }
  }
  if (periodicNotice) children.push(Box({ flexDirection: 'column', marginTop: 1, children: [rule(), line('Local · refreshes every 5s', { dimColor: true })] }));
  return Box({ flexDirection: 'column', paddingX: 1, paddingY: 1, children });
}
