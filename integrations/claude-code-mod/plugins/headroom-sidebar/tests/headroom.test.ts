// Native Claude test-kit gate. Run: claude plugin test plugins/headroom-sidebar
// Not executed by Node tests, and never represented as passing without Claude.
import { describe, expect, mock, test } from 'claude-code/testing';

const sid = '11111111-1111-4111-8111-111111111111';
const epoch = '33333333-3333-4333-8333-333333333333';
const payload = {
  schema_version: 1, session_id: sid, epoch,
  basis: 'retained_request_totals_not_unique_context_or_lifetime',
  totals: { requests: 0, accounted_requests: 0, failed_requests: 0, unaccounted_requests: 0, saved: null, transforms: {} },
  latest: null, requests: [], retention: { window_full: false }, log_full_messages: false,
};

describe('headroom', () => {
  test('native controls pause compression, reset stats, select windows and draw bars', async ($, on) => {
    const clock = mock.clock(on);
    mock.env(on, { HEADROOM_MOD_SESSION_ID: sid, HEADROOM_MOD_URL: 'http://127.0.0.1:8787' });
    on('session.start', ($, e) => ({ cwd: e.cwd }));
    on('session.id', () => ({ value: sid }));
    on('session.usage', () => ({ value: { startedAt: 0, rateLimits: [], context: { tokens: 50000, window: 200000, percent: 25 } } }));
    on('command.register', ($, e) => ({ value: { command: e.name } }));
    on('ui.open', () => ({ value: { isPlaced: true } } as any));
    let enabled = true, reset = false;
    const urls: string[] = [];
    on('http.fetch', ($, e) => {
      urls.push(e.url);
      const init = (e as any).init;
      if (e.url.endsWith('/compression')) {
        expect(init.method).toBe('POST');
        enabled = JSON.parse(init.body).enabled;
      }
      if (e.url.endsWith('/reset')) { expect(init.method).toBe('POST'); reset = true; }
      return { value: { status: 200, ok: true, headers: {}, text: JSON.stringify({ ...payload,
        compression_enabled: enabled,
        totals: { ...payload.totals, requests: reset ? 0 : 1, saved: reset ? 0 : 400, percent: reset ? null : 40 },
      }) } };
    });
    await $.session.start({ surface: 'terminal', isInteractive: true, cwd: '/work' } as any);
    await clock.advance(2); await clock.settle();
    const ui = await $.ui.mount({ plugin: 'headroom-sidebar', surface: 'terminal', component: 'Pane', requestId: 'headroom-sidebar',
      props: { bodyColumns: 48, placement: 'dock', scroll: { bodyRows: 40 } } as any });
    expect(await ui.find({ type: 'Text', text: /[█░]+.*40\.0%/ })).toBeDefined();
    expect(await ui.find({ type: 'Text', text: /[█░]+.*25\.0%/ })).toBeDefined();
    expect(await ui.find({ type: 'Text', text: /provider cache-read share/ })).toBeUndefined();
    await ui.press({ key: 'details' }); await clock.settle();
    expect(await ui.find({ type: 'Text', text: /provider cache-read share/ })).toBeDefined();
    await ui.press({ key: 'details' }); await clock.settle();
    await ui.press({ key: 'compression' }); await clock.settle();
    expect(enabled).toBe(false);
    expect(await ui.find({ type: 'Text', text: /COMPRESSION PAUSED/ })).toBeDefined();
    await clock.advance(2); await clock.settle();
    await ui.press({ key: 'compression' }); await clock.settle();
    expect(enabled).toBe(true);
    await clock.advance(2); await clock.settle();
    await ui.press({ key: 'window-1h' }); await clock.advance(2); await clock.settle();
    expect(urls[urls.length - 1]).toContain('window=1h');
    await ui.press({ key: 'reset' }); await clock.advance(2); await clock.settle();
    expect(await ui.find({ type: 'Text', text: /0 requests/ })).toBeDefined();
    ui.unmount();
  });
  test('native startup launches Headroom and renders exact restart guidance', async ($, on) => {
    const clock = mock.clock(on);
    mock.env(on, {});
    on('session.start', ($, e) => ({ cwd: e.cwd }));
    on('session.id', () => ({ value: sid }));
    on('command.register', ($, e) => ({ value: { command: e.name } }));
    on('ui.open', () => ({ value: { isPlaced: true } } as any));
    let command = '';
    on('tool.call', ($, e) => {
      command = (e as any).command;
      return { text: 'Headroom is ready.', result: {} } as any;
    });
    on('http.fetch', () => ({ value: { status: 200, ok: true, headers: {}, text: '{}' } }));
    await $.session.start({ surface: 'terminal', isInteractive: true, cwd: '/work' } as any);
    await clock.advance(2); await clock.settle();
    const ui = await $.ui.mount({ plugin: 'headroom-sidebar', surface: 'terminal', component: 'Pane', requestId: 'headroom-sidebar',
      props: { bodyColumns: 80, placement: 'dock', scroll: { bodyRows: 35 } } } as any);
    await ui.press({ key: 'start' });
    await clock.settle();
    expect(command).toBe("headroom-mod start --proxy-url 'http://127.0.0.1:8787'");
    expect(await ui.find({ type: 'Text', text: new RegExp(`headroom-mod run.*--resume ${sid}`) })).toBeDefined();
    await ui.unmount();
  });
  test('native request buttons inspect all sides and navigate messages', async ($, on) => {
    const clock = mock.clock(on);
    mock.env(on, { HEADROOM_MOD_SESSION_ID: sid, HEADROOM_MOD_URL: 'http://127.0.0.1:8787' });
    on('session.start', ($, e) => ({ cwd: e.cwd }));
    on('session.id', () => ({ value: sid }));
    on('session.usage', () => ({ value: { startedAt: 0, rateLimits: [], context: { tokens: 600, window: 200000, percent: 0.3 } } }));
    on('command.register', ($, e) => ({ value: { command: e.name } }));
    on('ui.open', () => ({ value: { isPlaced: true } } as any));
    const row = { request_id: 'native-req', timestamp: '2026-10-06T17:41:49Z', model: 'native-test',
      before: 1000, after: 600, saved: 400, percent: 40, overhead_ms: 1,
      accounting: 'complete', failed: false, inspectable_id: true, has_messages: true, transforms: [] };
    const urls: string[] = [];
    on('http.fetch', ($, e) => {
      urls.push(e.url);
      const url = new URL(e.url);
      const data = url.pathname.includes('/requests/')
        ? { schema_version: 1, session_id: sid, epoch, request_id: row.request_id,
          available: true, side: url.searchParams.get('side'), message: Number(url.searchParams.get('message')),
          message_count: 2, page: 0, pages: 1, truncated: false, text: 'Native preview fixture' }
        : { ...payload, totals: { ...payload.totals, requests: 1, accounted_requests: 1, saved: 400 },
          latest: row, requests: [row], log_full_messages: true };
      return { value: { status: 200, ok: true, headers: {}, text: JSON.stringify(data) } };
    });
    await $.session.start({ surface: 'terminal', isInteractive: true, cwd: '/work' } as any);
    await clock.advance(2);
    await clock.settle();
    const ui = await $.ui.mount({ plugin: 'headroom-sidebar', surface: 'terminal', component: 'Pane',
      requestId: 'headroom-sidebar', props: { bodyColumns: 80, placement: 'dock', scroll: { bodyRows: 35 } } } as any);
    await ui.press({ key: 'requests' });
    await ui.press({ key: 'inspect-native-req' });
    await clock.settle();
    expect(await ui.find({ type: 'Text', text: /Native preview fixture/ })).toBeDefined();
    for (const side of ['original', 'compressed', 'diff']) {
      await ui.press({ key: side });
      await clock.settle();
      expect(urls[urls.length - 1]).toContain(`side=${side}`);
    }
    await ui.press({ key: 'message-next' });
    await clock.settle();
    expect(urls[urls.length - 1]).toContain('message=1');
    await ui.press({ key: 'message-prev' });
    await clock.settle();
    expect(urls[urls.length - 1]).toContain('message=0');
    await ui.press({ key: 'requests' });
    expect(await ui.find({ type: 'Text', text: /Page 1\/1/ })).toBeDefined();
    await ui.unmount();
  });
  test('native buttons switch unlinked tabs and use host close controls without fetching telemetry', async ($, on) => {
    const clock = mock.clock(on);
    mock.env(on, {});
    on('session.start', ($, e) => ({ cwd: e.cwd }));
    on('session.id', () => ({ value: sid }));
    on('command.register', ($, e) => ({ value: { command: e.name } }));
    on('ui.open', () => ({ value: { isPlaced: true } } as any));
    let reads = 0;
    on('ui.render', { component: 'Pane' }, ($, e) => $.ui.resolve(e).Text({ children: 'Host pane fallback' }));
    on('http.fetch', () => { reads++; throw new Error('Unlinked pane must not fetch'); });
    await $.session.start({ surface: 'terminal', isInteractive: true, cwd: '/work' } as any);
    await clock.advance(2);
    await clock.settle();
    const ui = await $.ui.mount({ plugin: 'headroom-sidebar', surface: 'terminal',
      component: 'Pane', requestId: 'headroom-sidebar',
      props: { bodyColumns: 80, placement: 'dock', scroll: { bodyRows: 35 } } } as any);
    expect(await ui.find({ type: 'Text', text: /OVERVIEW/ })).toBeDefined();
    await ui.press({ key: 'requests' });
    expect(await ui.find({ type: 'Text', text: /REQUESTS/ })).toBeDefined();
    expect(await ui.find({ type: 'Text', text: /Request history requires/ })).toBeDefined();
    await ui.press({ key: 'overview' });
    expect(await ui.find({ type: 'Text', text: /OVERVIEW/ })).toBeDefined();
    await ui.press({ key: 'refresh' });
    await clock.settle();
    expect(reads).toBe(0);
    expect(await ui.find({ key: 'close' })).toBeUndefined();
    expect(await ui.find({ key: 'start' })).toBeDefined();
    await ui.unmount();
  });
  test('native state, command registration and narrow terminal fallback', async ($, on) => {
    const clock = mock.clock(on);
    mock.env(on, { HEADROOM_MOD_SESSION_ID: sid, HEADROOM_MOD_URL: 'http://127.0.0.1:8787' });
    on('session.start', ($, e) => ({ cwd: e.cwd }));
    on('session.id', () => ({ value: sid }));
    on('session.usage', () => ({ value: { startedAt: 0, rateLimits: [], context: { tokens: 600, window: 200000, percent: 0.3 } } }));
    on('command.register', ($, e) => ({ value: { command: e.name } }));
    on('ui.open', () => ({ value: { isPlaced: false } } as any));
    on('http.fetch', () => ({ value: { status: 200, ok: true, headers: {}, text: JSON.stringify(payload) } }));
    on('ui.render', { component: 'AbovePrompt' }, ($, e) => $.ui.resolve(e).Text({ children: 'existing band' }));
    await $.session.start({ surface: 'terminal', isInteractive: true, cwd: '/work' } as any);
    await clock.advance(2);
    await clock.settle();
    const ui = await $.ui.mount({ plugin: 'headroom-sidebar', surface: 'terminal', component: 'AbovePrompt',
      props: { hasSurvey: false, isWorking: false, maxRows: 10, bodyColumns: 100 } } as any);
    expect(await ui.find({ type: 'Text', text: /Headroom live/ })).toBeDefined();
    expect(await ui.find({ type: 'Text', text: /existing band/ })).toBeDefined();
    await ui.unmount();
  });

  test('native pane draws a scoped empty state, not proxy-wide zero savings', async ($, on) => {
    const clock = mock.clock(on);
    mock.env(on, { HEADROOM_MOD_SESSION_ID: sid, HEADROOM_MOD_URL: 'http://127.0.0.1:8787' });
    on('session.start', ($, e) => ({ cwd: e.cwd }));
    on('session.id', () => ({ value: sid }));
    on('session.usage', () => ({ value: { startedAt: 0, rateLimits: [], context: { tokens: 600, window: 200000, percent: 0.3 } } }));
    on('command.register', ($, e) => ({ value: { command: e.name } }));
    on('ui.open', () => ({ value: { isPlaced: true } } as any));
    on('http.fetch', () => ({ value: { status: 200, ok: true, headers: {}, text: JSON.stringify(payload) } }));
    await $.session.start({ surface: 'terminal', isInteractive: true, cwd: '/work' } as any);
    await clock.advance(2);
    await clock.settle();
    const ui = await $.ui.mount({ plugin: 'headroom-sidebar', surface: 'terminal', component: 'Pane', requestId: 'headroom-sidebar',
      props: { bodyColumns: 48, placement: 'dock', scroll: { bodyRows: 35 } } } as any);
    expect(await ui.find({ type: 'Text', text: /No requests in this window/ })).toBeDefined();
    expect(await ui.find({ type: 'Text', text: /0 requests/ })).toBeDefined();
    await ui.unmount();
  });
});
