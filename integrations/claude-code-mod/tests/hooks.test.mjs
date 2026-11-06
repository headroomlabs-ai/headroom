import test from 'node:test';
import assert from 'node:assert/strict';
import { host, press, words, settle, summary, response, detail, SID, OTHER, EPOCH, row } from './host-fixture.mjs';
import { textPages } from '../plugins/headroom-sidebar/hooks/core.mjs';

test('Sidebar leaves Snip command available and opens through its own command', async () => {
  const h = host();
  await h.start();
  const registrations = h.calls.filter(c => c[0] === 'command.register');
  assert.equal(registrations[0][1].name, 'headroom-sidebar');
  let snipCalls = 0;
  const result = await h.dispatch('command.run', { command: 'headroom', args: '' }, async () => {
    snipCalls++;
    return { text: 'Snip opened.' };
  });
  assert.equal(snipCalls, 1);
  assert.equal(result.text, 'Snip opened.');
  await h.dispatch('command.run', { command: 'headroom-sidebar', args: 'close' });
  assert.equal(h.state().open, false);
  await h.dispatch('command.run', { command: 'headroom-sidebar', args: '' });
  assert.equal(h.state().open, true);
});

test('sidebar startup uses a native tool and preserves exact restart guidance during polling', async () => {
  const h = host();
  delete h.env.HEADROOM_MOD_SESSION_ID;
  await h.start();
  let command;
  h.$.tool = { call: async args => { command = args; } };
  h.$.http.fetch = async () => ({ ok: true });
  await press(h, 'start');
  assert.equal(command.tool, 'Bash');
  assert.match(command.command, /^headroom-mod start --proxy-url 'http:\/\/127\.0\.0\.1:\d+'$/);
  assert.match(h.state().startNotice, new RegExp(`headroom-mod run.*--resume ${SID}`));
  await h.advance(5000);
  assert.match(words(await h.pane()), new RegExp(`--resume ${SID}`));
});

test('denied startup is shown as a failure without restart guidance', async () => {
  const h = host();
  delete h.env.HEADROOM_MOD_SESSION_ID;
  await h.start();
  h.$.tool = { call: async () => { throw new Error('Permission denied'); } };
  await press(h, 'start');
  assert.match(h.state().startNotice, /startup failed: Permission denied/);
  assert.doesNotMatch(h.state().startNotice, /Exit Claude/);
});

test('a failed launcher tool result does not claim startup success', async () => {
  const h = host();
  delete h.env.HEADROOM_MOD_SESSION_ID;
  await h.start();
  h.$.tool = { call: async () => ({ isError: true, text: 'headroom-mod: command not found' }) };
  await press(h, 'start');
  assert.match(h.state().startNotice, /startup failed: headroom-mod: command not found/);
  assert.doesNotMatch(h.state().startNotice, /Exit Claude/);
});

const reads = h => h.calls.filter(c => c[0] === 'http.fetch');

test('linked initial connection failure describes unavailable telemetry rather than missing linking', async () => {
  const h = host({ transport: async () => { throw new Error('companion offline'); } });
  await h.start();
  assert.equal(h.state().connection, 'offline');
  assert.doesNotMatch(words(await h.pane()), /require[s]? a linked conversation/);
  assert.match(words(await h.pane()), /companion responds/);
});

test('resuming the original linked session after state reset restores periodic refresh', async () => {
  const h = host();
  await h.start();
  await h.dispatch('command.run', { command: 'clear' }, async () => {
    h.ctl.sid = OTHER;
    await h.$.state.set({ plugin: 'headroom-sidebar', key: 'model' }, undefined);
    return { text: 'cleared' };
  });
  await h.dispatch('command.run', { command: 'resume' }, async () => {
    h.ctl.sid = SID;
    await h.$.state.set({ plugin: 'headroom-sidebar', key: 'model' }, undefined);
    return { text: 'resumed' };
  });
  await h.dispatch('command.run', { command: 'headroom-sidebar', args: '' });
  await h.advance(2);
  assert.equal(h.state().connection, 'live');
  const n = reads(h).length;
  await h.advance(5000);
  assert.equal(reads(h).length, n + 1);
});

test('unlinked conversation has distinct tabs and an exact resume command without reading telemetry', async () => {
  const h = host({ env: { HEADROOM_MOD_SESSION_ID: undefined, HEADROOM_MOD_URL: undefined } });
  await h.start();
  let text = words(await h.pane());
  assert.match(text, /OVERVIEW/);
  assert.match(text, /started without Headroom session linking/);
  assert.match(text, new RegExp(`headroom-mod run --resume ${SID}`));
  await press(h, 'requests');
  text = words(await h.pane());
  assert.match(text, /REQUESTS/);
  assert.match(text, /Request history requires a linked conversation/);
  assert.equal(h.state().tab, 'requests');
  assert.equal(reads(h).length, 0);
  await h.dispatch('ui.close', { id: 'headroom-sidebar' }, async () => ({}));
  assert.equal(h.state().open, false);
});

test('backward chunk navigation returns to the previous chunk last screen', async () => {
  const chunks = ['first chunk\n'.repeat(80), 'second chunk'];
  const h = host({ transport: async url => {
    if (!url.includes('/requests/')) return response(summary());
    const page = Number(new URL(url).searchParams.get('page'));
    return response({ ...detail(), page, pages: 2, text: chunks[page] });
  } });
  await h.start(); await press(h, 'requests'); await press(h, 'inspect-req-1');
  const screens = textPages(chunks[0], 48, 30).length;
  assert.ok(screens > 1);
  for (let i = 0; i < screens; i++) await press(h, 'text-next');
  assert.equal(h.state().detail.page, 1);
  await press(h, 'text-prev');
  assert.equal(h.state().detail.page, 0);
  assert.match(words(await h.pane()), new RegExp(`Screen ${screens}/${screens}`));
  await press(h, 'text-prev');
  assert.match(words(await h.pane()), new RegExp(`Screen ${screens - 1}/${screens}`));
  await press(h, 'text-next'); await press(h, 'text-next');
  assert.equal(h.state().detail.page, 1);
  assert.match(words(await h.pane()), /Screen 1\/1/);
});

test('loads, registers command, opens without stealing focus and shows live scoped numbers', async () => {
  const h = host(); await h.start();
  assert.equal(h.state().connection, 'live');
  assert.equal(h.state().summary.totals.saved, 400);
  assert.equal(h.calls.find(c => c[0] === 'ui.open')[1].focus, undefined);
  assert.match(words(await h.pane()), /400 removed/);
  assert.match(words(await h.pane()), /CLAUDE CONTEXT/);
  assert.ok(reads(h).every(c => c[1].includes(SID) && c[2].method === 'GET' && !c[2].body));
});

test('render on terminal and desktop does no I/O', async () => {
  const h = host(); await h.start(); const n = reads(h).length;
  assert.match(words(await h.pane('terminal')), /HEADROOM/);
  assert.match(words(await h.pane('desktop')), /HEADROOM/);
  assert.equal(reads(h).length, n);
});

test('preserves engine result and skips subagent-triggered refresh', async () => {
  const h = host(); await h.start(); const n = reads(h).length;
  const result = await h.dispatch('turn.complete', { agentId: 'child' }, async () => ({ text: 'original answer' }));
  await h.advance(10);
  assert.equal(result.text, 'original answer'); assert.equal(reads(h).length, n);
});

test('missing or mismatched native session never reads global stats', async () => {
  for (const env of [{ HEADROOM_MOD_SESSION_ID: '' }, { HEADROOM_MOD_SESSION_ID: OTHER }]) {
    const h = host({ env }); await h.start(); assert.equal(reads(h).length, 0); assert.equal(h.state().summary, null);
  }
});

test('invalid endpoint fails closed and does not leave an old body on reload', async () => {
  const h = host(); await h.start(); await press(h, 'requests'); await press(h, 'inspect-req-1');
  assert.equal(h.state().detail.available, true);
  h.env.HEADROOM_MOD_URL = 'https://evil.example'; await h.start();
  assert.equal(h.state().detail, null); assert.equal(h.state().summary, null); assert.equal(h.state().connection, 'setup');
});

test('polling never downloads messages; inspection does so only on explicit action', async () => {
  const h = host(); await h.start(); await h.advance(5001);
  assert.ok(reads(h).every(c => !c[1].includes('/requests/')));
  await press(h, 'requests'); await press(h, 'inspect-req-1');
  assert.equal(h.state().detail.text, 'compressed content');
  assert.ok(reads(h).at(-1)[1].includes('/requests/req-1?side=compressed'));
  await press(h, 'overview'); assert.equal(h.state().detail, null);
});

test('close discards sensitive content and pauses polling', async () => {
  const h = host(); await h.start(); await press(h, 'requests'); await press(h, 'inspect-req-1');
  await h.dispatch('ui.close', { id: 'headroom-sidebar' }, async () => ({})); const n = reads(h).length;
  await h.advance(15000); assert.equal(reads(h).length, n); assert.equal(h.state().detail, null); assert.equal(h.state().open, false);
});

test('session switch drops summary, inspector and late HTTP completion', async () => {
  let resolve;
  const h = host({ transport: () => new Promise(r => { resolve = r; }) });
  await h.start(); h.ctl.sid = OTHER;
  await h.dispatch('command.run', { command: 'clear' });
  resolve(response(summary())); await settle();
  assert.equal(h.state().summary, null); assert.equal(h.state().detail, null); assert.equal(h.state().connection, 'setup');
});

test('proxy restart replaces rather than adds totals and clears inspector', async () => {
  const h = host(); await h.start(); await press(h, 'requests'); await press(h, 'inspect-req-1');
  const changed = summary(SID, OTHER); changed.totals.saved = 50;
  h.ctl.transport = async () => response(changed);
  await h.advance(5000);
  assert.equal(h.state().summary.totals.saved, 50); assert.equal(h.state().detail, null); assert.match(h.state().notice, /restarted/);
});

test('overlapping refreshes never create overlapping HTTP reads', async () => {
  let release;
  const h = host({ transport: () => new Promise(r => { release = r; }) });
  await h.start(); await h.dispatch('turn.complete', {}); await h.advance(10);
  assert.equal(reads(h).length, 1);
  release(response(summary())); await settle(); assert.equal(h.state().connection, 'live');
});

test('inspection starting while polling awaits usage does not mark the proxy offline', async () => {
  const h = host(); await h.start(); await press(h, 'requests');
  let releaseUsage;
  h.$.session.usage = () => new Promise(resolve => { releaseUsage = resolve; });
  await h.dispatch('turn.complete', {}); await h.advance(2);
  let releaseRead;
  h.ctl.transport = () => new Promise(resolve => { releaseRead = resolve; });
  const inspecting = press(h, 'inspect-req-1'); await settle();
  releaseUsage({ context: null }); await settle();
  assert.equal(h.state().connection, 'live');
  releaseRead(response(detail())); await inspecting;
  assert.equal(h.state().detail.available, true);
});

test('timeout reports offline but preserves lock until actual host read completes', async () => {
  let release;
  const h = host({ transport: () => new Promise(r => { release = r; }) });
  await h.start(); await h.advance(16000);
  assert.equal(reads(h).length, 1); assert.equal(h.state().connection, 'offline'); assert.match(h.state().notice, /timed out/);
  release(response(summary())); await settle();
  h.ctl.transport = async () => response(summary()); await h.advance(5000);
  assert.equal(h.state().connection, 'live');
});

test('failed reads label prior metrics stale and erase inspection', async () => {
  const h = host(); await h.start(); await press(h, 'requests'); await press(h, 'inspect-req-1');
  h.ctl.transport = async () => { throw new Error('connection refused'); }; await h.advance(5000);
  assert.equal(h.state().connection, 'offline'); assert.match(h.state().notice, /stale/); assert.equal(h.state().detail, null);
});

test('late inspector response after close cannot repopulate state', async () => {
  const h = host(); await h.start(); await press(h, 'requests');
  let release; h.ctl.transport = () => new Promise(r => { release = r; });
  const pressing = press(h, 'inspect-req-1'); await settle();
  await h.dispatch('ui.close', { id: 'headroom-sidebar', origin: { kind: 'person' } });
  release(response(detail())); await pressing;
  assert.equal(h.state().detail, null); assert.equal(h.state().open, false);
});

test('unrelated panes and commands pass through', async () => {
  const h = host(); await h.start();
  assert.deepEqual(await h.dispatch('ui.render', { component: 'Pane', requestId: 'diff' }, async () => ({ type: 'other' })), { type: 'other' });
  assert.deepEqual(await h.dispatch('command.run', { command: 'unrelated' }, async () => ({ text: 'unchanged' })), { text: 'unchanged' });
});

test('narrow fallback composes with other mods and yields to surveys', async () => {
  const h = host({ placed: false }); await h.start();
  const base = { type: 'Text', children: 'another mod' };
  const e = { component: 'AbovePrompt', props: { hasSurvey: false } };
  assert.match(words(await h.dispatch('ui.render', e, async () => base)), /another mod/);
  assert.match(words(await h.dispatch('ui.render', e, async () => base)), /Widen/);
  assert.deepEqual(await h.dispatch('ui.render', { ...e, props: { hasSurvey: true } }, async () => base), base);
});

test('request list paginates and filter excludes unchanged rows', async () => {
  const s = summary(); s.requests = Array.from({ length: 12 }, (_, i) => ({ ...row(`req-${i}`), saved: i ? 0 : 400 }));
  const h = host({ transport: async () => response(s) }); await h.start(); await press(h, 'requests');
  assert.match(words(await h.pane()), /Page 1\/3/); await press(h, 'older'); assert.match(words(await h.pane()), /Page 2\/3/);
  await press(h, 'filter'); assert.match(words(await h.pane()), /Page 1\/1/);
});

test('session end cancels timers and drops retained summaries', async () => {
  const h = host(); await h.start(); await h.dispatch('session.end'); const n = reads(h).length;
  await h.advance(30000); assert.equal(reads(h).length, n); assert.equal(h.state().summary, null);
});

test('a refused telemetry timer cannot fail a completed model turn', async () => {
  const h = host(); await h.start();
  h.$.clock.after = () => { throw new Error('policy refused timer'); };
  const result = await h.dispatch('turn.complete', {}, async () => ({ text: 'real model answer' }));
  assert.equal(result.text, 'real model answer');
});

test('native clear resets stored state and reopening shows an unlinked notice without reading the old session', async () => {
  const h = host(); await h.start();
  await press(h, 'requests'); await press(h, 'inspect-req-1');
  const n = reads(h).length;
  await h.dispatch('command.run', { command: 'clear' }, async () => {
    h.ctl.sid = OTHER;
    await h.$.state.set({ plugin: 'headroom-sidebar', key: 'model' }, undefined);
    return { text: 'cleared' };
  });
  await h.dispatch('command.run', { command: 'headroom-sidebar', args: '' });
  assert.equal(h.state().connection, 'setup');
  assert.equal(h.state().summary, null);
  assert.equal(h.state().detail, null);
  assert.equal(h.state().open, true);
  assert.equal(h.state().resumeId, OTHER);
  assert.match(words(await h.pane()), /Conversation changed since launch/);
  assert.doesNotMatch(words(await h.pane()), /Claude started without/);
  assert.match(words(await h.pane()), /relaunch/i);
  await h.advance(15000);
  assert.equal(reads(h).length, n);
});
