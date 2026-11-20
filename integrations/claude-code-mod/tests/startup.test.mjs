import test from 'node:test';
import assert from 'node:assert/strict';
import { renderPane } from '../plugins/headroom-sidebar/hooks/render.mjs';
import { initialState } from '../plugins/headroom-sidebar/hooks/core.mjs';
import { host, press, summary, response, settle } from './host-fixture.mjs';

const ui = Object.fromEntries(['Box', 'Text', 'Button'].map(type => [type, props => ({ type, ...props })]));
function buttons(node) {
  if (!node || typeof node !== 'object') return [];
  return [...(node.type === 'Button' ? [node] : []), ...[node.children].flat().flatMap(buttons)];
}
test('setup sidebar offers startup and delegates close to the host', () => {
  let started = false;
  const pane = renderPane(ui, initialState(), { start: () => { started = true; }, tab() {}, refresh() {} });
  const controls = buttons(pane);
  assert.equal(controls.some(b => b.label === 'Close'), false);
  const start = controls.find(b => b.label === 'Start Headroom');
  assert.ok(start);
  start.onPress();
  assert.equal(started, true);
});

test('live sidebar has no startup control', () => {
  const pane = renderPane(ui, { ...initialState(), connection: 'live' }, { tab() {}, refresh() {} });
  assert.equal(buttons(pane).some(b => b.label === 'Start Headroom'), false);
});

test('startup waits for pending telemetry before checking health', async () => {
  const h = host({ transport: async () => response({ ...summary(), compression_enabled: true }) });
  await h.start();
  await h.$.state.set({ key: 'model' }, { ...h.state(), connection: 'offline' });
  let release;
  h.ctl.transport = url => url.endsWith('/health')
    ? Promise.resolve(response({ service: 'headroom-claude-mod' }))
    : new Promise(resolve => { release = resolve; });
  h.$.tool = { call: async args => { assert.equal(args.timeout, 120000); return {}; } };
  await h.advance(5000);
  const starting = press(h, 'start');
  await settle();
  assert.equal(h.calls.some(c => c[0] === 'http.fetch' && c[1].endsWith('/health')), false);
  release(response({ ...summary(), compression_enabled: true }));
  await starting;
  assert.match(h.state().startNotice, /headroom-mod run/);
  assert.equal(h.calls.filter(c => c[0] === 'http.fetch' && c[1].endsWith('/health')).length, 1);
});
