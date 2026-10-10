// Actual HTTP transport + shipped hooks + synthetic Headroom records; no Claude runtime.
import assert from 'node:assert/strict';
import { host, press, words } from '../tests/host-fixture.mjs';
const base = process.argv[2];
const h = host({ env: { HEADROOM_MOD_URL: base }, transport: async (url, init) => {
  const response = await fetch(url, init);
  return { status: response.status, ok: response.ok, headers: Object.fromEntries(response.headers), text: await response.text() };
} });
await h.start();
for (let i = 0; i < 300 && h.state().connection !== 'live'; i++) await new Promise(resolve => setTimeout(resolve, 10));
assert.equal(h.state().connection, 'live');
assert.equal(h.state().summary.totals.saved, 400);
assert.equal(h.state().summary.totals.requests, 1);
assert.ok(!JSON.stringify(h.state()).includes('OTHER_SESSION_SECRET'));
await press(h, 'requests');
await press(h, 'inspect-req-1');
assert.equal(h.state().detail.available, true);
assert.match(h.state().detail.text, /compressed tool output/);
await press(h, 'original');
assert.match(h.state().detail.text, /original tool output/);
await press(h, 'diff');
assert.match(h.state().detail.text, /original request/);
await h.dispatch('ui.close', { id: 'headroom-sidebar' }, async () => ({}));
assert.equal(h.state().detail, null);
console.log(JSON.stringify({ result: 'PASS', test: 'real HTTP bridge with synthetic Headroom records',
  checks: ['scoped metrics', 'other-session exclusion', 'original inspection', 'compressed inspection', 'request diff', 'close clears content'],
  note: 'Claude API is a protocol fixture; this is not native Claude or a real model/compression run.' }, null, 2));
