// Shipped hooks over actual HTTP to actual proxy records. Host API fixture, not Claude.
import assert from 'node:assert/strict';
import { host, press, words, SID, OTHER } from '../tests/host-fixture.mjs';

const [base, requestId, saved, capture] = process.argv.slice(2);
const transport = async (url, init) => {
  const response = await fetch(url, init);
  return { status: response.status, ok: response.ok, headers: Object.fromEntries(response.headers), text: await response.text() };
};
const h = host({ env: { HEADROOM_MOD_URL: base }, transport });
await h.start();
for (let i = 0; i < 300 && h.state().connection !== 'live'; i++) await new Promise(resolve => setTimeout(resolve, 10));
assert.equal(h.state().connection, 'live');
assert.equal(h.state().summary.session_id, SID);
assert.equal(h.state().summary.totals.saved, Number(saved));
assert.equal(h.state().summary.totals.requests, 1);
assert.ok(!JSON.stringify(h.state()).includes('OTHER_SESSION_SECRET'));
await press(h, 'requests');
await press(h, `inspect-${requestId}`);
if (capture === 'true') {
  assert.equal(h.state().detail.available, true);
  assert.ok(h.state().detail.text.length > 0);
  await press(h, 'original');
  await press(h, 'message-next');
  await press(h, 'message-next');
  assert.match(h.state().detail.text, /orders/);
  await press(h, 'diff');
  assert.match(h.state().detail.text, /original request/);
} else {
  assert.equal(h.state().detail.available, false);
  assert.equal(h.state().detail.text, undefined);
  assert.match(h.state().detail.reason, /log-messages|capture/i);
}
await h.dispatch('ui.close', { id: 'headroom-sidebar' }, async () => ({}));
assert.equal(h.state().detail, null);
// A changed native session must stop reads using the launcher-linked old UUID.
h.ctl.sid = OTHER;
const before = h.calls.filter(c => c[0] === 'http.fetch').length;
await h.dispatch('command.run', { command: 'headroom-sidebar' });
await h.advance(10000);
assert.equal(h.state().summary, null);
assert.equal(h.state().detail, null);
assert.equal(h.calls.filter(c => c[0] === 'http.fetch').length, before);
assert.match(words(await h.pane()), /relaunch|not linked/i);
console.log('PASS: real HTTP sidebar metrics, inspection, isolation, close and session-change lifecycle');
