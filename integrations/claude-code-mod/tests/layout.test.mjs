import test from 'node:test';
import assert from 'node:assert/strict';
import { host, press, words, summary, response } from './host-fixture.mjs';

test('overview keeps secondary telemetry behind details without losing its values', async () => {
  const h = host(); await h.start();
  assert.match(words(await h.pane()), /400 tokens removed/);
  assert.doesNotMatch(words(await h.pane()), /Cache reads|mean compression overhead|smart_crusher/);
  await press(h, 'details');
  let text = words(await h.pane());
  assert.match(text, /200.*100/);
  assert.match(text, /8\.0 ms/);
  assert.match(text, /smart_crusher.*1/);
  await h.advance(5000);
  assert.equal(h.state().showDetails, true);
  await press(h, 'details');
  assert.doesNotMatch(words(await h.pane()), /smart_crusher/);
});

test('collapsed details never hide failed accounting or retention warnings', async () => {
  const s = summary();
  s.totals.failed_requests = 1; s.totals.unaccounted_requests = 1;
  s.latest.failed = true; s.latest.accounting = 'missing';
  s.retention.window_full = true;
  const h = host({ transport: async () => response(s) }); await h.start();
  const text = words(await h.pane());
  assert.match(text, /1 failed/);
  assert.match(text, /excluded from savings/);
  assert.match(text, /accounting (?:is )?(?:unavailable|unknown)/i);
  assert.match(text, /older requests|older history/i);
});
