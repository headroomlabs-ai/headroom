import test from 'node:test';
import assert from 'node:assert/strict';
import { localUrl, safeText, percent, count, parseResponse, textPages } from '../plugins/headroom-sidebar/hooks/core.mjs';
import { summary, response, SID, OTHER, detail } from './host-fixture.mjs';

for (const raw of ['https://evil.example', 'http://evil.example', 'http://127.0.0.1/path', 'http://u:p@127.0.0.1', 'http://127.0.0.1:0',
  'http://127.0.0.1:99999', 'http://2130706433', 'http://0177.0.0.1', 'http://127.0.0.1?x=y', 'http://127.0.0.1#secret']) {
  test(`reject nonlocal/noncanonical endpoint ${raw}`, () => assert.throws(() => localUrl(raw)));
}
test('local URL normalizes localhost and preserves explicit port 80', () => {
  assert.equal(localUrl('http://localhost:8787/'), 'http://127.0.0.1:8787');
  assert.equal(localUrl('http://127.0.0.1:80'), 'http://127.0.0.1:80');
  assert.equal(localUrl('http://[::1]:8080'), 'http://[::1]:8080');
});
test('unknown metrics remain unknown, not zero', () => {
  assert.equal(percent(null), '—'); assert.equal(count(undefined), '—'); assert.equal(count(0), '0'); assert.equal(percent(-20), '-20.0%');
});
test('terminal controls and bidi controls cannot render', () => assert.equal(safeText('\x1b[2J\u202e'), '�[2J�'));
test('reject cross-session, unsupported schema, invalid JSON, oversized body', () => {
  assert.throws(() => parseResponse(response(summary(OTHER)), SID));
  assert.throws(() => parseResponse(response({ ...summary(), schema_version: 999 }), SID));
  assert.throws(() => parseResponse({ ok: true, text: '<html>' }, SID));
  assert.throws(() => parseResponse({ ok: true, text: 'x'.repeat(400001) }, SID));
});
test('metadata normalizer discards arbitrary content and tags', () => {
  const s = summary(); s.request_messages = 'SECRET'; s.requests[0].tags = { key: 'SECRET' }; s.requests[0].request_messages = 'SECRET';
  assert.ok(!JSON.stringify(parseResponse(response(s), SID)).includes('SECRET'));
});
test('invalid rows do not crash render downstream', () => {
  const s = summary(); s.requests = [null]; assert.throws(() => parseResponse(response(s), SID));
});
test('inspection validates request, pagination, text budget', () => {
  assert.throws(() => parseResponse(response(detail()), SID, 'another-request'));
  assert.throws(() => parseResponse(response({ ...detail(), page: -1 }), SID, 'req-1'));
  assert.throws(() => parseResponse(response({ ...detail(), text: 'a'.repeat(1501) }), SID, 'req-1'));
});
test('path-like request ids are never inspectable', () => {
  const s = summary(); s.requests[0].request_id = '../../v1/messages';
  assert.equal(parseResponse(response(s), SID).requests[0].inspectable_id, false);
});
test('text pagination is bounded and complete', () => {
  const input = 'a'.repeat(400);
  const pages = textPages(input, 30, 15);
  assert.ok(pages.length > 1);
  assert.equal(pages.join('').replace(/\n/g, ''), input);
});
