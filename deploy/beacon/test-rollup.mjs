/**
 * Self-check for the hourly compaction.  node test-rollup.mjs [dir]
 *
 * The one thing that must never drift: the rollup and the QUALIFY in
 * headroom-beacon-stats/beacon.sh have to agree on which heartbeat wins. If
 * they disagree the reports get quietly wrong rather than loudly broken, so
 * this asserts the JS picks exactly the max-seq row per (install, session).
 *
 * Point it at a directory of real beacon objects to check against the corpus:
 *   aws s3 sync s3://headroom-telemetry/sessions/dt=.../hh=.../ /tmp/hr/ ...
 *   node test-rollup.mjs /tmp/hr
 * With no argument it runs on a small fixture and needs no network.
 */
import { readdirSync, readFileSync } from 'node:fs';
import assert from 'node:assert/strict';
import worker, { finalizeHour, inflate, oldestRawDay, rollupSlice } from './worker.js';

// R2 returns at most 1000 keys per list page, so on a real slice the cursor
// loop is load-bearing. The stub paginates at a deliberately tiny size so that
// loop is exercised by every case below: with a single-page stub, a regression
// that dropped the cursor would still print "ok" while silently rolling up only
// the first page of every slice.
const PAGE = 3;
const SLICES = '0123456789abcdef';

/** The slice of the R2 binding the rollup uses, backed by a plain object. */
function stubBucket(files, { failKeys = new Set() } = {}) {
  const reads = [];
  return {
    files,
    reads,
    list: async ({ prefix, cursor, delimiter }) => {
      const keys = Object.keys(files)
        .filter((k) => k.startsWith(prefix))
        .sort();
      if (delimiter) {
        const seen = new Set();
        for (const k of keys) {
          const cut = k.indexOf(delimiter, prefix.length);
          if (cut >= 0) seen.add(k.slice(0, cut + 1));
        }
        return { objects: [], delimitedPrefixes: [...seen], truncated: false };
      }
      const start = cursor ? keys.indexOf(cursor) : 0;
      const page = keys.slice(start, start + PAGE);
      const next = start + PAGE;
      return {
        objects: page.map((key) => ({ key, uploaded: files[key].uploaded ?? new Date(0) })),
        truncated: next < keys.length,
        cursor: next < keys.length ? keys[next] : undefined,
      };
    },
    get: async (key) => {
      reads.push(key);
      if (failKeys.has(key)) throw new Error(`simulated R2 failure: ${key}`);
      if (!(key in files)) return null;
      return { text: async () => String(files[key]) };
    },
    head: async (key) => (key in files ? { key } : null),
    put: async (key, body) => {
      files[key] = body;
    },
    delete: async (keys) => {
      for (const key of keys) delete files[key];
    },
  };
}

const beacon = (install, id, seq) =>
  JSON.stringify({ resource: { 'headroom.install_id': install }, session: { id, seq } });

const PART = 'dt=2026-08-06/hh=14';

/** Roll every slice of PART, merge, and decode whatever landed under rollup/. */
async function run(files, opts = {}) {
  const CORPUS = stubBucket({ ...files }, opts);
  let threw = null;
  try {
    for (const slice of SLICES) await rollupSlice({ CORPUS }, PART, slice);
    await finalizeHour({ CORPUS }, PART);
  } catch (err) {
    threw = err;
  }
  const body = CORPUS.files[`rollup/${PART}/data.ndjson`];
  return {
    threw,
    rawReads: CORPUS.reads.filter((k) => k.startsWith('sessions/')).length,
    empty: `rollup/${PART}/empty` in CORPUS.files,
    rolled: Object.keys(CORPUS.files).filter((k) => k.startsWith('rollup/')),
    parts: Object.keys(CORPUS.files).filter((k) => k.startsWith('rollup-parts/')),
    rows: body ? body.split('\n').map((l) => JSON.parse(l)) : [],
  };
}

// 1. Highest seq wins across slices, out-of-order input, one row per
//    (install, session). The names start with different hex characters, so one
//    session's heartbeats land in different slices and only the merge can pick
//    the winner.
{
  const files = {
    [`sessions/${PART}/a.json`]: [beacon('i1', 's1', 3), beacon('i1', 's2', 1)].join('\n'),
    [`sessions/${PART}/b.json`]: beacon('i1', 's1', 9),
    [`sessions/${PART}/c.json`]: beacon('i1', 's1', 7),
    // Same session id under a different install must not collapse together.
    [`sessions/${PART}/d.json`]: beacon('i2', 's1', 2),
    [`sessions/${PART}/e.json`]: beacon('i1', 's1', 5),
    // Four in one slice, so that slice spans list pages.
    [`sessions/${PART}/f1.json`]: beacon('i3', 's1', 1),
    [`sessions/${PART}/f2.json`]: beacon('i3', 's1', 4),
    [`sessions/${PART}/f3.json`]: beacon('i3', 's1', 2),
    [`sessions/${PART}/f4.json`]: beacon('i3', 's1', 3),
  };
  const { rows, rawReads, threw, parts } = await run(files);
  assert.equal(threw, null);
  assert.equal(rawReads, 9, 'reads every object across every slice and page');
  assert.equal(rows.length, 4, 'one row per (install, session)');
  const seq = Object.fromEntries(
    rows.map((r) => [`${r.resource['headroom.install_id']} ${r.session.id}`, r.session.seq])
  );
  assert.deepEqual(seq, { 'i1 s1': 9, 'i1 s2': 1, 'i2 s1': 2, 'i3 s1': 4 });
  assert.deepEqual(parts, [], 'parts are removed once the hour is merged');
}

// 2. An unparseable record loses only itself. Content this Worker wrote with
//    JSON.stringify never becomes valid later, so blocking the hour on it would
//    strand the hour rather than one record.
{
  const files = {
    [`sessions/${PART}/a.json`]: '{ this is not json',
    [`sessions/${PART}/b.json`]: `\n${beacon('i1', 's1', 4)}\n`,
  };
  const quiet = console.error;
  console.error = () => {};
  let result;
  try {
    result = await run(files);
  } finally {
    console.error = quiet;
  }
  assert.equal(result.threw, null, 'corrupt content does not abandon the hour');
  assert.deepEqual(result.rows.map((r) => r.session.seq), [4], 'survives a corrupt object');
}

// 3. A failed get is transient, so its slice must NOT be written — a rollup is
//    built once and then trusted forever, so a short read would silently become
//    the permanent record. The slices that did read keep their parts, which is
//    the point: a retry redoes one slice, not the hour.
{
  const files = {
    [`sessions/${PART}/a.json`]: beacon('i1', 's1', 1),
    [`sessions/${PART}/b.json`]: beacon('i1', 's2', 1),
  };
  const CORPUS = stubBucket({ ...files }, { failKeys: new Set([`sessions/${PART}/b.json`]) });
  await rollupSlice({ CORPUS }, PART, 'a');
  await assert.rejects(() => rollupSlice({ CORPUS }, PART, 'b'), /simulated R2 failure/);
  assert.equal(await finalizeHour({ CORPUS }, PART), false, 'no merge while a slice is missing');
  const keys = Object.keys(CORPUS.files);
  assert.ok(!keys.some((k) => k.startsWith('rollup/')), 'nothing published on a partial read');
  assert.ok(keys.includes(`rollup-parts/${PART}/a`), 'the slice that read cleanly is kept');
  assert.ok(!keys.includes(`rollup-parts/${PART}/b`), 'the failed slice wrote nothing');
}

// 4. scheduled() enqueues one message per slice for every hour with no rollup,
//    and leaves alone both finished hours and hours it enqueued recently —
//    otherwise every cron run would enqueue the whole in-flight backlog again.
//    queue() then turns those messages into the rollup.
{
  const at = Date.parse('2026-08-06T15:05:00Z');
  const CORPUS = stubBucket({
    'sessions/dt=2026-08-06/hh=12/a.json': beacon('i1', 's1', 1),
    'sessions/dt=2026-08-06/hh=13/a.json': beacon('i1', 's2', 1),
    'sessions/dt=2026-08-06/hh=14/a.json': beacon('i1', 's3', 2),
    'sessions/dt=2026-08-06/hh=14/b.json': beacon('i1', 's3', 6),
    'rollup/dt=2026-08-06/hh=12/data.ndjson': beacon('i1', 's1', 1),
    'rollup-parts/dt=2026-08-06/hh=13/queued': Object.assign(new String(''), {
      uploaded: new Date(at - 3600_000),
    }),
  });
  const sent = [];
  const env = { CORPUS, ROLLUP_QUEUE: { sendBatch: async (msgs) => sent.push(...msgs) } };
  await worker.scheduled({ scheduledTime: at }, env);

  const hours = (h) => sent.filter((m) => m.body.part === `dt=2026-08-06/hh=${h}`);
  assert.deepEqual(hours('14').map((m) => m.body.slice).join(''), SLICES, 'every slice of the missing hour');
  assert.equal(hours('12').length, 0, 'a finished hour is not enqueued');
  assert.equal(hours('13').length, 0, 'an hour enqueued an hour ago is not enqueued again');
  assert.equal(hours('00').length, 16, 'backfill reaches the start of the oldest raw day');

  const acked = [];
  const messages = hours('14').map((m) => ({
    body: m.body,
    ack: () => acked.push(m.body.slice),
    retry: () => assert.fail(`slice ${m.body.slice} was retried`),
  }));
  await worker.queue({ messages }, env);
  assert.equal(acked.length, 16);
  const row = JSON.parse(CORPUS.files[`rollup/${PART}/data.ndjson`]);
  assert.equal(row.session.seq, 6, 'the queue path publishes the merged hour');

  // A redelivered message for a finished hour must not write a part: nothing
  // would ever merge or remove it.
  await worker.queue({ messages: [messages[0]] }, env);
  assert.ok(!Object.keys(CORPUS.files).some((k) => k.startsWith(`rollup-parts/${PART}/`)));
}

// 4b. The recovery paths. An hour enqueued longer ago than the re-enqueue
//     window is sent again; a slice that fails is handed back to the queue, not
//     acknowledged; a second copy of a finished slice reuses its part instead
//     of reading the slice again; and whatever a finished hour left under
//     rollup-parts/ is swept, because nothing else would ever remove it.
{
  const at = Date.parse('2026-08-06T15:05:00Z');
  const stale = (hoursAgo) =>
    Object.assign(new String(''), { uploaded: new Date(at - hoursAgo * 3600_000) });
  const CORPUS = stubBucket(
    {
      'sessions/dt=2026-08-06/hh=13/a.json': beacon('i1', 's2', 1),
      'sessions/dt=2026-08-06/hh=14/a.json': beacon('i1', 's3', 2),
      'sessions/dt=2026-08-06/hh=14/b.json': beacon('i1', 's3', 6),
      'rollup/dt=2026-08-06/hh=13/data.ndjson': beacon('i1', 's2', 1),
      'rollup-parts/dt=2026-08-06/hh=13/queued': stale(1),
      'rollup-parts/dt=2026-08-06/hh=13/a': beacon('i1', 's2', 1),
      'rollup-parts/dt=2026-08-06/hh=14/queued': stale(4),
    },
    { failKeys: new Set(['sessions/dt=2026-08-06/hh=14/b.json']) }
  );
  const sent = [];
  const env = { CORPUS, ROLLUP_QUEUE: { sendBatch: async (msgs) => sent.push(...msgs) } };
  await worker.scheduled({ scheduledTime: at }, env);

  assert.equal(
    sent.filter((m) => m.body.part === PART).length,
    16,
    'an hour enqueued four hours ago is enqueued again'
  );
  assert.ok(
    !Object.keys(CORPUS.files).some((k) => k.startsWith('rollup-parts/dt=2026-08-06/hh=13/')),
    'leftovers of a finished hour are swept'
  );

  const outcome = [];
  const message = (slice) => ({
    body: { part: PART, slice },
    ack: () => outcome.push(`ack ${slice}`),
    retry: (opts) => outcome.push(`retry ${slice} ${opts.delaySeconds}`),
  });
  const quiet = console.error;
  console.error = () => {};
  try {
    await worker.queue({ messages: [message('a'), message('b')] }, env);
  } finally {
    console.error = quiet;
  }
  assert.deepEqual(outcome, ['ack a', 'retry b 60'], 'a failed slice goes back to the queue');

  const before = CORPUS.reads.length;
  await worker.queue({ messages: [message('a')] }, env);
  assert.equal(CORPUS.reads.length, before, 'a finished slice is not read a second time');
}

// 5. An empty hour writes a marker, not a zero-byte NDJSON. Without it the hour
//    stays "missing" and is enqueued again on every run forever.
{
  const { rows, empty, rolled } = await run({});
  assert.deepEqual(rows, []);
  assert.ok(empty, 'empty hour leaves a marker');
  assert.ok(
    rolled.every((k) => !k.endsWith('.ndjson')),
    'no zero-byte ndjson for readers to special-case'
  );
}

// 6. oldestRawDay floors the backfill. A fixed lookback window silently strands
//    every hour older than it once analysis stopped reading sessions/.
{
  const CORPUS = stubBucket({
    'sessions/dt=2026-08-03/hh=01/a.json': beacon('i1', 's1', 1),
    'sessions/dt=2026-08-06/hh=14/b.json': beacon('i1', 's2', 1),
    'sessions/dt=2026-08-07/hh=00/c.json': beacon('i1', 's3', 1),
  });
  assert.equal(await oldestRawDay({ CORPUS }), '2026-08-03');
  assert.equal(await oldestRawDay({ CORPUS: stubBucket({}) }), null, 'empty bucket -> null');
}

// 7. Against real objects, if a directory was given: same answer as the QUALIFY
//    in beacon.sh, which is `count(DISTINCT install||session)` rows, each
//    carrying that pair's max seq.
const dir = process.argv[2];
if (dir) {
  const files = {};
  for (const f of readdirSync(dir).filter((f) => f.endsWith('.json'))) {
    files[`sessions/${PART}/${f}`] = readFileSync(`${dir}/${f}`, 'utf8');
  }
  const { rows, rawReads, threw } = await run(files);
  assert.equal(threw, null);

  const expected = new Map();
  for (const text of Object.values(files)) {
    for (const line of text.split('\n')) {
      if (!line.trim()) continue;
      const r = JSON.parse(line);
      const k = `${r.resource?.['headroom.install_id']} ${r.session?.id}`;
      expected.set(k, Math.max(expected.get(k) ?? -1, r.session?.seq ?? 0));
    }
  }
  assert.equal(rawReads, Object.keys(files).length);
  assert.equal(rows.length, expected.size, 'row count matches DISTINCT sessions');
  for (const r of rows) {
    const k = `${r.resource['headroom.install_id']} ${r.session.id}`;
    assert.equal(r.session.seq, expected.get(k), `max seq for ${k}`);
  }
  console.log(
    `real corpus: ${rawReads} objects -> ${rows.length} sessions in 1 object`
  );
}

// 8. Upload compression. A schema-v2 event is ~8KB of repetitive JSON and
//    gzips ~6x, so fetch() sniffs the gzip magic number and inflates. Three
//    things have to hold, and none of them can be checked in production without
//    risking every upload:
//
//    a) a real gzip body round-trips to the exact bytes that went in;
//    b) DecompressionStream exists in this runtime at all (it is the one part
//       of the ingest path with no fallback -- if it is missing, inflate throws,
//       fetch answers 400, and every compressed upload is refused);
//    c) the size cap is enforced WHILE the stream is read, not after, or a
//       decompression bomb has already been decompressed by the time it is
//       noticed.
{
  const gzipBytes = async (text) => {
    const stream = new Response(new TextEncoder().encode(text)).body.pipeThrough(
      new CompressionStream('gzip')
    );
    return new Uint8Array(await new Response(stream).arrayBuffer());
  };

  const payload = JSON.stringify({
    resourceLogs: [{ scopeLogs: [{ logRecords: [{ body: { stringValue: 'x'.repeat(4000) } }] }] }],
  });
  const packed = await gzipBytes(payload);

  assert.equal(packed[0], 0x1f, 'gzip magic byte 0');
  assert.equal(packed[1], 0x8b, 'gzip magic byte 1');
  assert.ok(packed.length < payload.length / 3, 'gzip did not actually compress');
  assert.equal(await inflate(packed.buffer, 256 * 1024), payload, 'round trip changed the body');

  // A bomb: highly compressible input that expands past the cap. The cap must
  // bite, and it must bite from the stream rather than from the final size.
  const bomb = await gzipBytes('A'.repeat(2 * 1024 * 1024));
  assert.ok(bomb.length < 64 * 1024, 'fixture is not actually a bomb');
  await assert.rejects(
    () => inflate(bomb.buffer, 256 * 1024),
    /too large/,
    'a decompression bomb was inflated in full'
  );

  console.log(`gzip: ${payload.length} B -> ${packed.length} B, bomb refused`);
}

// ---- fetch(): the RATE_LIMIT binding -----------------------------------------
//
// A body with no log records stops at the 204 right after parsing, so these
// exercise the limiter without touching R2.
{
  const post = () =>
    new Request('https://beacon.test/v1/logs', { method: 'POST', body: '{}' });
  const limiter = (impl) => ({ RATE_LIMIT: { limit: impl } });

  const over = await worker.fetch(post(), limiter(async () => ({ success: false })), {});
  assert.equal(over.status, 429, 'over the limit is refused');

  const under = await worker.fetch(post(), limiter(async () => ({ success: true })), {});
  assert.equal(under.status, 204, 'under the limit proceeds');

  const quiet = console.error;
  console.error = () => {};
  let down;
  try {
    down = await worker.fetch(post(), limiter(async () => { throw new Error('limiter down'); }), {});
  } finally {
    console.error = quiet;
  }
  assert.equal(down.status, 204, 'a limiter outage fails open instead of dropping beacons');

  const unbound = await worker.fetch(post(), {}, {});
  assert.equal(unbound.status, 204, 'no binding fails open');

  let keys = [];
  await worker.fetch(post(), limiter(async (opts) => { keys.push(opts.key); return { success: true }; }), {});
  assert.deepEqual(keys, ['all'], 'one shared key: the sender is never read');

  console.log('rate limit: 429 over, fail-open on outage and when unbound');
}

// ---- fetch(): the nested allowlist -------------------------------------------
//
// The allowlist has to hold BELOW the top-level keys too: a nested field a
// client bug adds, or free text under an allowed key, must not be stored. And
// nothing the current client sends may be lost on the way in. The sample event
// is the real client's output; the lists it leaves empty are filled with rows
// of the shape session.py emits, so every nested field is exercised.
{
  /** Plain JS -> OTLP AnyValue, as _any_value() in session.py builds it. */
  const any = (v) =>
    typeof v === 'boolean' ? { boolValue: v }
    : Number.isInteger(v) ? { intValue: String(v) }
    : typeof v === 'number' ? { doubleValue: v }
    : typeof v === 'string' ? { stringValue: v }
    : Array.isArray(v) ? { arrayValue: { values: v.map(any) } }
    : { kvlistValue: { values: Object.entries(v).map(([key, x]) => ({ key, value: any(x) })) } };

  const sample = JSON.parse(readFileSync(new URL('./sample-event.json', import.meta.url), 'utf8'));
  const resourceAttrs = sample.resourceLogs[0].resource.attributes;
  const plain = (v) =>
    'kvlistValue' in v
      ? Object.fromEntries(v.kvlistValue.values.map((kv) => [kv.key, plain(kv.value)]))
      : 'arrayValue' in v ? v.arrayValue.values.map(plain)
      : 'intValue' in v ? Number(v.intValue)
      : Object.values(v)[0];
  const full = plain(sample.resourceLogs[0].scopeLogs[0].logRecords[0].body);
  full.compression.transforms.extracted_3_dynamic_elements = 2;
  full.compression.by_strategy = [{ strategy: 'smart_crusher', n: 2, tokens_in: 1500, tokens_out: 700 }];
  full.skips = { 'passthrough:bypass_header': 1 };
  full.failure_statuses = { 529: 1 };
  full.models.push('us.anthropic.claude-3-5-sonnet-20241022-v2:0', 'vertex_ai/gemini-1.5-pro@001');
  full.quality.waste = [{ kind: 'json_bloat', tokens: 40 }];
  full.shapes.by_content = [{ content: 'json', strategy: 'smart_crusher', n: 1, tokens_in: 9, tokens_out: 3 }];
  full.shapes.by_tool = [{ shape: 'f2d3_a', n: 1, tokens_in: 9, tokens_out: 3 }];
  full.trajectory.kinds = 's12p3z40';
  full.errors = { count: 1, by_status: { 429: 1 } };
  full.strata = [{ dim: 'arm', value: 'treatment', n: 2 }];
  full.config = [{ key: 'mode', value: 'token' }];
  full.routing = { decisions: 3, by_model: [{ model: 'claude-haiku-4-5', n: 3 }], min_prefix: 1024 };

  /** One OTLP record through fetch(); returns what would be written to R2. */
  const store = async (body, resource = resourceAttrs) => {
    const otlp = { resourceLogs: [{ resource: { attributes: resource },
      scopeLogs: [{ logRecords: [{ body: any(body) }] }] }] };
    const pending = [];
    const written = {};
    const CORPUS = { put: async (key, value) => { written[key] = value; } };
    const res = await worker.fetch(
      new Request('https://beacon.test/v1/logs', { method: 'POST', body: JSON.stringify(otlp) }),
      { CORPUS },
      { waitUntil: (p) => pending.push(p) }
    );
    await Promise.all(pending);
    assert.equal(res.status, 204);
    const [line] = Object.values(written);
    const { received_at, ...rec } = JSON.parse(line);
    assert.ok(received_at);
    return rec;
  };

  const { resource, ...stored } = await store(full);
  assert.deepEqual(stored, full, 'a field the current client sends was dropped');
  assert.deepEqual(resource, plain({ kvlistValue: { values: resourceAttrs } }));

  const injected = structuredClone(full);
  injected.session.note = 'hello';                               // unknown nested key
  injected.tokens.prompt_sample = 12;                            // unknown nested counter
  injected.compression.by_strategy[0].prompt = 'secret';         // unknown key in a record
  injected.models.push('Please summarise this customer email'); // free text under a list
  // Credential-shaped fixtures are assembled at run time so secret scanners do not
  // flag this file; they are not real keys.
  const fakeKey = ['sk', 'live', 'abc123'].join('_');
  const fakeAwsId = 'AK' + 'IA' + 'X'.repeat(16);
  injected.models.push(fakeKey, 'https://internal.example/admin'); // slug-shaped secret, URL
  injected.strata.push({ dim: 'arm', value: fakeAwsId, n: 1 }); // key-shaped value
  injected.config.push({ key: 'endpoint', value: 'https://internal.example/x?token=1 2' });
  injected.session.ended = { nested: 'object where a slug belongs' };
  injected.routing.note = 'free text with spaces';
  injected.trajectory.kinds = 'see attached transcript';
  const hostile = [...resourceAttrs.filter((a) => a.key !== 'headroom.stack'),
    { key: 'headroom.stack', value: any({ cwd: '/home/me' }) }];
  const { resource: badResource, ...dropped } = await store(injected, hostile);

  const expected = structuredClone(full);
  delete expected.session.ended;
  delete expected.trajectory.kinds;
  expected.config.push({ key: 'endpoint' });
  expected.strata.push({ dim: 'arm', n: 1 });
  assert.deepEqual(dropped, expected, 'a nested field outside the allowlist was stored');
  assert.equal(badResource['headroom.stack'], undefined, 'non-scalar resource value stored');

  console.log('nested allowlist: client fields kept, unknown nested keys and free text dropped');
}

console.log('ok');
