/**
 * Self-check for scheduled()'s hourly compaction.  node test-rollup.mjs [dir]
 *
 * The one thing that must never drift: rollupHour() and the QUALIFY in
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
import { inflate, oldestRawDay, rollupHour } from './worker.js';

// R2 returns at most 1000 keys per list page, so on a real hour (~4,000
// objects) the cursor loop in rollupHour is load-bearing. The stub paginates at
// a deliberately tiny size so that loop is exercised by every case below: with
// a single-page stub, a regression that dropped the cursor would still print
// "ok" while silently rolling up only the first page of every hour.
const PAGE = 3;

/** The slice of the R2 binding rollupHour uses, backed by a plain object. */
function stubBucket(files, { failKeys = new Set() } = {}) {
  const written = {};
  const reads = [];
  return {
    written,
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
        objects: page.map((key) => ({ key })),
        truncated: next < keys.length,
        cursor: next < keys.length ? keys[next] : undefined,
      };
    },
    get: async (key) => {
      reads.push(key);
      if (failKeys.has(key)) throw new Error(`simulated R2 failure: ${key}`);
      if (!(key in files)) return null;
      return { text: async () => files[key] };
    },
    put: async (key, body) => {
      written[key] = body;
    },
  };
}

const beacon = (install, id, seq) =>
  JSON.stringify({ resource: { 'headroom.install_id': install }, session: { id, seq } });

const PART = 'dt=2026-08-06/hh=14';

/** Run rollupHour against a stub bucket and decode whatever it wrote. */
async function run(files, opts = {}) {
  const CORPUS = stubBucket(files, opts);
  const spend = { read: 0 };
  let threw = null;
  let out = null;
  try {
    out = await rollupHour({ CORPUS }, PART, spend);
  } catch (err) {
    threw = err;
  }
  const body = CORPUS.written[`rollup/${PART}/data.ndjson`];
  return {
    threw,
    spend,
    wrote: out ? out.wrote : 0,
    empty: `rollup/${PART}/empty` in CORPUS.written,
    keys: Object.keys(CORPUS.written),
    rows: body ? body.split('\n').map((l) => JSON.parse(l)) : [],
  };
}

// 1. Highest seq wins, out-of-order input, one row per (install, session).
//    More objects than PAGE, so the list cursor loop runs.
{
  const files = {
    [`sessions/${PART}/a.json`]: [beacon('i1', 's1', 3), beacon('i1', 's2', 1)].join('\n'),
    [`sessions/${PART}/b.json`]: beacon('i1', 's1', 9),
    [`sessions/${PART}/c.json`]: beacon('i1', 's1', 7),
    // Same session id under a different install must not collapse together.
    [`sessions/${PART}/d.json`]: beacon('i2', 's1', 2),
    [`sessions/${PART}/e.json`]: beacon('i1', 's1', 5),
  };
  const { rows, spend, threw } = await run(files);
  assert.equal(threw, null);
  // 5 objects at PAGE=3 is two pages: proves the cursor loop, which is
  // load-bearing at the real ~4,000 objects/hour.
  assert.ok(Object.keys(files).length > PAGE, 'fixture must span pages');
  assert.equal(spend.read, 5, 'reads every object across every page');
  assert.equal(rows.length, 3, 'one row per (install, session)');
  const seq = Object.fromEntries(
    rows.map((r) => [`${r.resource['headroom.install_id']} ${r.session.id}`, r.session.seq])
  );
  assert.deepEqual(seq, { 'i1 s1': 9, 'i1 s2': 1, 'i2 s1': 2 });
}

// 2. An unparseable record loses only itself. Content this Worker wrote with
//    JSON.stringify never becomes valid later, so blocking the hour on it would
//    strand the hour rather than one record.
{
  const files = {
    [`sessions/${PART}/a.json`]: '{ this is not json',
    [`sessions/${PART}/b.json`]: `\n${beacon('i1', 's1', 4)}\n`,
  };
  const { rows, threw } = await run(files);
  assert.equal(threw, null, 'corrupt content does not abandon the hour');
  assert.deepEqual(rows.map((r) => r.session.seq), [4], 'survives a corrupt object');
}

// 3. A failed get is transient, so the hour must NOT be written — a rollup is
//    built once and then trusted forever, so a short read would silently become
//    the permanent record.
{
  const files = {
    [`sessions/${PART}/a.json`]: beacon('i1', 's1', 1),
    [`sessions/${PART}/b.json`]: beacon('i1', 's2', 1),
  };
  const { threw, keys } = await run(files, {
    failKeys: new Set([`sessions/${PART}/b.json`]),
  });
  assert.ok(threw, 'a failed get throws so the hour is retried');
  assert.deepEqual(keys, [], 'nothing written on a partial read');
}

// 4. Spend is reported even when the hour throws. Charging a flat guess instead
//    lets a run that failed late overshoot the subrequest ceiling.
{
  const files = Object.fromEntries(
    Array.from({ length: 7 }, (_, i) => [`sessions/${PART}/o${i}.json`, beacon('i1', `s${i}`, 1)])
  );
  const { threw, spend } = await run(files, {
    failKeys: new Set([`sessions/${PART}/o6.json`]),
  });
  assert.ok(threw);
  assert.equal(spend.read, 7, 'caller sees real spend, not a guess');
}

// 5. An empty hour writes a marker, not a zero-byte NDJSON. Without it the hour
//    stays "missing" and is re-listed on every run forever.
{
  const { rows, empty, keys } = await run({});
  assert.deepEqual(rows, []);
  assert.ok(empty, 'empty hour leaves a marker');
  assert.ok(
    keys.every((k) => !k.endsWith('.ndjson')),
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
  const { rows, spend, threw } = await run(files);
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
  assert.equal(spend.read, Object.keys(files).length);
  assert.equal(rows.length, expected.size, 'row count matches DISTINCT sessions');
  for (const r of rows) {
    const k = `${r.resource['headroom.install_id']} ${r.session.id}`;
    assert.equal(r.session.seq, expected.get(k), `max seq for ${k}`);
  }
  console.log(
    `real corpus: ${spend.read} objects -> ${rows.length} sessions in 1 object` +
      ` (${Math.ceil(spend.read / PAGE)} list pages)`
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

console.log('ok');
