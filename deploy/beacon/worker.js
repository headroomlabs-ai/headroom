/**
 * Headroom telemetry beacon receiver.
 *
 * This file is open source on purpose. It is the other half of the promise
 * made in headroom/telemetry/session.py: users can read exactly what the
 * client sends AND exactly what happens to it on arrival. "Trust us" is not a
 * privacy policy.
 *
 * Deployed at otlp.headroomlabs.ai. Three jobs:
 *
 *   1. Allowlist. Drop every field not on ALLOWED_KEYS before anything is
 *      written. This is the only privacy control that works retroactively —
 *      if a future client version ships a bug that leaks a field, we cannot
 *      patch the installs already in the wild, but we can stop storing it
 *      here in one deploy.
 *
 *   2. Flatten. OTLP AnyValue nesting is portable but miserable to query
 *      ({"kvlistValue":{"values":[{"key":"tokens",...}]}}). We keep OTLP on
 *      the wire so the backend stays vendor-swappable, and store plain JSON so
 *      DuckDB can read it without unwrapping anything.
 *
 *   3. Fan out. R2 for the durable corpus; optionally a metrics vendor for
 *      dashboards. Adding a destination is one more call here — never a
 *      client release.
 *
 * What this deliberately does NOT do: log, store, or forward the source IP.
 * Cloudflare offers it as cf-connecting-ip; it is the one field that would
 * deanonymise install_id, so it is never read.
 */

// Mostly mirrors the payload built by _Session.payload(); an extension may
// also emit its own event carrying one of these top-level keys. A key absent
// here is dropped, not stored. Adding a metric means adding it here first —
// that friction is the point, and it is also the only privacy control that
// works retroactively, so it must land BEFORE any client starts sending the
// key or that traffic is silently discarded and unrecoverable.
const ALLOWED_KEYS = [
  'schema_version',
  'session',
  'tokens',
  'rates',
  'compression',
  'skips',
  'sources',
  'providers',
  'models',
  'failures',
  'failure_statuses',
  // Model-routing summary. Emitted by a routing extension rather than by the
  // proxy itself -- see proxy/route_advice.py for the decision seam. Same rule
  // as everything above: counters and model ids, no free text. Allowlisted
  // here so the corpus can answer what the proxy alone cannot -- a provider's
  // real minimum cacheable prefix, how long a cache actually survives, and how
  // far predicted cache hits are from the ones that happened.
  'routing',
];

// Resource attributes we keep. Same rule: allowlist, not denylist.
const ALLOWED_RESOURCE = [
  'service.name',
  'service.version',
  'headroom.install_id',
  'headroom.install_mode',
  'headroom.stack',
  'os.type',
  'host.arch',
];

// A beacon event is ~2KB. Anything far past that is a bug or an attack.
const MAX_BODY_BYTES = 64 * 1024;

/** OTLP AnyValue -> plain JS. The inverse of _any_value() in session.py. */
function unwrap(value) {
  if (value == null) return null;
  if ('stringValue' in value) return value.stringValue;
  if ('boolValue' in value) return value.boolValue;
  if ('intValue' in value) return Number(value.intValue);
  if ('doubleValue' in value) return value.doubleValue;
  if ('arrayValue' in value) return (value.arrayValue.values || []).map(unwrap);
  if ('kvlistValue' in value) {
    const out = {};
    for (const kv of value.kvlistValue.values || []) out[kv.key] = unwrap(kv.value);
    return out;
  }
  return null;
}

function pick(obj, allowed) {
  const out = {};
  if (!obj || typeof obj !== 'object') return out;
  for (const key of allowed) {
    if (key in obj) out[key] = obj[key];
  }
  return out;
}

/** OTLP ExportLogsServiceRequest -> flat, allowlisted records. */
function extract(payload) {
  const records = [];
  for (const rl of payload.resourceLogs || []) {
    const resource = {};
    for (const attr of rl.resource?.attributes || []) {
      resource[attr.key] = unwrap(attr.value);
    }
    const cleanResource = pick(resource, ALLOWED_RESOURCE);

    for (const sl of rl.scopeLogs || []) {
      for (const rec of sl.logRecords || []) {
        const body = unwrap(rec.body);
        if (!body || typeof body !== 'object') continue;
        records.push({
          ...pick(body, ALLOWED_KEYS),
          resource: cleanResource,
          // Server-stamped. A client clock can be wrong or forged; this is the
          // timestamp partitioning and retention actually rely on.
          received_at: new Date().toISOString(),
        });
      }
    }
  }
  return records;
}

// ----------------------------------------------------------------- rollup --
//
// The corpus is one object per heartbeat, ~1KB each — 65k on 2026-08-06 and
// climbing. DuckDB reads them correctly, but a full `pull` is ~100k HTTPS round
// trips for 95MB: minutes of pure per-object latency, no real bytes or compute.
// Listing the bucket alone took 88 seconds.
//
// This job collapses each COMPLETE hour into one object under rollup/, keeping
// only the highest-seq heartbeat per (install, session). One measured hour
// (dt=2026-08-06/hh=14): 3,938 objects and 3,938 rows in, 1 object and 1,061
// rows out. Analysis reads rollup/**, never sessions/**. Raw is left exactly as
// written, so any rollup can be rebuilt by deleting it.
//
// Hourly rather than daily because every R2 binding call is a subrequest: a day
// is ~65k of them against a 10k-per-invocation ceiling, an hour is ~4k.

const READ_BUDGET = 60000; // objects per run; see [limits] in wrangler.toml
// A get costs ~45ms of round trip and almost no CPU, so this is what decides
// whether a run finishes: at 20 an hour took ~3 minutes, against a 15-minute
// wall clock for a cron invocation. Raise it if an hour ever stops fitting.
const FANOUT = 100;        // concurrent R2 gets

const partition = (d) =>
  `dt=${d.toISOString().slice(0, 10)}/hh=${d.toISOString().slice(11, 13)}`;

/**
 * One hour of heartbeats -> one deduped NDJSON object.
 *
 * Returns `{ read, wrote }`. Spend is reported through the mutable `spend`
 * accumulator so the caller still knows it even when this throws: the budget
 * has to track real spend, and a flat guess lets a run that failed late
 * overshoot the subrequest ceiling and get killed inside an hour that would
 * otherwise have succeeded.
 *
 * Writes nothing unless the whole hour read cleanly. A rollup is built once and
 * then treated as done forever, so a partial read would silently become the
 * permanent record — better to write nothing and let the next run retry.
 */
export async function rollupHour(env, part, spend = { read: 0 }) {
  const best = new Map();
  let failed = 0;   // transient: retry the hour
  let corrupt = 0;  // permanent: record and move on
  let cursor;
  do {
    const page = await env.CORPUS.list({ prefix: `sessions/${part}/`, cursor });
    for (let i = 0; i < page.objects.length; i += FANOUT) {
      // allSettled, not all: one transient R2 error among the ~4,000 gets in a
      // real hour would otherwise reject the batch and discard the whole hour.
      const settled = await Promise.allSettled(
        page.objects
          .slice(i, i + FANOUT)
          .map((o) => env.CORPUS.get(o.key).then((r) => (r ? r.text() : null)))
      );
      for (const outcome of settled) {
        spend.read++;
        // A miss counts as a failure too. The key came from a LIST, so the
        // object existed; treating it as empty would quietly shrink the rollup.
        if (outcome.status !== 'fulfilled' || outcome.value === null) {
          failed++;
          continue;
        }
        for (const line of outcome.value.split('\n')) {
          if (!line) continue;
          let rec;
          try {
            rec = JSON.parse(line);
          } catch {
            // Counted and logged, but NOT a reason to abandon the hour. A
            // failed get is transient and worth retrying; content this Worker
            // itself wrote with JSON.stringify does not become valid later, so
            // blocking on it would strand the hour until its raw objects
            // expire and then lose the whole hour instead of one record.
            corrupt++;
            continue;
          }
          // A session heartbeats every 5 minutes carrying CUMULATIVE totals, so
          // the highest seq IS the whole session and every earlier row is a
          // strict subset. Sessions straddle hours, so readers still dedupe
          // across rollups on this same key — this only shrinks each hour.
          const id = `${rec.resource?.['headroom.install_id']} ${rec.session?.id}`;
          const prev = best.get(id);
          if (!prev || (rec.session?.seq ?? 0) > (prev.session?.seq ?? 0)) {
            best.set(id, rec);
          }
        }
      }
    }
    cursor = page.truncated ? page.cursor : undefined;
  } while (cursor);

  if (failed) {
    throw new Error(`${part}: ${failed} of ${spend.read} objects unreadable`);
  }
  if (corrupt) {
    console.error(`rollup ${part}: skipped ${corrupt} unparseable record(s)`);
  }

  // A genuinely empty hour gets a marker rather than a zero-byte NDJSON that
  // every reader would have to special-case. Without it the hour stays
  // "missing" and is re-listed on every run for the life of the bucket.
  if (best.size === 0) {
    await env.CORPUS.put(`rollup/${part}/empty`, '');
    return { read: spend.read, wrote: 0 };
  }
  await env.CORPUS.put(
    `rollup/${part}/data.ndjson`,
    [...best.values()].map((r) => JSON.stringify(r)).join('\n'),
    { httpMetadata: { contentType: 'application/x-ndjson' } }
  );
  return { read: spend.read, wrote: best.size };
}

/** Oldest `dt=` day still under sessions/, or null. One delimited LIST. */
export async function oldestRawDay(env) {
  const page = await env.CORPUS.list({ prefix: 'sessions/', delimiter: '/' });
  const days = (page.delimitedPrefixes || [])
    .map((p) => p.slice('sessions/dt='.length).replace(/\/$/, ''))
    .filter((d) => /^\d{4}-\d{2}-\d{2}$/.test(d))
    .sort();
  return days.length ? days[0] : null;
}

export default {
  /** Hourly cron. Builds every complete hour back to the oldest raw data. */
  async scheduled(event, env) {
    // Backfill reaches all the way to the oldest surviving raw day, NOT a fixed
    // window. A fixed window silently strands everything older than it the
    // moment analysis stopped reading sessions/ — the raw objects are still
    // there, but nothing would ever compact them, so they vanish from every
    // report. Bounding by real data instead means the floor rises only when a
    // lifecycle rule actually expires the raw objects.
    const oldest = await oldestRawDay(env);
    if (!oldest) return;
    const floorMs = Date.parse(`${oldest}T00:00:00Z`);
    if (Number.isNaN(floorMs)) return;

    // Only list from the floor forward. Rollups older than the oldest raw day
    // can never be rebuilt, so enumerating them answers nothing — this is what
    // keeps the listing bounded by retention rather than by total history.
    const done = new Set();
    let cursor;
    do {
      const page = await env.CORPUS.list({
        prefix: 'rollup/',
        startAfter: `rollup/dt=${oldest}`,
        cursor,
      });
      for (const o of page.objects) {
        // Tolerates both `<part>/data.ndjson` and the `<part>/empty` marker.
        const rel = o.key.slice('rollup/'.length);
        const cut = rel.lastIndexOf('/');
        if (cut > 0) done.add(rel.slice(0, cut));
      }
      cursor = page.truncated ? page.cursor : undefined;
    } while (cursor);

    // Newest first, so a backlog drains from the present backwards and the
    // freshest hour is never the one starved by the budget. Starts one hour
    // back: the current hour is still being written to.
    let budget = READ_BUDGET;
    for (let t = event.scheduledTime - 3600_000; t >= floorMs && budget > 0; t -= 3600_000) {
      const part = partition(new Date(t));
      if (done.has(part)) continue;
      // Shared with rollupHour so a throw still reports what it spent.
      const spend = { read: 0 };
      try {
        await rollupHour(env, part, spend);
      } catch (err) {
        // Newest-first means an hour that always throws — one grown past the
        // subrequest ceiling, say — would otherwise block every older hour
        // behind it forever. Skip it and keep draining; it has no marker, so
        // the next run retries it.
        console.error(`rollup ${part} failed after ${spend.read} objects: ${err}`);
      }
      budget -= spend.read;
    }
  },

  async fetch(request, env, ctx) {
    if (request.method !== 'POST') {
      return new Response('beacon: POST OTLP logs to /v1/logs', { status: 405 });
    }
    const url = new URL(request.url);
    if (url.pathname !== '/v1/logs') {
      return new Response('not found', { status: 404 });
    }

    const raw = await request.arrayBuffer();
    if (raw.byteLength > MAX_BODY_BYTES) {
      return new Response('payload too large', { status: 413 });
    }

    let records;
    try {
      records = extract(JSON.parse(new TextDecoder().decode(raw)));
    } catch {
      // Malformed input is not worth a retry storm from clients.
      return new Response('bad request', { status: 400 });
    }
    if (records.length === 0) return new Response(null, { status: 204 });

    // Hive-style partitioning so DuckDB can prune by date without a catalog.
    // Shares partition() with the rollup: the cron lists `sessions/<part>/`, so
    // two independent spellings of this scheme would mean the writer and the
    // compactor could drift apart and silently match zero objects.
    // ponytail: one object per request. Compacted hourly into rollup/ by
    // scheduled() above — analysis reads that, never this.
    const key = `sessions/${partition(new Date())}/${crypto.randomUUID()}.json`;
    const ndjson = records.map((r) => JSON.stringify(r)).join('\n');

    // Respond immediately; durability work continues after the response.
    // The client is fire-and-forget and ignores the status anyway — making it
    // wait on R2 would only add latency to someone else's coding session.
    ctx.waitUntil(
      env.CORPUS.put(key, ndjson, {
        httpMetadata: { contentType: 'application/x-ndjson' },
      })
    );

    // Optional second lane: forward verbatim OTLP to a metrics backend for
    // dashboards. Configured by secret, so it can be added or swapped with a
    // `wrangler secret put` and no code change.
    if (env.METRICS_OTLP_URL) {
      ctx.waitUntil(
        fetch(env.METRICS_OTLP_URL, {
          method: 'POST',
          headers: {
            'content-type': 'application/json',
            authorization: env.METRICS_OTLP_AUTH || '',
          },
          body: JSON.stringify({ resourceLogs: [{ scopeLogs: [{ logRecords: records.map((r) => ({ body: { stringValue: JSON.stringify(r) } })) }] }] }),
        }).catch(() => {})
      );
    }

    return new Response(null, { status: 204 });
  },
};
