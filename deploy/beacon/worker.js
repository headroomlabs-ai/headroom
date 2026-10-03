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

  // ---- schema v2 -----------------------------------------------------------
  //
  // Additive. Every v1 key above is untouched, so a v1 client keeps storing
  // exactly what it stores today and rows from the two versions union cleanly
  // (`schema_version` separates them, `union_by_name` handles the rest).
  //
  // Read the ordering rule at the top of this list literally: this deploy must
  // go out BEFORE the client release that starts sending these, or the first
  // weeks of the new signals are dropped on arrival and cannot be recovered.
  // Removing one line here is also the rollback -- it reverts that signal for
  // every install in the wild without a client release, which is the whole
  // reason the allowlist is server-side.

  // Regret. `reread_compressed_tokens` is the counter-pressure on `saved_pct`:
  // tokens the agent had to fetch again because compression removed them.
  // Without it a compression ratio is unfalsifiable.
  'quality',
  // Fixed-bucket distributions for quantities otherwise reported only as
  // session sums. Ships its own bucket edges, so no reader needs to know which
  // client version wrote a row.
  'hist',
  // (content_type x strategy x yield), and the same for JSON tool output keyed
  // by structural shape. The input term the corpus has never had.
  // Structural descriptors only -- never TOIN's structure_hash, which is a
  // digest of field names.
  'shapes',
  // Prompt-cache survival by gap since the previous turn: the empirical TTL
  // and minimum-prefix curve per provider.
  'cache',
  // Session shape over time -- log-spaced turn buckets plus a run-length
  // encoded turn-kind string from a closed seven-letter alphabet.
  'trajectory',
  // Identified client harness, per turn. `headroom.stack` answers this by
  // detection and resolves to a literal "proxy" for most of the fleet.
  'clients',
  // 4xx, kept strictly separate from the 5xx-only `failures` above so neither
  // metric changes meaning. 429 rate is the routing signal here.
  'errors',
  // output_shaper A/B strata as validated enums. Model family is not among
  // them -- see the note in session.py's payload().
  'strata',
  // Allowlisted, slug-validated configuration, so the corpus can tell
  // "compression underperformed" from "compression was switched off".
  'config',
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

// Cap on what ARRIVES, compressed or not.
//
// Was 64KB, chosen when the comment above it read "a beacon event is ~2KB" --
// i.e. 32x the payload. Schema v2 made a typical event ~8KB uncompressed, and
// a session that fills every table (the shape cross product is 10 content
// types x 13 strategies) reaches ~82KB. That silently eroded the multiple to
// under 1x: a busy session sending uncompressed -- the kill switch is set, or
// the gzip fallback fired -- would have been answered 413 and, correctly, not
// retried, so the event would simply be lost.
//
// Restored to ~3x the worst legitimate payload. This is not the abuse control
// and never was: the endpoint is unauthenticated by design and the WAF rate
// limit (60 req/min/IP, see wrangler.toml) is what bounds a bad actor. At 60
// requests a minute the difference between these two ceilings is 3.8 MB/min
// and 15 MB/min, neither of which is interesting to Cloudflare.
const MAX_BODY_BYTES = 256 * 1024;

// Cap on what an arriving body EXPANDS to. `raw.byteLength` cannot see this:
// a small gzip can expand to gigabytes, and a decompression bomb that is only
// noticed after it has been decompressed has already won. Enforced while the
// stream is read, not after. ~3x the worst legitimate payload, same as above.
const MAX_DECOMPRESSED_BYTES = 256 * 1024;

/**
 * Decompress a gzip body, refusing anything that expands past `limit`.
 *
 * Counts bytes as chunks arrive and aborts mid-stream, so a bomb costs the
 * limit rather than whatever it would have expanded to.
 */
export async function inflate(buffer, limit) {
  const stream = new Response(buffer).body.pipeThrough(new DecompressionStream('gzip'));
  const reader = stream.getReader();
  const chunks = [];
  let total = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      total += value.byteLength;
      if (total > limit) throw new Error('decompressed body too large');
      chunks.push(value);
    }
  } finally {
    // Releases the stream on the bomb path; a no-op once it has closed.
    reader.cancel().catch(() => {});
  }
  const joined = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    joined.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return new TextDecoder().decode(joined);
}

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
      // Sniff the gzip magic number rather than trusting Content-Encoding.
      // Three things can each independently decide whether a body arrives
      // compressed -- the client, Cloudflare's edge, and any proxy between
      // them -- and only the bytes know which of them acted. Keying off the
      // header instead means a body that something already decompressed gets
      // fed to DecompressionStream, or vice versa, and every upload 400s.
      //
      // It also means an old client that never sets the header keeps working
      // through the identical code path it uses today: no header, no magic
      // number, straight to TextDecoder.
      const bytes = new Uint8Array(raw);
      const gzipped = bytes.length >= 2 && bytes[0] === 0x1f && bytes[1] === 0x8b;
      const text = gzipped
        ? await inflate(raw, MAX_DECOMPRESSED_BYTES)
        : new TextDecoder().decode(raw);
      records = extract(JSON.parse(text));
    } catch {
      // Malformed input is not worth a retry storm from clients.
      //
      // This 400 is also load-bearing in the other direction: it is what a
      // client that gzipped against a Worker too old to inflate sees, and
      // _post_blocking treats a 4xx on a compressed body as "this endpoint
      // does not do gzip", falls back to sending it uncompressed, and stops
      // compressing for the life of the process. So deploying the client
      // before this Worker costs one retry per process, not the data.
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
