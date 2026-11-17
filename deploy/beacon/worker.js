/**
 * Headroom telemetry beacon receiver.
 *
 * This file is open source on purpose. It is the other half of the promise
 * made in headroom/telemetry/session.py: users can read exactly what the
 * client sends AND exactly what happens to it on arrival. "Trust us" is not a
 * privacy policy.
 *
 * Deployed at headroom-beacon.headroom-beacon.workers.dev (otlp.headroomlabs.ai
 * is planned but not yet routed; see wrangler.toml). Three jobs:
 *
 *   1. Allowlist. Drop every field not on ALLOWED_KEYS, and every nested
 *      field not described by NESTED, before anything is written. This is the only privacy control that works retroactively —
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
 * deanonymise install_id, so it is never read. Cloudflare's own per-request
 * invocation logs would record it regardless of this code, which is why
 * wrangler.toml turns them off (observability.logs.invocation_logs = false).
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

// What may sit UNDER each allowed key. Without this the allowlist stops at the
// top level and a nested field a client bug adds is stored verbatim -- the very
// leak the header says one deploy here can stop. Mirrors _Session.payload() and
// _payload_v2() in headroom/telemetry/session.py. A nested key absent here is
// dropped, so the ordering rule on ALLOWED_KEYS applies to it too: teach it
// here before a client sends it.
//
// Leaves are 'num' (a finite number), 'bool', or 'str': a token of at most 128
// characters with no whitespace, which is every string the client sends (slugs,
// hex ids, public model ids, versions). Free text does not fit it. `[spec]` is
// a list of spec; mapOf(spec) is an object whose KEYS are data (transform
// names, skip reasons, status codes), each key itself held to 'str'.
const MAP = Symbol('map');
const mapOf = (spec) => ({ [MAP]: spec });
const nums = (...keys) => Object.fromEntries(keys.map((k) => [k, 'num']));

const NESTED = {
  schema_version: 'num',
  session: {
    id: 'str', seq: 'num', duration_s: 'num', turns: 'num', ended: 'str', final: 'bool',
  },
  tokens: nums(
    'original', 'attempted', 'input', 'output', 'saved', 'tool_saved',
    'cache_read', 'cache_write', 'uncached'
  ),
  rates: nums(
    'saved_pct', 'eligible_pct', 'yield_pct', 'all_layers_saved_pct',
    'all_layers_yield_pct', 'cache_read_pct', 'overhead_pct'
  ),
  compression: {
    transforms: mapOf('num'),
    by_strategy: [{ strategy: 'str', ...nums('n', 'tokens_in', 'tokens_out') }],
    ...nums(
      'overhead_ms_total', 'latency_ms_total', 'overhead_ms_per_turn',
      'latency_ms_per_turn', 'passthrough_turns', 'response_cache_hits'
    ),
  },
  skips: mapOf('num'),
  sources: mapOf('num'),
  providers: ['str'],
  models: ['str'],
  failures: 'num',
  failure_statuses: mapOf('num'),
  // Emitted by an extension, not by session.py, so this repository does not
  // define its shape. Bounded rather than described: numbers, booleans and
  // tokens only, at most ROUTING_DEPTH levels deep. It gets a real schema here
  // once the extension's payload is written down.
  routing: 'any',
  quality: {
    ...nums('reread_tokens', 'reread_compressed_tokens', 'waste_turns'),
    waste: [{ kind: 'str', tokens: 'num' }],
  },
  hist: mapOf({ edges: ['num'], counts: ['num'] }),
  shapes: {
    by_content: [{ content: 'str', strategy: 'str', ...nums('n', 'tokens_in', 'tokens_out') }],
    by_tool: [{ shape: 'str', ...nums('n', 'tokens_in', 'tokens_out') }],
    truncated: 'bool',
  },
  cache: {
    gap_edges_s: ['num'], hits: ['num'], misses: ['num'],
    ...nums('write_5m', 'write_1h', 'inferred_turns'),
  },
  trajectory: {
    buckets: 'num', turns: ['num'], tokens: ['num'], saved: ['num'],
    // Runs over the closed _TURN_KINDS alphabet, at most MAX_TRAJECTORY_RUNS.
    kinds: /^(?:[cfepzso]\d{1,9}){0,64}$/,
    kinds_truncated: 'bool',
    msgs_max: 'num',
  },
  clients: [{ client: 'str', n: 'num' }],
  errors: { count: 'num', by_status: mapOf('num') },
  strata: [{ dim: 'str', value: 'str', n: 'num' }],
  config: [{ key: 'str', value: 'str' }],
};

// A key allowed at the top but given no shape would be dropped in full. Fail
// the deploy instead of shipping that.
if ([...ALLOWED_KEYS].sort().join() !== Object.keys(NESTED).sort().join()) {
  throw new Error('beacon: ALLOWED_KEYS and NESTED disagree');
}

const TOKEN = /^[\w.:/@+|-]{0,128}$/;
// A slug-shaped string can still be a URL or a credential. Model ids and the
// other slugs the client sends never carry a scheme or a well-known key prefix,
// so those are dropped even though they pass TOKEN.
const SECRETISH = /:\/\/|^(?:sk|pk|rk)[-_]|^(?:ghp|gho|ghu|ghs|ghr)_|^github_pat_|^glpat-|^xox[abposr]-|^(?:AKIA|ASIA)[A-Z0-9]{12}|^AIza|^hf_|^eyJ/i;
const isToken = (s) => TOKEN.test(s) && !SECRETISH.test(s);
const ROUTING_DEPTH = 4;

/** `value` reduced to what `spec` allows, or undefined when none of it is. */
function clean(spec, value, depth = 0) {
  if (spec === 'any') {
    if (depth > ROUTING_DEPTH) return undefined;
    spec = Array.isArray(value) ? ['any']
      : value && typeof value === 'object' ? mapOf('any')
      : typeof value === 'string' ? 'str'
      : typeof value === 'number' ? 'num' : 'bool';
  }
  if (spec === 'num') return Number.isFinite(value) ? value : undefined;
  if (spec === 'bool') return typeof value === 'boolean' ? value : undefined;
  if (spec === 'str') return typeof value === 'string' && isToken(value) ? value : undefined;
  if (spec instanceof RegExp) {
    return typeof value === 'string' && spec.test(value) ? value : undefined;
  }
  if (Array.isArray(spec)) {
    if (!Array.isArray(value)) return undefined;
    return value.map((v) => clean(spec[0], v, depth + 1)).filter((v) => v !== undefined);
  }
  if (!value || typeof value !== 'object' || Array.isArray(value)) return undefined;
  const out = {};
  if (spec[MAP]) {
    for (const [key, v] of Object.entries(value)) {
      const kept = isToken(key) ? clean(spec[MAP], v, depth + 1) : undefined;
      if (kept !== undefined) out[key] = kept;
    }
    return out;
  }
  for (const [key, s] of Object.entries(spec)) {
    if (!Object.hasOwn(value, key)) continue;
    const kept = clean(s, value[key], depth + 1);
    if (kept !== undefined) out[key] = kept;
  }
  return out;
}

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
// and never was: the endpoint is unauthenticated by design and the RATE_LIMIT
// binding (see fetch() and wrangler.toml) is what bounds a bad actor.
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

function pick(obj, allowed, specOf) {
  const out = {};
  if (!obj || typeof obj !== 'object') return out;
  for (const key of allowed) {
    if (!(key in obj)) continue;
    const kept = clean(specOf(key), obj[key]);
    if (kept !== undefined) out[key] = kept;
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
    // Every resource attribute the client sends is a plain string.
    const cleanResource = pick(resource, ALLOWED_RESOURCE, () => 'str');

    for (const sl of rl.scopeLogs || []) {
      for (const rec of sl.logRecords || []) {
        const body = unwrap(rec.body);
        if (!body || typeof body !== 'object') continue;
        records.push({
          ...pick(body, ALLOWED_KEYS, (key) => NESTED[key]),
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
// The corpus is one object per heartbeat. Analysis reads rollup/**, never
// sessions/**: each COMPLETE hour is collapsed into one object under rollup/,
// keeping only the highest-seq heartbeat per (install, session). Raw is left
// exactly as written, so any rollup can be rebuilt by deleting it.
//
// One invocation cannot do an hour on its own, and that is measured, not
// feared. A Worker may have six requests in flight, so reads ran at ~12
// objects/s whatever the code asked for, and a cron run is stopped at 15
// minutes: ~11k objects. The fleet passed that on 2026-08-19 and by October 98%
// of hours were being killed mid-read and writing nothing.
//
// So an hour is cut into slices and each slice is its own queue message, which
// means its own invocation with its own six connections and its own 15 minutes:
//
//   scheduled()   finds hours with no rollup and enqueues one message per slice
//   rollupSlice() reads one slice, writes rollup-parts/<hour>/<slice>
//   finalizeHour() once every slice is in, merges them into rollup/<hour>/
//
// Slices are the first hex character of the object name. Names are
// crypto.randomUUID() (see fetch()), so the sixteen slices are even, and the
// split needs nothing but a LIST prefix — which is also why it works on every
// object already in the bucket. Raise to two characters (256 slices) when a
// slice stops fitting; the per-slice cost falls sixteen-fold.

const SLICES = '0123456789abcdef';
// Requests in flight per invocation. Six is the platform's ceiling: asking for
// more only queues inside the runtime.
const POOL = 6;
// A queued hour that still has no rollup after this long is enqueued again:
// the safety net for a slice that ran out of retries. A second copy of a slice
// that did finish is cheap, because queue() reuses the part already written.
const REQUEUE_AFTER_MS = 3 * 3600_000;
// Hours enqueued per cron run, sized so they drain well inside REQUEUE_AFTER_MS
// (200 hours is 3,200 slices of a couple of minutes each, run in parallel). A
// bigger backlog is not stuck, the next run simply continues it.
const MAX_HOURS_PER_RUN = 200;
// Wait before a failed slice is tried again, so a short R2 outage does not burn
// every retry within seconds.
const RETRY_DELAY_S = 60;
// R2 deletes at most this many keys per call.
const DELETE_BATCH = 1000;

const partition = (d) =>
  `dt=${d.toISOString().slice(0, 10)}/hh=${d.toISOString().slice(11, 13)}`;

/** Every object under a prefix, following the cursor. */
async function listAll(env, options) {
  const objects = [];
  let cursor;
  do {
    const page = await env.CORPUS.list({ ...options, cursor });
    objects.push(...page.objects);
    cursor = page.truncated ? page.cursor : undefined;
  } while (cursor);
  return objects;
}

/**
 * Run `fn` over `items` with at most POOL calls in flight. Rejects on the first
 * failure and starts nothing new after it: the slice is going to be retried
 * whole, so further reads would be thrown away.
 */
async function pool(items, fn) {
  let next = 0;
  let failed = false;
  const worker = async () => {
    while (!failed && next < items.length) {
      try {
        await fn(items[next++]);
      } catch (err) {
        failed = true;
        throw err;
      }
    }
  };
  await Promise.all(Array.from({ length: Math.min(POOL, items.length) }, worker));
}

/**
 * Fold NDJSON `text` into `best`, keeping the highest-seq record per session.
 * Returns how many lines would not parse.
 *
 * A session heartbeats every 5 minutes carrying CUMULATIVE totals, so the
 * highest seq IS the whole session and every earlier row is a strict subset.
 * Sessions straddle hours, so readers still dedupe across rollups on this same
 * key — this only shrinks each hour.
 *
 * The winner is kept as freshly serialised text, not as a slice of `text`: a
 * slice would pin the whole source string in memory, and the merge reads
 * sixteen of them against a 128MB isolate.
 */
function keepBest(best, text) {
  let corrupt = 0;
  for (const line of text.split('\n')) {
    if (!line) continue;
    let rec;
    try {
      rec = JSON.parse(line);
    } catch {
      // Counted, but NOT a reason to abandon the slice. Content this Worker
      // itself wrote with JSON.stringify does not become valid later, so
      // blocking on it would strand the hour instead of losing one record.
      corrupt++;
      continue;
    }
    const id = `${rec.resource?.['headroom.install_id']} ${rec.session?.id}`;
    const seq = rec.session?.seq ?? 0;
    const prev = best.get(id);
    if (!prev || seq > prev.seq) best.set(id, { seq, line: JSON.stringify(rec) });
  }
  return corrupt;
}

const ndjson = (best) => [...best.values()].map((r) => r.line).join('\n');

/** True once an hour has its rollup (or its `empty` marker). */
async function hourDone(env, part) {
  const page = await env.CORPUS.list({ prefix: `rollup/${part}/`, limit: 1 });
  return page.objects.length > 0;
}

/**
 * One slice of one hour -> rollup-parts/<part>/<slice>.
 *
 * Throws if any listed object cannot be read, and writes nothing: the queue
 * redelivers the message, and a part that was short would silently become the
 * permanent record once merged. A slice with no objects still writes its (empty)
 * part, because "all sixteen present" is what finalizeHour waits for.
 */
export async function rollupSlice(env, part, slice) {
  const objects = await listAll(env, { prefix: `sessions/${part}/${slice}` });
  const best = new Map();
  let corrupt = 0;
  await pool(objects, async (o) => {
    const body = await env.CORPUS.get(o.key);
    // The key came from a LIST, so the object existed; treating a miss as
    // empty would quietly shrink the rollup.
    if (!body) throw new Error(`${o.key}: listed but unreadable`);
    // Folded as it arrives, so only the winners are held, never the slice.
    corrupt += keepBest(best, await body.text());
  });
  if (corrupt) {
    console.error(`rollup ${part}/${slice}: skipped ${corrupt} unparseable record(s)`);
  }
  await env.CORPUS.put(`rollup-parts/${part}/${slice}`, ndjson(best));
  return { read: objects.length, kept: best.size };
}

/**
 * Merge an hour's sixteen parts into rollup/<part>/, then remove the parts.
 * Returns false, having written nothing, while any slice is still missing.
 *
 * Safe to run twice at once: both runs read the same parts and write the same
 * bytes. The one that loses the race finds a part already deleted and stops.
 */
export async function finalizeHour(env, part) {
  const parts = await listAll(env, { prefix: `rollup-parts/${part}/` });
  const have = new Set(parts.map((o) => o.key.slice(o.key.lastIndexOf('/') + 1)));
  if (![...SLICES].every((s) => have.has(s))) return false;

  // One session's heartbeats are spread across slices, so the parts overlap and
  // the merge is a real dedupe, not a concatenation. Read one at a time: the
  // parts together are several times the size of the result.
  const best = new Map();
  for (const slice of SLICES) {
    const body = await env.CORPUS.get(`rollup-parts/${part}/${slice}`);
    if (!body) return false;
    keepBest(best, await body.text());
  }

  // A genuinely empty hour gets a marker rather than a zero-byte NDJSON that
  // every reader would have to special-case. Without it the hour stays
  // "missing" and is enqueued again for the life of the bucket.
  if (best.size === 0) {
    await env.CORPUS.put(`rollup/${part}/empty`, '');
  } else {
    await env.CORPUS.put(`rollup/${part}/data.ndjson`, ndjson(best), {
      httpMetadata: { contentType: 'application/x-ndjson' },
    });
  }
  await env.CORPUS.delete(parts.map((o) => o.key));
  return true;
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

/** `dt=…/hh=…` of a key under `prefix`, i.e. everything up to its last slash. */
const partOf = (key, prefix) => key.slice(prefix.length, key.lastIndexOf('/'));

export default {
  /**
   * Hourly cron. Enqueues every complete hour that has no rollup, back to the
   * oldest raw data. Does none of the reading itself — see the note above.
   */
  async scheduled(event, env) {
    // Backfill reaches all the way to the oldest surviving raw day, NOT a fixed
    // window. A fixed window silently strands everything older than it the
    // moment analysis stopped reading sessions/. Bounding by real data instead
    // means the floor rises only when a lifecycle rule expires the raw objects.
    const oldest = await oldestRawDay(env);
    if (!oldest) return;
    const floorMs = Date.parse(`${oldest}T00:00:00Z`);
    if (Number.isNaN(floorMs)) return;

    // Only list from the floor forward. Rollups older than the oldest raw day
    // can never be rebuilt, so enumerating them answers nothing.
    const done = new Set(
      (await listAll(env, { prefix: 'rollup/', startAfter: `rollup/dt=${oldest}` })).map((o) =>
        partOf(o.key, 'rollup/')
      )
    );
    const inFlight = await listAll(env, { prefix: 'rollup-parts/' });

    // Sweep what a finished hour left behind. finalizeHour removes its parts,
    // but three things can outlive it: a duplicate slice that was still reading
    // when the hour was merged, parts whose delete failed after the publish,
    // and a `queued` marker written after a very small hour had already merged.
    // Nothing else would ever remove them, and this prefix is listed every run.
    const leftover = inFlight
      .filter((o) => done.has(partOf(o.key, 'rollup-parts/')))
      .map((o) => o.key);
    for (let i = 0; i < leftover.length; i += DELETE_BATCH) {
      await env.CORPUS.delete(leftover.slice(i, i + DELETE_BATCH));
    }

    // When each in-flight hour was enqueued, so an hour still being worked on
    // is not enqueued again every time the cron fires.
    const queuedAt = new Map(
      inFlight
        .filter((o) => o.key.endsWith('/queued'))
        .map((o) => [partOf(o.key, 'rollup-parts/'), o.uploaded.getTime()])
    );

    // Newest first, so the freshest hour is never the one a backlog starves.
    // Starts one hour back: the current hour is still being written to.
    let enqueued = 0;
    for (
      let t = event.scheduledTime - 3600_000;
      t >= floorMs && enqueued < MAX_HOURS_PER_RUN;
      t -= 3600_000
    ) {
      const part = partition(new Date(t));
      if (done.has(part)) continue;
      const at = queuedAt.get(part);
      if (at !== undefined && event.scheduledTime - at < REQUEUE_AFTER_MS) continue;
      try {
        // Marker first: if the send then fails, the hour waits for the
        // re-enqueue instead of being sent twice by the next run.
        await env.CORPUS.put(`rollup-parts/${part}/queued`, '');
        await env.ROLLUP_QUEUE.sendBatch([...SLICES].map((slice) => ({ body: { part, slice } })));
      } catch (err) {
        // One hour failing to enqueue must not end the run for the rest.
        console.error(`rollup ${part}: enqueue failed: ${err}`);
      }
      enqueued++;
    }
  },

  /**
   * Queue consumer: one message is one slice of one hour. A slice that throws
   * is retried by the queue on its own, so one bad read costs that slice a
   * retry instead of costing the hour everything it had already read.
   */
  async queue(batch, env) {
    for (const message of batch.messages) {
      const { part, slice } = message.body;
      try {
        // A redelivery, or a re-enqueue that raced the merge: nothing to do,
        // and writing a part now would leave it behind for good.
        if (!(await hourDone(env, part))) {
          // The hour is closed, so a part that is already there is still
          // right: a second copy of the message only has to try the merge.
          if (!(await env.CORPUS.head(`rollup-parts/${part}/${slice}`))) {
            await rollupSlice(env, part, slice);
          }
          await finalizeHour(env, part);
        }
        message.ack();
      } catch (err) {
        console.error(`rollup ${part}/${slice} failed: ${err}`);
        message.retry({ delaySeconds: RETRY_DELAY_S });
      }
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

    // Abuse bound, checked before the body is read so a flood costs neither
    // the inflate nor an R2 write. Keyed on one constant rather than the
    // sender: a per-IP limit would mean reading cf-connecting-ip, which this
    // Worker never does (see the header). The price is that a flood also
    // sheds real events until the window rolls over -- acceptable for
    // fire-and-forget telemetry, and the cap sits ~10x above normal traffic.
    //
    // 429 is safe for old clients: only 400/415 make them drop gzip
    // (_GZIP_REFUSED_STATUSES in headroom/telemetry/session.py).
    //
    // Fails open: with no binding (local tests, an old wrangler.toml) or when
    // the rate-limit service itself errors, the upload proceeds. A limiter
    // outage must not drop every legitimate beacon; the cost is that a flood
    // during that outage is unbounded until the service recovers.
    if (env.RATE_LIMIT) {
      let limited = false;
      try {
        const { success } = await env.RATE_LIMIT.limit({ key: 'all' });
        limited = !success;
      } catch (err) {
        console.error('beacon: rate limiter unavailable, failing open:', String(err));
      }
      if (limited) return new Response('rate limited', { status: 429 });
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
    // the queue consumer above — analysis reads that, never this. The random
    // UUID is load-bearing there: its first character is the slice.
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
