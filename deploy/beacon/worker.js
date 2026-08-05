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

// Mirrors the payload built by _Session.payload(). A key absent here is
// dropped, not stored. Adding a metric means adding it here first — that
// friction is the point.
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

export default {
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

    const now = new Date();
    const day = now.toISOString().slice(0, 10);
    const hour = now.toISOString().slice(11, 13);
    // Hive-style partitioning so DuckDB can prune by date without a catalog.
    // ponytail: one object per request. At beacon volume that is a few hundred
    // thousand objects a month, which globs fine. Add a daily compaction job
    // when the file count starts to slow queries, not before.
    const key = `sessions/dt=${day}/hh=${hour}/${crypto.randomUUID()}.json`;
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
