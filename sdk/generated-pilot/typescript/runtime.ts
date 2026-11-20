/** Handwritten JSON transport kernel, copied verbatim by sdkgen. No retries. */
export type JsonValue = null | boolean | number | string | JsonValue[] | { [key: string]: JsonValue };
export interface Schema {
  $ref?: string; type?: string; anyOf?: Schema[]; enum?: (string | boolean | number)[];
  properties?: Record<string, Schema>; required?: string[];
  additionalProperties?: boolean | Schema; items?: Schema;
  "x-headroom-opaque"?: string;
}
export class ProtocolError extends Error {}
export class APIError extends Error {
  constructor(readonly status: number, readonly headers: Headers, readonly body: Uint8Array) {
    super(`Headroom HTTP ${status}`);
  }
}
export function validate(value: unknown, schema: Schema, schemas: Record<string, Schema>, path = "$", depth = 0): void {
  if (depth > 100) throw new ProtocolError(`${path}: maximum JSON nesting exceeded`);
  if (schema.$ref) return validate(value, schemas[schema.$ref.split("/").at(-1)!], schemas, path, depth + 1);
  if (schema.anyOf) {
    for (const part of schema.anyOf) {
      try { validate(value, part, schemas, path, depth + 1); return; } catch (error) {
        if (!(error instanceof ProtocolError)) throw error;
      }
    }
    throw new ProtocolError(`${path}: outside nullable union`);
  }
  const object = typeof value === "object" && value !== null && !Array.isArray(value);
  const checks: Record<string, boolean> = {
    string: typeof value === "string", integer: typeof value === "number" && Number.isSafeInteger(value),
    number: typeof value === "number" && Number.isFinite(value), boolean: typeof value === "boolean",
    null: value === null, array: Array.isArray(value), object,
  };
  if (schema.type && !checks[schema.type]) throw new ProtocolError(`${path}: expected ${schema.type}`);
  if (schema.enum && !schema.enum.includes(value as never)) throw new ProtocolError(`${path}: unknown enum value`);
  if (object) {
    const record = value as Record<string, unknown>;
    for (const key of schema.required ?? []) {
      if (!Object.hasOwn(record, key)) throw new ProtocolError(`${path}.${key}: required field missing`);
    }
    for (const [key, item] of Object.entries(record)) {
      if (item === undefined && !(schema.required ?? []).includes(key)) continue; // JSON.stringify omits optional undefined fields.
      const rule = Object.hasOwn(schema.properties ?? {}, key) ? schema.properties![key] : (schema.additionalProperties ?? true);
      if (rule === false) throw new ProtocolError(`${path}.${key}: additional property forbidden`);
      validate(item, typeof rule === "object" ? rule : {}, schemas, `${path}.${key}`, depth + 1);
    }
  } else if (Array.isArray(value)) {
    value.forEach((item, i) => validate(item, schema.items ?? {}, schemas, `${path}[${i}]`, depth + 1));
  } else if (typeof value === "number") {
    if (!Number.isFinite(value) || (Number.isInteger(value) && !Number.isSafeInteger(value))) {
      throw new ProtocolError(`${path}: number cannot be represented safely in this target`);
    }
  } else if (value !== null && !["string", "boolean"].includes(typeof value)) {
    throw new ProtocolError(`${path}: not a JSON value`);
  }
}
export function pathSegment(value: string): string {
  if (typeof value !== "string" || ["", ".", ".."].includes(value)) throw new Error("Invalid path segment");
  return encodeURIComponent(value).replace(/[!'()*]/g, c => `%${c.charCodeAt(0).toString(16).toUpperCase()}`);
}
export interface TransportOptions {
  headers?: HeadersInit; timeoutMs?: number; maxResponseBytes?: number;
  fetch?: typeof globalThis.fetch;
}
export interface RequestOptions { signal?: AbortSignal }
export class Transport {
  private readonly base: string;
  private readonly options: TransportOptions;
  constructor(baseUrl = "http://localhost:8787", options: TransportOptions = {}) {
    const url = new URL(baseUrl);
    if (!["http:", "https:"].includes(url.protocol) || url.username || url.password || url.search || url.hash || /[\s\x00-\x1f]/.test(baseUrl)) {
      throw new Error("Invalid base URL");
    }
    if ((options.timeoutMs ?? 30000) <= 0 || (options.maxResponseBytes ?? 16777216) <= 0) throw new Error("Invalid transport limits");
    this.base = url.toString().replace(/\/+$/, "");
    this.options = {...options, headers: new Headers(options.headers)};
  }
  async request<T>(method: string, path: string, body: unknown, requestModel: string | null,
                   responseModel: string, schemas: Record<string, Schema>, options: RequestOptions = {}): Promise<T> {
    if (requestModel) validate(body, schemas[requestModel], schemas);
    const headers = new Headers(this.options.headers);
    headers.set("Accept", "application/json");
    if (requestModel) headers.set("Content-Type", "application/json");
    const controller = new AbortController();
    const abort = () => controller.abort(options.signal?.reason);
    options.signal?.addEventListener("abort", abort, {once: true});
    if (options.signal?.aborted) abort();
    const timer = setTimeout(() => controller.abort(new Error("Headroom request timed out")), this.options.timeoutMs ?? 30000);
    try {
      const response = await (this.options.fetch ?? globalThis.fetch)(this.base + path, {
        method, headers, body: requestModel ? JSON.stringify(body) : undefined,
        signal: controller.signal, redirect: "manual",
      });
      const chunks: Uint8Array[] = []; let length = 0;
      const reader = response.body?.getReader();
      if (reader) {
        try {
          while (true) {
            const {done, value} = await reader.read(); if (done) break;
            length += value.byteLength;
            if (length > (this.options.maxResponseBytes ?? 16777216)) {
              await reader.cancel(); throw new ProtocolError("Response exceeds configured byte limit");
            }
            chunks.push(value);
          }
        } finally { reader.releaseLock(); }
      }
      const bytes = new Uint8Array(length); let offset = 0;
      for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
      if (!response.ok) throw new APIError(response.status, response.headers, bytes);
      const media = (response.headers.get("content-type") ?? "").split(";", 1)[0].trim().toLowerCase();
      if (media !== "application/json" && !media.endsWith("+json")) throw new ProtocolError("Expected a JSON Content-Type");
      let value: unknown;
      try { value = JSON.parse(new TextDecoder("utf-8", {fatal: true}).decode(bytes)); }
      catch { throw new ProtocolError("Invalid JSON response"); }
      validate(value, schemas[responseModel], schemas);
      return value as T;
    } finally {
      clearTimeout(timer); options.signal?.removeEventListener("abort", abort);
    }
  }
}
