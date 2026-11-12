import { createRequire, syncBuiltinESMExports } from "node:module";

const nodeRequire = createRequire(import.meta.url);
const http = nodeRequire("node:http") as typeof import("node:http");
const https = nodeRequire("node:https") as typeof import("node:https");
const http2 = nodeRequire("node:http2") as typeof import("node:http2");
const childProcess = nodeRequire("node:child_process") as typeof import("node:child_process");
const fs = nodeRequire("node:fs") as typeof import("node:fs");

export const BASE_URL_HEADER = "x-headroom-base-url";
export const ORIGINAL_PATH_HEADER = "x-headroom-original-path";
export const PROJECT_HEADER = "x-headroom-project";
export const SESSION_TOKEN_HEADER = "x-headroom-session-token";
const PROXY_ENV = "HEADROOM_OPENCODE_TRANSPORT_PROXY_URL";
export const EXCLUDE_HOSTS_ENV = "HEADROOM_OPENCODE_EXCLUDE_HOSTS";
export const WIRE_ROOTS_ENV = "HEADROOM_OPENCODE_WIRE_ROOTS";
const STATE_KEY = Symbol.for("headroom.opencode.transport");

type FetchArgs = Parameters<typeof fetch>;
type HttpRequest = typeof http.request;
type HttpGet = typeof http.get;
type HttpsRequest = typeof https.request;
type HttpsGet = typeof https.get;
type Http2Connect = typeof http2.connect;
type ChildSpawn = typeof childProcess.spawn;
type ChildExec = typeof childProcess.exec;
type ChildExecFile = typeof childProcess.execFile;
type ChildFork = typeof childProcess.fork;

interface InstallOptions {
  proxyUrl: string;
  project?: string;
  excludeHosts?: string[];
  debug?: boolean;
  sessionToken?: string;
  wireRoots?: string | Iterable<unknown>;
}

interface TransportState {
  refs: number;
  proxyUrl: string;
  project: string | undefined;
  excludeHosts: string[];
  wireRoots: WireRoot[];
  debug: boolean;
  sessionToken: string | undefined;
  originalFetch: typeof fetch;
  originalHttpRequest: HttpRequest;
  originalHttpGet: HttpGet;
  originalHttpsRequest: HttpsRequest;
  originalHttpsGet: HttpsGet;
  originalHttp2Connect: Http2Connect;
  originalChildSpawn: ChildSpawn;
  originalChildExec: ChildExec;
  originalChildExecFile: ChildExecFile;
  originalChildFork: ChildFork;
  originalEnv: Record<string, string | undefined>;
}

interface GlobalWithHeadroomTransport {
  [STATE_KEY]?: TransportState;
}

interface NodeRequestParts {
  url?: URL;
  options: Record<string, unknown>;
  callback?: (...args: unknown[]) => unknown;
}

function getState(): TransportState | undefined {
  return (globalThis as GlobalWithHeadroomTransport)[STATE_KEY];
}

function setState(state: TransportState | undefined): void {
  (globalThis as GlobalWithHeadroomTransport)[STATE_KEY] = state;
}

// ponytail: the shim only exists next to the checkout build
// (plugins/opencode/dist/). The wheel shipped the entry bundle alone, so
// `--import=<missing file>` killed every Node child at startup — including
// OpenCode's stdio MCP servers (issue #2798). No shim on disk, no injection:
// children go direct instead of dying. Upgrade path is bundling the shim into
// _dist/ so wheel installs get child-process routing back.
function shimImportSpecifier(): string | undefined {
  const shim = new URL("../hook-shim/handler.js", import.meta.url);
  return fs.existsSync(shim) ? shim.href : undefined;
}

function withNodeImportOption(existing: string | undefined, shim: string): string {
  const parts = existing?.trim() ? existing.trim().split(/\s+/) : [];
  const alreadyPresent = parts.some((part, index) => {
    return part === `--import=${shim}` || (part === "--import" && parts[index + 1] === shim);
  });
  if (!alreadyPresent) {
    parts.push(`--import=${shim}`);
  }
  return parts.join(" ");
}

// The exported variable mirrors the EFFECTIVE list, not merely a non-empty
// one: an explicit `excludeHosts: []` overrides a pre-existing variable for
// this process, so a child that inherited the stale value would bypass hosts
// the parent routes. Delete it when the resolved list is empty.
function withExcludeHostsEnv(env: NodeJS.ProcessEnv, excludeHosts: string[]): void {
  if (excludeHosts.length > 0) {
    env[EXCLUDE_HOSTS_ENV] = excludeHosts.join(",");
  } else {
    delete env[EXCLUDE_HOSTS_ENV];
  }
}

// Same rule as excludeHosts: the child must inherit this process's roots, so
// the variable always mirrors the effective set and is removed when empty.
function withWireRootsEnv(env: NodeJS.ProcessEnv, wireRoots: WireRoot[]): void {
  if (wireRoots.length > 0) {
    env[WIRE_ROOTS_ENV] = wireRoots.map((root) => `${root.origin}${root.path}`).join(",");
  } else {
    delete env[WIRE_ROOTS_ENV];
  }
}

function withShimEnv(
  env: NodeJS.ProcessEnv | Record<string, unknown> | undefined,
  proxyUrl: string,
  excludeHosts: string[],
  wireRoots: WireRoot[],
): NodeJS.ProcessEnv {
  const nextEnv = { ...(env ?? process.env) } as NodeJS.ProcessEnv;
  nextEnv[PROXY_ENV] = proxyUrl;
  withExcludeHostsEnv(nextEnv, excludeHosts);
  withWireRootsEnv(nextEnv, wireRoots);
  const shim = shimImportSpecifier();
  if (shim) {
    nextEnv.NODE_OPTIONS = withNodeImportOption(nextEnv.NODE_OPTIONS, shim);
  }
  return nextEnv;
}

// installProcessEnv writes these process-wide keys; uninstall puts them back.
const PROCESS_ENV_KEYS = [PROXY_ENV, EXCLUDE_HOSTS_ENV, WIRE_ROOTS_ENV, "NODE_OPTIONS"] as const;

function snapshotProcessEnv(): Record<string, string | undefined> {
  return Object.fromEntries(PROCESS_ENV_KEYS.map((key) => [key, process.env[key]]));
}

function restoreProcessEnv(saved: Record<string, string | undefined>): void {
  for (const key of PROCESS_ENV_KEYS) {
    if (saved[key] === undefined) {
      delete process.env[key];
    } else {
      process.env[key] = saved[key];
    }
  }
}

function installProcessEnv(proxyUrl: string, excludeHosts: string[], wireRoots: WireRoot[]): void {
  process.env[PROXY_ENV] = proxyUrl;
  withExcludeHostsEnv(process.env, excludeHosts);
  withWireRootsEnv(process.env, wireRoots);
  const shim = shimImportSpecifier();
  if (shim) {
    process.env.NODE_OPTIONS = withNodeImportOption(process.env.NODE_OPTIONS, shim);
  }
}

function isOptions(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value) && !(value instanceof URL);
}

function injectOptionsEnv(args: unknown[], optionIndex: number, state: TransportState): unknown[] {
  const nextArgs = [...args];
  const callback = typeof nextArgs.at(-1) === "function" ? nextArgs.pop() : undefined;
  const existing = isOptions(nextArgs[optionIndex]) ? { ...(nextArgs[optionIndex] as Record<string, unknown>) } : {};
  existing.env = withShimEnv(
    existing.env as NodeJS.ProcessEnv | undefined,
    state.proxyUrl,
    state.excludeHosts,
    state.wireRoots,
  );
  if (process.platform === "win32" && existing.windowsHide === undefined) {
    existing.windowsHide = true;
  }

  if (isOptions(nextArgs[optionIndex])) {
    nextArgs[optionIndex] = existing;
  } else {
    nextArgs.splice(optionIndex, 0, existing);
  }

  if (callback) {
    nextArgs.push(callback);
  }
  return nextArgs;
}

function wrapSpawn(originalSpawn: ChildSpawn): ChildSpawn {
  return function headroomSpawn(this: unknown, ...args: unknown[]) {
    const state = getState();
    if (!state) {
      return Reflect.apply(originalSpawn, this, args);
    }
    const optionIndex = Array.isArray(args[1]) ? 2 : 1;
    return Reflect.apply(originalSpawn, this, injectOptionsEnv(args, optionIndex, state));
  } as ChildSpawn;
}

function wrapExec(originalExec: ChildExec): ChildExec {
  return function headroomExec(this: unknown, ...args: unknown[]) {
    const state = getState();
    if (!state) {
      return Reflect.apply(originalExec, this, args);
    }
    return Reflect.apply(originalExec, this, injectOptionsEnv(args, 1, state));
  } as ChildExec;
}

function wrapExecFile(originalExecFile: ChildExecFile): ChildExecFile {
  return function headroomExecFile(this: unknown, ...args: unknown[]) {
    const state = getState();
    if (!state) {
      return Reflect.apply(originalExecFile, this, args);
    }
    const optionIndex = Array.isArray(args[1]) ? 2 : 1;
    return Reflect.apply(originalExecFile, this, injectOptionsEnv(args, optionIndex, state));
  } as ChildExecFile;
}

function wrapFork(originalFork: ChildFork): ChildFork {
  return function headroomFork(this: unknown, ...args: unknown[]) {
    const state = getState();
    if (!state) {
      return Reflect.apply(originalFork, this, args);
    }
    const optionIndex = Array.isArray(args[1]) ? 2 : 1;
    return Reflect.apply(originalFork, this, injectOptionsEnv(args, optionIndex, state));
  } as ChildFork;
}

function normalizeProxyUrl(proxyUrl: string): URL {
  return new URL(proxyUrl);
}

// A proxy URL that does not parse must not break traffic that is not LLM
// traffic: the caller falls through to the original function. Reported once
// per value, not once per request.
const reportedProxyUrls = new Set<string>();

function usableProxyUrl(proxyUrl: string): URL | undefined {
  try {
    const proxy = normalizeProxyUrl(proxyUrl);
    // "user:pass@host:8787" parses with scheme "user:"; only http(s) can be routed to.
    if (proxy.protocol !== "http:" && proxy.protocol !== "https:") throw new TypeError("unsupported scheme");
    return proxy;
  } catch {
    if (!reportedProxyUrls.has(proxyUrl)) {
      reportedProxyUrls.add(proxyUrl);
      // The value is never logged: a malformed URL can still carry user:password@.
      console.warn(`headroom: the configured proxy URL is ${proxyUrl === "" ? "empty" : "not a valid http(s) URL"}; requests are not routed`);
    }
    return undefined;
  }
}

// A hostname reaches this module as WHATWG `URL.hostname`, so a few spellings of
// the same host are still distinct strings: "example.com." is the same name as
// "example.com" (a fully qualified name), and an IPv4-mapped IPv6 loopback
// arrives canonicalized as "[::ffff:7f00:1]" - never as "127.0.0.1" or "::1".
// Compare canonical forms, or the loopback and exclude checks miss (#4747).
function canonicalHostname(hostname: string): string {
  return hostname.toLowerCase().replace(/^\[|\]$/g, "").replace(/\.+$/, "");
}

function isLoopback(hostname: string): boolean {
  const normalized = canonicalHostname(hostname);
  if (normalized === "localhost" || normalized === "::1" || normalized === "0.0.0.0") {
    return true;
  }
  // 127.0.0.0/8 is entirely loopback, not just 127.0.0.1.
  if (isLoopbackIpv4(normalized)) {
    return true;
  }
  const mapped = /^::ffff:(.+)$/.exec(normalized)?.[1];
  if (!mapped) {
    return false;
  }
  // URL canonicalization writes the embedded IPv4 in hex ("::ffff:7f00:1"),
  // but accept the dotted form too, which some non-WHATWG callers pass through.
  if (isLoopbackIpv4(mapped)) {
    return true;
  }
  // Two remaining groups are the full 32-bit address: 127.1.2.3 is
  // "::ffff:7f01:203", so the last two octets share the low group.
  const groups = /^([0-9a-f]{1,4}):([0-9a-f]{1,4})$/.exec(mapped);
  if (!groups) {
    return false;
  }
  const address = (parseInt(groups[1], 16) << 16) | parseInt(groups[2], 16);
  return address >>> 24 === 127;
}

function isLoopbackIpv4(host: string): boolean {
  return /^127\.\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(host);
}

// "example.com", ".example.com" and "*.example.com" all mean the host itself
// plus every subdomain. Entries are bare hosts: no scheme, port, or path. A
// string is the comma-separated env form; plugin options arrive from untyped
// JSON, so a lone string there is treated the same way instead of iterated
// character by character.
export function normalizeExcludeHosts(entries: string | Iterable<unknown>): string[] {
  const hosts = new Set<string>();
  for (const entry of typeof entries === "string" ? entries.split(",") : entries) {
    const host = canonicalHostname(String(entry).trim()).replace(/^(?:\*\.|\.)/, "");
    if (host) {
      hosts.add(host);
    }
  }
  return [...hosts];
}

function isExcludedHost(hostname: string, excludeHosts: string[]): boolean {
  const normalized = canonicalHostname(hostname);
  return excludeHosts.some((host) => normalized === host || normalized.endsWith(`.${host}`));
}

// A provider base URL the user configured (for example
// https://gateway.invalid/tenant/acme) says where that provider mounts its
// wire: the client appends /chat/completions to exactly that path. These are
// facts about the configuration, not guesses from a path's shape, so they are
// matched by origin and path prefix instead of by root name (#4747 review).
//
// This is what keeps arbitrary compatible gateways routable. A root-name
// allowlist can only ever recognise the names we thought of, so
// /tenant/acme/chat/completions fell through while OpenCode 1.x still sent it
// through the patched fetch - silently disabling compression for a base that
// modelBaseRoutesThroughProxy accepts.
interface WireRoot {
  origin: string;
  path: string;
}

function normalizeWireRoots(entries: string | Iterable<unknown>): WireRoot[] {
  const roots = new Map<string, WireRoot>();
  for (const entry of typeof entries === "string" ? entries.split(",") : entries) {
    const raw = String(entry).trim();
    if (!raw) {
      continue;
    }
    try {
      const url = new URL(raw);
      if (url.protocol !== "http:" && url.protocol !== "https:") {
        continue;
      }
      const path = url.pathname.replace(/\/+$/, "");
      roots.set(`${url.origin}${path}`, { origin: url.origin, path });
    } catch {
      // Not a URL: nothing to match against. A bad entry must not disable the
      // root set, so skip it rather than throwing.
    }
  }
  return [...roots.values()];
}

function isConfiguredWireRoot(origin: string, pathname: string, wireRoots: WireRoot[]): boolean {
  return wireRoots.some(
    (root) =>
      root.origin === origin &&
      (root.path === "" ? pathname.startsWith("/") : pathname === root.path || pathname.startsWith(`${root.path}/`)),
  );
}

// Only recognized LLM API endpoints route through Headroom; any other path
// (WebFetch, registries, GitHub, unknown services) must reach its original URL
// untouched.
//
// The endpoint suffix alone cannot decide this. Discord's
// /api/v10/channels/123/messages and a forms API's /forms/123/responses end in
// the same two segments, so a bare suffix match rerouted them - with the
// caller's Authorization header - to the proxy (#4747). A path therefore also
// has to be rooted where providers mount a wire:
//
// * the segment right before the endpoint is a version ("v1", "v1beta", "v4")
//   or a base a compatible gateway mounts the wire directly under ("api"),
//   which keeps prefixed gateways working (/api/coding/paas/v4/chat/completions,
//   /base/v1/messages, /zen/v1/responses, and a custom base URL of
//   https://gateway.example/api), or
// * the path contains a named wire root, which is what Azure OpenAI needs: its
//   deployment name sits between /openai/ and the endpoint
//   (/openai/deployments/{name}/chat/completions).
//
// A bare /messages or /responses on a provider origin still counts: there is no
// segment at all.
//
// The immediate segment is what keeps this tight. "api" is a root only when it
// abuts the endpoint, so /api/chat/completions routes while Discord's
// /api/v10/channels/123/messages (immediate segment "123") does not. For the
// same reason a version segment counts only in that position, not anywhere.
//
// The residual over-match is documented, not solved: /api/v2/messages still
// routes. Telling that apart from a provider needs the host or the body, and
// excludeHosts already covers any host a user controls.
const NAMED_WIRE_ROOTS = new Set(["backend-api", "codex", "openai", "anthropic", "zen"]);

// Custom OpenAI-compatible gateways commonly mount the wire directly under one
// of these, with no version segment: a model base URL of `.../api` plus
// `/chat/completions` is a real configuration (#4747 review).
const IMMEDIATE_WIRE_ROOTS = new Set(["api"]);

function isVersionSegment(segment: string): boolean {
  return /^v\d+[a-z0-9]*$/i.test(segment);
}

function hasWireRootBefore(body: string): boolean {
  const segments = body.split("/").filter(Boolean);
  const last = segments.at(-1);
  return (
    last === undefined ||
    isVersionSegment(last) ||
    IMMEDIATE_WIRE_ROOTS.has(last) ||
    segments.some((s) => NAMED_WIRE_ROOTS.has(s))
  );
}

// Native Gemini model-generation endpoints use colon-action suffixes
// (:generateContent, :streamGenerateContent). These are matched by exact
// suffix so that lookalike paths containing the marker but not ending with it
// (e.g. /v1/models/gemini:generateContent/status) remain unrouted. They are
// specific enough to need no wire root.
function isLlmEndpointPath(pathname: string, origin = "", wireRoots: WireRoot[] = []): boolean {
  if (pathname.endsWith(":generateContent") || pathname.endsWith(":streamGenerateContent")) {
    return true;
  }
  for (const suffix of ["/chat/completions", "/responses", "/messages"]) {
    if (!pathname.endsWith(suffix)) {
      continue;
    }
    // A configured base URL is authoritative for its own origin: the client
    // appends the endpoint to it. Still require the suffix, so a WebFetch to
    // some other path on that host stays out (#4747 review).
    if (isConfiguredWireRoot(origin, pathname, wireRoots)) {
      return true;
    }
    return hasWireRootBefore(pathname.slice(0, -suffix.length));
  }
  return false;
}

function isRoutableUpstream(url: URL, proxy: URL, excludeHosts: string[]): boolean {
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    return false;
  }
  if (isLoopback(url.hostname)) {
    return false;
  }
  if (url.origin === proxy.origin) {
    return false;
  }
  if (isExcludedHost(url.hostname, excludeHosts)) {
    return false;
  }
  return true;
}

// Every LLM endpoint matched above is a POST. A GET/DELETE/... to a path that
// merely ends in /messages or /responses (chat APIs, form responses) is not an
// LLM call and must reach its original URL (#4417). A missing method means
// GET, matching fetch and node:http defaults.
function isLlmMethod(method: unknown): boolean {
  return typeof method === "string" && method.toUpperCase() === "POST";
}

function shouldRoute(
  url: URL,
  proxy: URL,
  excludeHosts: string[],
  method: unknown,
  wireRoots: WireRoot[] = [],
): boolean {
  if (!isLlmMethod(method) || !isRoutableUpstream(url, proxy, excludeHosts)) {
    return false;
  }
  return isLlmEndpointPath(url.pathname, url.origin, wireRoots);
}

// OpenCode 2.x hands the plugin model base URLs (e.g.
// https://opencode.ai/zen/go/v1), not inference endpoints: setup appends the
// wire path when it rewrites the model, so the endpoint-suffix check cannot
// apply. Eligibility is the shared host/protocol contract only — http(s),
// not loopback, not the proxy itself, not an excluded host.
export function modelBaseRoutesThroughProxy(
  baseUrl: string,
  proxyUrl: string,
  excludeHosts: string[] = [],
): boolean {
  try {
    return isRoutableUpstream(new URL(baseUrl), normalizeProxyUrl(proxyUrl), excludeHosts);
  } catch {
    return false;
  }
}

function routedUrl(upstream: URL, proxy: URL): URL {
  return new URL(`${upstream.pathname}${upstream.search}`, proxy.origin);
}

function normalizedOpenAiProxyPath(pathname: string): string | undefined {
  if (pathname.endsWith("/chat/completions")) {
    return "/v1/chat/completions";
  }
  if (pathname.endsWith("/responses")) {
    return "/v1/responses";
  }
  return undefined;
}

function routedUrlForOpenCode(upstream: URL, proxy: URL): { url: URL; originalPath: string | undefined } {
  const normalizedPath = normalizedOpenAiProxyPath(upstream.pathname);
  if (!normalizedPath) {
    return {
      url: routedUrl(upstream, proxy),
      originalPath: undefined,
    };
  }

  return {
    url: new URL(`${normalizedPath}${upstream.search}`, proxy.origin),
    originalPath: upstream.pathname,
  };
}

function requestUrl(input: RequestInfo | URL): URL {
  if (input instanceof Request) {
    return new URL(input.url);
  }
  if (input instanceof URL) {
    return input;
  }
  return new URL(String(input));
}

function mergeFetchHeaders(
  input: RequestInfo | URL,
  init: RequestInit | undefined,
  upstream: URL | undefined,
  originalPath: string | undefined = undefined,
  project: string | undefined = undefined,
  sessionToken: string | undefined = undefined,
): Headers {
  const headers = new Headers(input instanceof Request ? input.headers : undefined);
  if (init?.headers) {
    new Headers(init.headers).forEach((value, key) => headers.set(key, value));
  }
  if (upstream) {
    headers.set(BASE_URL_HEADER, upstream.origin);
    headers.delete("host");
  }
  if (originalPath) {
    headers.set(ORIGINAL_PATH_HEADER, originalPath);
  }
  if (project) {
    headers.set(PROJECT_HEADER, project);
  }
  if (sessionToken) {
    headers.set(SESSION_TOKEN_HEADER, sessionToken);
  }
  return headers;
}

function withRoutedFetchInput(
  input: RequestInfo | URL,
  init: RequestInit | undefined,
  proxy: URL,
  project: string | undefined,
  excludeHosts: string[],
  sessionToken: string | undefined,
  wireRoots: WireRoot[] = [],
): FetchArgs {
  const upstream = requestUrl(input);
  const method = init?.method ?? (input instanceof Request ? input.method : "GET");
  if (!shouldRoute(upstream, proxy, excludeHosts, method, wireRoots)) {
    return [input, init];
  }

  const { url: nextUrl, originalPath } = routedUrlForOpenCode(upstream, proxy);
  const nextInit = {
    ...init,
    headers: mergeFetchHeaders(input, init, upstream, originalPath, project, sessionToken),
  };

  if (input instanceof Request) {
    return [new Request(nextUrl, input), nextInit];
  }
  return [nextUrl, nextInit];
}

function splitNodeArgs(args: unknown[], defaultProtocol: string): NodeRequestParts {
  const callback = typeof args.at(-1) === "function" ? (args.at(-1) as (...args: unknown[]) => unknown) : undefined;
  const withoutCallback = callback ? args.slice(0, -1) : args;
  const [first, second] = withoutCallback;
  const options = typeof second === "object" && second !== null ? { ...(second as Record<string, unknown>) } : {};

  if (first instanceof URL) {
    return { url: first, options, callback };
  }
  if (typeof first === "string") {
    try {
      return { url: new URL(first), options, callback };
    } catch {
      return { options, callback };
    }
  }
  if (typeof first === "object" && first !== null) {
    const requestOptions = { ...(first as Record<string, unknown>), ...options };
    return { url: urlFromRequestOptions(requestOptions, defaultProtocol), options: requestOptions, callback };
  }
  return { options, callback };
}

// An options object without `protocol` means the module's own default:
// https.request({ hostname }) is https, http.request({ hostname }) is http.
function urlFromRequestOptions(options: Record<string, unknown>, defaultProtocol: string): URL | undefined {
  const protocol = String(options.protocol ?? defaultProtocol);
  if (protocol !== "http:" && protocol !== "https:") {
    return undefined;
  }

  const hostValue = options.hostname ?? options.host;
  if (!hostValue) {
    return undefined;
  }

  const hostname = String(hostValue).replace(/:\d+$/, "");
  const port = options.port ? `:${String(options.port)}` : "";
  const path = String(options.path ?? "/");
  try {
    return new URL(`${protocol}//${hostname}${port}${path}`);
  } catch {
    return undefined;
  }
}

function headersForNodeRequest(
  options: Record<string, unknown>,
  upstream: URL,
  originalPath: string | undefined,
  project: string | undefined,
  sessionToken: string | undefined,
): Record<string, string> {
  const headers = new Headers(options.headers as HeadersInit | undefined);
  headers.set(BASE_URL_HEADER, upstream.origin);
  if (originalPath) {
    headers.set(ORIGINAL_PATH_HEADER, originalPath);
  }
  if (project) {
    headers.set(PROJECT_HEADER, project);
  }
  if (sessionToken) {
    headers.set(SESSION_TOKEN_HEADER, sessionToken);
  }
  headers.delete("host");

  const result: Record<string, string> = {};
  headers.forEach((value, key) => {
    result[key] = value;
  });
  return result;
}

function routedNodeOptions(
  parts: NodeRequestParts,
  proxy: URL,
  project: string | undefined,
  excludeHosts: string[],
  sessionToken: string | undefined,
  wireRoots: WireRoot[] = [],
): Record<string, unknown> | undefined {
  if (!parts.url || !shouldRoute(parts.url, proxy, excludeHosts, parts.options.method ?? "GET", wireRoots)) {
    return undefined;
  }

  const { url: nextUrl, originalPath } = routedUrlForOpenCode(parts.url, proxy);
  const {
    agent: _agent,
    auth: _auth,
    createConnection: _createConnection,
    defaultPort: _defaultPort,
    family: _family,
    headers: _headers,
    host: _host,
    hostname: _hostname,
    href: _href,
    lookup: _lookup,
    path: _path,
    pathname: _pathname,
    port: _port,
    protocol: _protocol,
    search: _search,
    servername: _servername,
    setHost: _setHost,
    ...rest
  } = parts.options;

  return {
    ...rest,
    protocol: nextUrl.protocol,
    hostname: nextUrl.hostname,
    port: nextUrl.port || undefined,
    path: `${nextUrl.pathname}${nextUrl.search}`,
    headers: headersForNodeRequest(parts.options, parts.url, originalPath, project, sessionToken),
  };
}

function wrapRequest(
  originalHttpRequest: HttpRequest,
  originalHttpsRequest: HttpsRequest,
  originalRequest: HttpRequest | HttpsRequest,
): HttpRequest | HttpsRequest {
  return function headroomRequest(this: unknown, ...args: unknown[]) {
    const state = getState();
    if (!state) {
      return Reflect.apply(originalRequest, this, args);
    }

    const proxy = usableProxyUrl(state.proxyUrl);
    if (!proxy) {
      return Reflect.apply(originalRequest, this, args);
    }
    const parts = splitNodeArgs(args, originalRequest === originalHttpsRequest ? "https:" : "http:");
    const nextOptions = routedNodeOptions(
      parts,
      proxy,
      state.project,
      state.excludeHosts,
      state.sessionToken,
      state.wireRoots,
    );
    if (!nextOptions) {
      return Reflect.apply(originalRequest, this, args);
    }

    const targetRequest = proxy.protocol === "https:" ? originalHttpsRequest : originalHttpRequest;
    const nextArgs = parts.callback ? [nextOptions, parts.callback] : [nextOptions];
    return Reflect.apply(targetRequest, this, nextArgs);
  } as HttpRequest | HttpsRequest;
}

function wrapGet(request: HttpRequest | HttpsRequest): HttpGet | HttpsGet {
  return function headroomGet(this: unknown, ...args: unknown[]) {
    const req = Reflect.apply(request, this, args);
    req.end();
    return req;
  } as HttpGet | HttpsGet;
}

// http2.connect() has no request path at connect time, so the authority alone
// cannot prove LLM traffic. Direct HTTP/2 connections always pass through
// untouched: rejecting external authorities turned WebFetch into a proxy
// error (#3633).
function wrapHttp2Connect(originalConnect: Http2Connect): Http2Connect {
  return function headroomHttp2Connect(this: unknown, ...args: unknown[]) {
    return Reflect.apply(originalConnect, this, args);
  } as Http2Connect;
}

export function installHeadroomTransport(options: InstallOptions): () => void {
  const excludeHosts = normalizeExcludeHosts(options.excludeHosts ?? process.env[EXCLUDE_HOSTS_ENV] ?? "");
  const wireRoots = normalizeWireRoots(options.wireRoots ?? process.env[WIRE_ROOTS_ENV] ?? "");
  const existing = getState();
  if (existing) {
    existing.refs += 1;
    existing.proxyUrl = options.proxyUrl;
    existing.project = options.project;
    existing.excludeHosts = excludeHosts;
    existing.wireRoots = wireRoots;
    existing.debug = Boolean(options.debug);
    existing.sessionToken = options.sessionToken;
    installProcessEnv(options.proxyUrl, excludeHosts, wireRoots);
    return () => uninstallHeadroomTransport();
  }

  const state: TransportState = {
    refs: 1,
    proxyUrl: options.proxyUrl,
    project: options.project,
    excludeHosts,
    wireRoots,
    debug: Boolean(options.debug),
    sessionToken: options.sessionToken,
    originalFetch: globalThis.fetch,
    originalHttpRequest: http.request,
    originalHttpGet: http.get,
    originalHttpsRequest: https.request,
    originalHttpsGet: https.get,
    originalHttp2Connect: http2.connect,
    originalChildSpawn: childProcess.spawn,
    originalChildExec: childProcess.exec,
    originalChildExecFile: childProcess.execFile,
    originalChildFork: childProcess.fork,
    originalEnv: snapshotProcessEnv(),
  };

  setState(state);
  installProcessEnv(options.proxyUrl, excludeHosts, wireRoots);
  globalThis.fetch = async (...args: FetchArgs) => {
    const current = getState();
    if (!current) {
      return state.originalFetch(...args);
    }
    const proxy = usableProxyUrl(current.proxyUrl);
    if (!proxy) {
      return state.originalFetch(...args);
    }
    const [nextInput, nextInit] = withRoutedFetchInput(
      args[0],
      args[1],
      proxy,
      current.project,
      current.excludeHosts,
      current.sessionToken,
      current.wireRoots,
    );
    return state.originalFetch(nextInput, nextInit);
  };

  http.request = wrapRequest(state.originalHttpRequest, state.originalHttpsRequest, state.originalHttpRequest) as HttpRequest;
  https.request = wrapRequest(state.originalHttpRequest, state.originalHttpsRequest, state.originalHttpsRequest) as HttpsRequest;
  http.get = wrapGet(http.request) as HttpGet;
  https.get = wrapGet(https.request) as HttpsGet;
  http2.connect = wrapHttp2Connect(state.originalHttp2Connect);
  childProcess.spawn = wrapSpawn(state.originalChildSpawn);
  childProcess.exec = wrapExec(state.originalChildExec);
  childProcess.execFile = wrapExecFile(state.originalChildExecFile);
  childProcess.fork = wrapFork(state.originalChildFork);
  syncBuiltinESMExports();

  return () => uninstallHeadroomTransport();
}

// The configured bases are only known after a round trip to OpenCode's server,
// which may still be starting. Routing must not wait for it, so this widens the
// rule afterwards instead: requests in between use the path heuristics.
export function setHeadroomWireRoots(wireRoots: string): void {
  const state = getState();
  if (!state) {
    return;
  }
  const existing = new Set(state.wireRoots.map((root) => `${root.origin}${root.path}`));
  const added = normalizeWireRoots(wireRoots).filter((root) => !existing.has(`${root.origin}${root.path}`));
  if (added.length === 0) {
    return;
  }
  state.wireRoots = [...state.wireRoots, ...added];
  withWireRootsEnv(process.env, state.wireRoots);
}

export function uninstallHeadroomTransport(): void {
  const state = getState();
  if (!state) {
    return;
  }

  state.refs -= 1;
  if (state.refs > 0) {
    return;
  }

  globalThis.fetch = state.originalFetch;
  http.request = state.originalHttpRequest;
  http.get = state.originalHttpGet;
  https.request = state.originalHttpsRequest;
  https.get = state.originalHttpsGet;
  http2.connect = state.originalHttp2Connect;
  childProcess.spawn = state.originalChildSpawn;
  childProcess.exec = state.originalChildExec;
  childProcess.execFile = state.originalChildExecFile;
  childProcess.fork = state.originalChildFork;
  restoreProcessEnv(state.originalEnv);
  syncBuiltinESMExports();
  setState(undefined);
}
