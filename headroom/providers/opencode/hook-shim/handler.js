// src/transport.ts
import { createRequire, syncBuiltinESMExports } from "module";
var nodeRequire = createRequire(import.meta.url);
var http = nodeRequire("node:http");
var https = nodeRequire("node:https");
var http2 = nodeRequire("node:http2");
var childProcess = nodeRequire("node:child_process");
var fs = nodeRequire("node:fs");
var BASE_URL_HEADER = "x-headroom-base-url";
var ORIGINAL_PATH_HEADER = "x-headroom-original-path";
var PROJECT_HEADER = "x-headroom-project";
var SESSION_TOKEN_HEADER = "x-headroom-session-token";
var PROXY_ENV = "HEADROOM_OPENCODE_TRANSPORT_PROXY_URL";
var EXCLUDE_HOSTS_ENV = "HEADROOM_OPENCODE_EXCLUDE_HOSTS";
var WIRE_ROOTS_ENV = "HEADROOM_OPENCODE_WIRE_ROOTS";
var STATE_KEY = /* @__PURE__ */ Symbol.for("headroom.opencode.transport");
function getState() {
  return globalThis[STATE_KEY];
}
function setState(state) {
  globalThis[STATE_KEY] = state;
}
function shimImportSpecifier() {
  const shim = new URL("../hook-shim/handler.js", import.meta.url);
  return fs.existsSync(shim) ? shim.href : void 0;
}
function withNodeImportOption(existing, shim) {
  const parts = existing?.trim() ? existing.trim().split(/\s+/) : [];
  const alreadyPresent = parts.some((part, index) => {
    return part === `--import=${shim}` || part === "--import" && parts[index + 1] === shim;
  });
  if (!alreadyPresent) {
    parts.push(`--import=${shim}`);
  }
  return parts.join(" ");
}
function withExcludeHostsEnv(env, excludeHosts) {
  if (excludeHosts.length > 0) {
    env[EXCLUDE_HOSTS_ENV] = excludeHosts.join(",");
  } else {
    delete env[EXCLUDE_HOSTS_ENV];
  }
}
function withWireRootsEnv(env, wireRoots) {
  if (wireRoots.length > 0) {
    env[WIRE_ROOTS_ENV] = wireRoots.map((root) => `${root.origin}${root.path}`).join(",");
  } else {
    delete env[WIRE_ROOTS_ENV];
  }
}
function withShimEnv(env, proxyUrl2, excludeHosts, wireRoots) {
  const nextEnv = { ...env ?? process.env };
  nextEnv[PROXY_ENV] = proxyUrl2;
  withExcludeHostsEnv(nextEnv, excludeHosts);
  withWireRootsEnv(nextEnv, wireRoots);
  const shim = shimImportSpecifier();
  if (shim) {
    nextEnv.NODE_OPTIONS = withNodeImportOption(nextEnv.NODE_OPTIONS, shim);
  }
  return nextEnv;
}
var PROCESS_ENV_KEYS = [PROXY_ENV, EXCLUDE_HOSTS_ENV, WIRE_ROOTS_ENV, "NODE_OPTIONS"];
function snapshotProcessEnv() {
  return Object.fromEntries(PROCESS_ENV_KEYS.map((key) => [key, process.env[key]]));
}
function restoreProcessEnv(saved) {
  for (const key of PROCESS_ENV_KEYS) {
    if (saved[key] === void 0) {
      delete process.env[key];
    } else {
      process.env[key] = saved[key];
    }
  }
}
function installProcessEnv(proxyUrl2, excludeHosts, wireRoots) {
  process.env[PROXY_ENV] = proxyUrl2;
  withExcludeHostsEnv(process.env, excludeHosts);
  withWireRootsEnv(process.env, wireRoots);
  const shim = shimImportSpecifier();
  if (shim) {
    process.env.NODE_OPTIONS = withNodeImportOption(process.env.NODE_OPTIONS, shim);
  }
}
function isOptions(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value) && !(value instanceof URL);
}
function injectOptionsEnv(args, optionIndex, state) {
  const nextArgs = [...args];
  const callback = typeof nextArgs.at(-1) === "function" ? nextArgs.pop() : void 0;
  const existing = isOptions(nextArgs[optionIndex]) ? { ...nextArgs[optionIndex] } : {};
  existing.env = withShimEnv(
    existing.env,
    state.proxyUrl,
    state.excludeHosts,
    state.wireRoots
  );
  if (process.platform === "win32" && existing.windowsHide === void 0) {
    existing.windowsHide = true;
  }
  if (isOptions(nextArgs[optionIndex]) || nextArgs[optionIndex] == null && optionIndex < nextArgs.length) {
    nextArgs[optionIndex] = existing;
  } else {
    nextArgs.splice(optionIndex, 0, existing);
  }
  if (callback) {
    nextArgs.push(callback);
  }
  return nextArgs;
}
function optionsIndexAfterArgs(args) {
  const params = typeof args.at(-1) === "function" ? args.slice(0, -1) : args;
  return Array.isArray(params[1]) || params.length > 2 && params[1] == null ? 2 : 1;
}
function wrapSpawn(originalSpawn) {
  return function headroomSpawn(...args) {
    const state = getState();
    if (!state) {
      return Reflect.apply(originalSpawn, this, args);
    }
    const optionIndex = optionsIndexAfterArgs(args);
    return Reflect.apply(originalSpawn, this, injectOptionsEnv(args, optionIndex, state));
  };
}
function wrapExec(originalExec) {
  return function headroomExec(...args) {
    const state = getState();
    if (!state) {
      return Reflect.apply(originalExec, this, args);
    }
    return Reflect.apply(originalExec, this, injectOptionsEnv(args, 1, state));
  };
}
function wrapExecFile(originalExecFile) {
  return function headroomExecFile(...args) {
    const state = getState();
    if (!state) {
      return Reflect.apply(originalExecFile, this, args);
    }
    const optionIndex = optionsIndexAfterArgs(args);
    return Reflect.apply(originalExecFile, this, injectOptionsEnv(args, optionIndex, state));
  };
}
function wrapFork(originalFork) {
  return function headroomFork(...args) {
    const state = getState();
    if (!state) {
      return Reflect.apply(originalFork, this, args);
    }
    const optionIndex = optionsIndexAfterArgs(args);
    return Reflect.apply(originalFork, this, injectOptionsEnv(args, optionIndex, state));
  };
}
function normalizeProxyUrl(proxyUrl2) {
  return new URL(proxyUrl2);
}
var reportedProxyUrls = /* @__PURE__ */ new Set();
function usableProxyUrl(proxyUrl2) {
  try {
    const proxy = normalizeProxyUrl(proxyUrl2);
    if (proxy.protocol !== "http:" && proxy.protocol !== "https:") throw new TypeError("unsupported scheme");
    return proxy;
  } catch {
    if (!reportedProxyUrls.has(proxyUrl2)) {
      reportedProxyUrls.add(proxyUrl2);
      console.warn(`headroom: the configured proxy URL is ${proxyUrl2 === "" ? "empty" : "not a valid http(s) URL"}; requests are not routed`);
    }
    return void 0;
  }
}
function canonicalHostname(hostname) {
  return hostname.toLowerCase().replace(/^\[|\]$/g, "").replace(/\.+$/, "");
}
function isLoopback(hostname) {
  const normalized = canonicalHostname(hostname);
  if (normalized === "localhost" || normalized === "::1" || normalized === "0.0.0.0") {
    return true;
  }
  if (isLoopbackIpv4(normalized)) {
    return true;
  }
  const mapped = /^::ffff:(.+)$/.exec(normalized)?.[1];
  if (!mapped) {
    return false;
  }
  if (isLoopbackIpv4(mapped)) {
    return true;
  }
  const groups = /^([0-9a-f]{1,4}):([0-9a-f]{1,4})$/.exec(mapped);
  if (!groups) {
    return false;
  }
  const address = parseInt(groups[1], 16) << 16 | parseInt(groups[2], 16);
  return address >>> 24 === 127;
}
function isLoopbackIpv4(host) {
  return /^127\.\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(host);
}
function normalizeExcludeHosts(entries) {
  const hosts = /* @__PURE__ */ new Set();
  for (const entry of typeof entries === "string" ? entries.split(",") : entries) {
    const host = canonicalHostname(String(entry).trim()).replace(/^(?:\*\.|\.)/, "");
    if (host) {
      hosts.add(host);
    }
  }
  return [...hosts];
}
function isExcludedHost(hostname, excludeHosts) {
  const normalized = canonicalHostname(hostname);
  return excludeHosts.some((host) => normalized === host || normalized.endsWith(`.${host}`));
}
function normalizeWireRoots(entries) {
  const roots = /* @__PURE__ */ new Map();
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
    }
  }
  return [...roots.values()];
}
function isConfiguredWireRoot(origin, pathname, wireRoots) {
  return wireRoots.some(
    (root) => root.origin === origin && (root.path === "" ? pathname.startsWith("/") : pathname === root.path || pathname.startsWith(`${root.path}/`))
  );
}
var NAMED_WIRE_ROOTS = /* @__PURE__ */ new Set(["backend-api", "codex", "openai", "anthropic", "zen"]);
var IMMEDIATE_WIRE_ROOTS = /* @__PURE__ */ new Set(["api"]);
function isVersionSegment(segment) {
  return /^v\d+[a-z0-9]*$/i.test(segment);
}
function hasWireRootBefore(body) {
  const segments = body.split("/").filter(Boolean);
  const last = segments.at(-1);
  return last !== void 0 && (isVersionSegment(last) || IMMEDIATE_WIRE_ROOTS.has(last) || segments.some((s) => NAMED_WIRE_ROOTS.has(s)));
}
function isLlmEndpointPath(pathname, origin = "", wireRoots = []) {
  if (pathname.endsWith(":generateContent") || pathname.endsWith(":streamGenerateContent")) {
    return true;
  }
  for (const suffix of ["/chat/completions", "/responses", "/messages"]) {
    if (!pathname.endsWith(suffix)) {
      continue;
    }
    if (isConfiguredWireRoot(origin, pathname, wireRoots)) {
      return true;
    }
    return hasWireRootBefore(pathname.slice(0, -suffix.length));
  }
  return false;
}
function isRoutableUpstream(url, proxy, excludeHosts) {
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
function isLlmMethod(method) {
  return typeof method === "string" && method.toUpperCase() === "POST";
}
function shouldRoute(url, proxy, excludeHosts, method, wireRoots = []) {
  if (!isLlmMethod(method) || !isRoutableUpstream(url, proxy, excludeHosts)) {
    return false;
  }
  return isLlmEndpointPath(url.pathname, url.origin, wireRoots);
}
function routedUrl(upstream, proxy) {
  return new URL(`${upstream.pathname}${upstream.search}`, proxy.origin);
}
function normalizedOpenAiProxyPath(pathname) {
  if (pathname.endsWith("/chat/completions")) {
    return "/v1/chat/completions";
  }
  if (pathname.endsWith("/responses")) {
    return "/v1/responses";
  }
  return void 0;
}
function routedUrlForOpenCode(upstream, proxy) {
  const normalizedPath = normalizedOpenAiProxyPath(upstream.pathname);
  if (!normalizedPath) {
    return {
      url: routedUrl(upstream, proxy),
      originalPath: void 0
    };
  }
  return {
    url: new URL(`${normalizedPath}${upstream.search}`, proxy.origin),
    originalPath: upstream.pathname
  };
}
function requestUrl(input) {
  if (input instanceof Request) {
    return new URL(input.url);
  }
  if (input instanceof URL) {
    return input;
  }
  return new URL(String(input));
}
function mergeFetchHeaders(input, init, upstream, originalPath = void 0, project = void 0, sessionToken = void 0) {
  const headers = new Headers(input instanceof Request ? input.headers : void 0);
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
function withRoutedFetchInput(input, init, proxy, project, excludeHosts, sessionToken, wireRoots = []) {
  const upstream = requestUrl(input);
  const method = init?.method ?? (input instanceof Request ? input.method : "GET");
  if (!shouldRoute(upstream, proxy, excludeHosts, method, wireRoots)) {
    return [input, init];
  }
  const { url: nextUrl, originalPath } = routedUrlForOpenCode(upstream, proxy);
  const nextInit = {
    ...init,
    headers: mergeFetchHeaders(input, init, upstream, originalPath, project, sessionToken)
  };
  if (input instanceof Request) {
    return [new Request(nextUrl, input), nextInit];
  }
  return [nextUrl, nextInit];
}
function splitNodeArgs(args, defaultProtocol) {
  const callback = typeof args.at(-1) === "function" ? args.at(-1) : void 0;
  const withoutCallback = callback ? args.slice(0, -1) : args;
  const [first, second] = withoutCallback;
  const options = typeof second === "object" && second !== null ? { ...second } : {};
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
    const requestOptions = { ...first, ...options };
    return { url: urlFromRequestOptions(requestOptions, defaultProtocol), options: requestOptions, callback };
  }
  return { options, callback };
}
function urlFromRequestOptions(options, defaultProtocol) {
  const protocol = String(options.protocol ?? defaultProtocol);
  if (protocol !== "http:" && protocol !== "https:") {
    return void 0;
  }
  const hostValue = options.hostname ?? options.host;
  if (!hostValue) {
    return void 0;
  }
  const hostname = String(hostValue).replace(/:\d+$/, "");
  const port = options.port ? `:${String(options.port)}` : "";
  const path = String(options.path ?? "/");
  try {
    return new URL(`${protocol}//${hostname}${port}${path}`);
  } catch {
    return void 0;
  }
}
function headersForNodeRequest(options, upstream, originalPath, project, sessionToken) {
  const headers = new Headers(options.headers);
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
  const result = {};
  headers.forEach((value, key) => {
    result[key] = value;
  });
  return result;
}
function routedNodeOptions(parts, proxy, project, excludeHosts, sessionToken, wireRoots = []) {
  if (!parts.url || !shouldRoute(parts.url, proxy, excludeHosts, parts.options.method ?? "GET", wireRoots)) {
    return void 0;
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
    port: nextUrl.port || void 0,
    path: `${nextUrl.pathname}${nextUrl.search}`,
    headers: headersForNodeRequest(parts.options, parts.url, originalPath, project, sessionToken)
  };
}
function wrapRequest(originalHttpRequest, originalHttpsRequest, originalRequest) {
  return function headroomRequest(...args) {
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
      state.wireRoots
    );
    if (!nextOptions) {
      return Reflect.apply(originalRequest, this, args);
    }
    const targetRequest = proxy.protocol === "https:" ? originalHttpsRequest : originalHttpRequest;
    const nextArgs = parts.callback ? [nextOptions, parts.callback] : [nextOptions];
    return Reflect.apply(targetRequest, this, nextArgs);
  };
}
function wrapGet(request) {
  return function headroomGet(...args) {
    const req = Reflect.apply(request, this, args);
    req.end();
    return req;
  };
}
function wrapHttp2Connect(originalConnect) {
  return function headroomHttp2Connect(...args) {
    return Reflect.apply(originalConnect, this, args);
  };
}
function installHeadroomTransport(options) {
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
  const state = {
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
    originalEnv: snapshotProcessEnv()
  };
  setState(state);
  installProcessEnv(options.proxyUrl, excludeHosts, wireRoots);
  globalThis.fetch = async (...args) => {
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
      current.wireRoots
    );
    return state.originalFetch(nextInput, nextInit);
  };
  http.request = wrapRequest(state.originalHttpRequest, state.originalHttpsRequest, state.originalHttpRequest);
  https.request = wrapRequest(state.originalHttpRequest, state.originalHttpsRequest, state.originalHttpsRequest);
  http.get = wrapGet(http.request);
  https.get = wrapGet(https.request);
  http2.connect = wrapHttp2Connect(state.originalHttp2Connect);
  childProcess.spawn = wrapSpawn(state.originalChildSpawn);
  childProcess.exec = wrapExec(state.originalChildExec);
  childProcess.execFile = wrapExecFile(state.originalChildExecFile);
  childProcess.fork = wrapFork(state.originalChildFork);
  syncBuiltinESMExports();
  return () => uninstallHeadroomTransport();
}
function uninstallHeadroomTransport() {
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
  setState(void 0);
}

// src/hook-shim.ts
var proxyUrl = process.env.HEADROOM_OPENCODE_TRANSPORT_PROXY_URL;
if (!proxyUrl) {
  throw new Error(
    "Headroom OpenCode transport shim loaded without HEADROOM_OPENCODE_TRANSPORT_PROXY_URL"
  );
}
installHeadroomTransport({ proxyUrl });
