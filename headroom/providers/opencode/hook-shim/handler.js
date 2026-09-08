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
var PROXY_ENV = "HEADROOM_OPENCODE_TRANSPORT_PROXY_URL";
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
function withShimEnv(env, proxyUrl2) {
  const nextEnv = { ...env ?? process.env };
  nextEnv[PROXY_ENV] = proxyUrl2;
  const shim = shimImportSpecifier();
  if (shim) {
    nextEnv.NODE_OPTIONS = withNodeImportOption(nextEnv.NODE_OPTIONS, shim);
  }
  return nextEnv;
}
function installProcessEnv(proxyUrl2) {
  process.env[PROXY_ENV] = proxyUrl2;
  const shim = shimImportSpecifier();
  if (shim) {
    process.env.NODE_OPTIONS = withNodeImportOption(process.env.NODE_OPTIONS, shim);
  }
}
function isOptions(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value) && !(value instanceof URL);
}
function injectOptionsEnv(args, optionIndex, proxyUrl2) {
  const nextArgs = [...args];
  const callback = typeof nextArgs.at(-1) === "function" ? nextArgs.pop() : void 0;
  const existing = isOptions(nextArgs[optionIndex]) ? { ...nextArgs[optionIndex] } : {};
  existing.env = withShimEnv(existing.env, proxyUrl2);
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
function wrapSpawn(originalSpawn) {
  return function headroomSpawn(...args) {
    const state = getState();
    if (!state) {
      return Reflect.apply(originalSpawn, this, args);
    }
    const optionIndex = Array.isArray(args[1]) ? 2 : 1;
    return Reflect.apply(originalSpawn, this, injectOptionsEnv(args, optionIndex, state.proxyUrl));
  };
}
function wrapExec(originalExec) {
  return function headroomExec(...args) {
    const state = getState();
    if (!state) {
      return Reflect.apply(originalExec, this, args);
    }
    return Reflect.apply(originalExec, this, injectOptionsEnv(args, 1, state.proxyUrl));
  };
}
function wrapExecFile(originalExecFile) {
  return function headroomExecFile(...args) {
    const state = getState();
    if (!state) {
      return Reflect.apply(originalExecFile, this, args);
    }
    const optionIndex = Array.isArray(args[1]) ? 2 : 1;
    return Reflect.apply(originalExecFile, this, injectOptionsEnv(args, optionIndex, state.proxyUrl));
  };
}
function wrapFork(originalFork) {
  return function headroomFork(...args) {
    const state = getState();
    if (!state) {
      return Reflect.apply(originalFork, this, args);
    }
    const optionIndex = Array.isArray(args[1]) ? 2 : 1;
    return Reflect.apply(originalFork, this, injectOptionsEnv(args, optionIndex, state.proxyUrl));
  };
}
function normalizeProxyUrl(proxyUrl2) {
  return new URL(proxyUrl2);
}
function isLoopback(hostname) {
  const normalized = hostname.toLowerCase().replace(/^\[|\]$/g, "");
  return normalized === "localhost" || normalized === "127.0.0.1" || normalized === "::1";
}
function shouldRoute(url, proxy) {
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    return false;
  }
  if (isLoopback(url.hostname)) {
    return false;
  }
  if (url.origin === proxy.origin) {
    return false;
  }
  return true;
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
function mergeFetchHeaders(input, init, upstream, originalPath = void 0, project = void 0) {
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
  return headers;
}
function withRoutedFetchInput(input, init, proxy, project) {
  const upstream = requestUrl(input);
  if (!shouldRoute(upstream, proxy)) {
    return [input, init];
  }
  const { url: nextUrl, originalPath } = routedUrlForOpenCode(upstream, proxy);
  const nextInit = {
    ...init,
    headers: mergeFetchHeaders(input, init, upstream, originalPath, project)
  };
  if (input instanceof Request) {
    return [new Request(nextUrl, input), nextInit];
  }
  return [nextUrl, nextInit];
}
function splitNodeArgs(args) {
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
    return { url: urlFromRequestOptions(requestOptions), options: requestOptions, callback };
  }
  return { options, callback };
}
function urlFromRequestOptions(options) {
  const protocol = String(options.protocol ?? "http:");
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
function headersForNodeRequest(options, upstream, originalPath, project) {
  const headers = new Headers(options.headers);
  headers.set(BASE_URL_HEADER, upstream.origin);
  if (originalPath) {
    headers.set(ORIGINAL_PATH_HEADER, originalPath);
  }
  if (project) {
    headers.set(PROJECT_HEADER, project);
  }
  headers.delete("host");
  const result = {};
  headers.forEach((value, key) => {
    result[key] = value;
  });
  return result;
}
function routedNodeOptions(parts, proxy, project) {
  if (!parts.url || !shouldRoute(parts.url, proxy)) {
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
    headers: headersForNodeRequest(parts.options, parts.url, originalPath, project)
  };
}
function wrapRequest(originalHttpRequest, originalHttpsRequest, originalRequest) {
  return function headroomRequest(...args) {
    const state = getState();
    if (!state) {
      return Reflect.apply(originalRequest, this, args);
    }
    const proxy = normalizeProxyUrl(state.proxyUrl);
    const parts = splitNodeArgs(args);
    const nextOptions = routedNodeOptions(parts, proxy, state.project);
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
  return function headroomHttp2Connect(authority, ...args) {
    const state = getState();
    if (state) {
      const proxy = normalizeProxyUrl(state.proxyUrl);
      const upstream = authority instanceof URL ? authority : new URL(String(authority));
      if (shouldRoute(upstream, proxy)) {
        throw new Error(
          `Headroom OpenCode wrap blocked direct HTTP/2 connection to ${upstream.origin}. Use fetch, http, or https so traffic can be routed through Headroom.`
        );
      }
    }
    return Reflect.apply(originalConnect, this, [authority, ...args]);
  };
}
function installHeadroomTransport(options) {
  const existing = getState();
  if (existing) {
    existing.refs += 1;
    existing.proxyUrl = options.proxyUrl;
    existing.project = options.project;
    existing.debug = Boolean(options.debug);
    installProcessEnv(options.proxyUrl);
    return () => uninstallHeadroomTransport();
  }
  const state = {
    refs: 1,
    proxyUrl: options.proxyUrl,
    project: options.project,
    debug: Boolean(options.debug),
    originalFetch: globalThis.fetch,
    originalHttpRequest: http.request,
    originalHttpGet: http.get,
    originalHttpsRequest: https.request,
    originalHttpsGet: https.get,
    originalHttp2Connect: http2.connect,
    originalChildSpawn: childProcess.spawn,
    originalChildExec: childProcess.exec,
    originalChildExecFile: childProcess.execFile,
    originalChildFork: childProcess.fork
  };
  setState(state);
  installProcessEnv(options.proxyUrl);
  globalThis.fetch = async (...args) => {
    const current = getState();
    if (!current) {
      return state.originalFetch(...args);
    }
    const proxy = normalizeProxyUrl(current.proxyUrl);
    const [nextInput, nextInit] = withRoutedFetchInput(args[0], args[1], proxy, current.project);
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
