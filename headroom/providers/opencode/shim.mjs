/**
 * headroom/providers/opencode/shim.mjs
 *
 * Loaded via NODE_OPTIONS=--import=<path> before any userland code runs.
 * Patches globalThis.fetch so that external AI-provider calls are routed
 * through the local Headroom proxy. The original upstream origin is passed
 * via the x-headroom-base-url header; proxy_routes.py already handles it.
 *
 * Design notes (why we only patch fetch and nothing else):
 *   - The Vercel AI SDK — which OpenCode uses — talks to providers via the
 *     Web Fetch API exclusively. It does NOT use Node's http/https modules.
 *   - Patching http, https, http2, or child_process (as some alternatives do)
 *     breaks unrelated tools (git, npm) and adds fragile surface area.
 *   - Child processes inherit NODE_OPTIONS automatically via the shell, so
 *     subagents already pick up this shim without any child_process patching.
 *
 * Cold-start retry: if the local proxy returns 502/503 or refuses the
 * connection (proxy still warming up), we retry up to 3 times with 75ms
 * delays before surfacing the error — matching OpenChamber's hold+poll
 * pattern so the first request never fails during proxy startup.
 */

const PROXY_URL = process.env.HEADROOM_PROXY_URL;

if (PROXY_URL) {
  const proxyOrigin = new URL(PROXY_URL).origin;
  const _fetch = globalThis.fetch;

  function isLoopback(hostname) {
    return (
      hostname === "localhost" ||
      hostname === "::1" ||
      hostname.startsWith("127.") ||
      hostname.endsWith(".local")
    );
  }

  function shouldRoute(url) {
    if (url.protocol !== "https:" && url.protocol !== "http:") return false;
    if (isLoopback(url.hostname)) return false;
    if (url.origin === proxyOrigin) return false; // avoid proxy→proxy loops
    return true;
  }

  async function fetchWithRetry(url, init, attempts = 3) {
    for (let i = 0; i < attempts; i++) {
      try {
        const res = await _fetch(url, init);
        // 502/503 = proxy not ready yet; retry unless this is the last attempt
        if (res.status < 502 || i === attempts - 1) return res;
      } catch (err) {
        if (i === attempts - 1) throw err;
      }
      await new Promise((r) => setTimeout(r, 75));
    }
  }

  globalThis.fetch = function headroomFetch(input, init = {}) {
    try {
      const raw =
        typeof input === "string"
          ? input
          : input instanceof URL
            ? input.href
            : input instanceof Request
              ? input.url
              : String(input);

      const url = new URL(raw);
      if (!shouldRoute(url)) return _fetch(input, init);

      // Rewrite destination to proxy, carry original origin as routing header
      const proxied = new URL(url.pathname + url.search, proxyOrigin);
      const headers = new Headers(
        init?.headers ?? (input instanceof Request ? input.headers : {}),
      );
      headers.set("x-headroom-base-url", url.origin);

      return fetchWithRetry(proxied.toString(), { ...init, headers });
    } catch {
      // Fail-open: if URL parsing fails (e.g. non-URL string), pass through
      return _fetch(input, init);
    }
  };
}
