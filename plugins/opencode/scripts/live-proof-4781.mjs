// Live proof for #4781: real Node process, real sockets, real built plugin.
// Nothing in the process under test is stubbed - globalThis.fetch is undici's.
// The upstream API and the Headroom proxy are local HTTP fixtures, because the
// issue's repro touches no provider account (a MessageBird request carries only
// a fake AccessKey). Both fixtures record what actually arrives.
import http from "node:http";
import { fileURLToPath, pathToFileURL } from "node:url";

// Resolve the committed wheel bundle relative to this script, not the cwd.
const BUNDLE = fileURLToPath(new URL("../../../headroom/providers/opencode/_dist/index.js", import.meta.url));

const UPSTREAM_PORT = 8899;
const PROXY_PORT = 8790;
// localtest.me resolves to 127.0.0.1 but is not a loopback spelling, so the
// transport treats it as a routable upstream and the request really leaves the
// process over TCP.
const HOST = "localtest.me";

const seen = { upstream: [], proxy: [], proxyRaw: [] };

function record(bucket, req, payload) {
  bucket.push({
    method: req.method,
    url: req.url,
    host: req.headers.host,
    auth: req.headers.authorization,
    baseUrl: payload.baseUrl,
    originalPath: payload.originalPath,
    body: payload.body,
  });
}

async function listen(name, port, bucket) {
  const server = http.createServer((req, res) => {
    let body = "";
    req.setEncoding("utf8");
    req.on("data", (chunk) => (body += chunk));
    req.on("end", () => {
      record(bucket, req, { body, baseUrl: req.headers["x-headroom-base-url"], originalPath: req.headers["x-headroom-original-path"] });
      if (name === "proxy") seen.proxyRaw.push({ url: req.url, headers: req.headers });
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ servedBy: name, path: req.url }));
    });
  });
  await new Promise((resolve) => server.listen(port, "127.0.0.1", resolve));
  return server;
}

const upstream = await listen("upstream", UPSTREAM_PORT, seen.upstream);
const proxy = await listen("proxy", PROXY_PORT, seen.proxy);

const mod = await import(pathToFileURL(BUNDLE).href);
const plugin = mod.default ?? mod;
const hooks = await plugin.server(
  { directory: process.cwd(), project: { id: "proof-4781" } },
  { proxyUrl: `http://127.0.0.1:${PROXY_PORT}`, debug: true },
);

const post = (path) =>
  fetch(`http://${HOST}:${UPSTREAM_PORT}${path}`, {
    method: "POST",
    headers: { authorization: "AccessKey FAKE-TEST-VALUE", "content-type": "application/json" },
    body: JSON.stringify({ originator: "+390000000000", recipients: ["+391111111111"], body: "hi" }),
  });

const cases = [
  ["MessageBird-shaped  POST /messages", "/messages", "upstream"],
  ["MessageBird batches  POST /messages/batches", "/messages/batches", "upstream"],
  ["LLM control         POST /v1/messages", "/v1/messages", "proxy"],
  ["LLM control         POST /v1/chat/completions", "/v1/chat/completions", "proxy"],
];

const table = [];
for (const [label, path, expected] of cases) {
  const before = { upstream: seen.upstream.length, proxy: seen.proxy.length };
  const response = await post(path);
  const payload = await response.json();
  const landed = seen.proxy.length > before.proxy ? "proxy" : "upstream";
  table.push({
    label,
    path,
    "served by": payload.servedBy,
    verdict: landed === "proxy" ? "REROUTED to proxy" : "left alone",
    expected,
    pass: landed === expected,
  });
}

console.log("\n=== decisions (real sockets, real fetch, real built plugin) ===");
for (const row of table) console.log(`${row.pass ? "PASS" : "FAIL"}  ${row.label.padEnd(46)} -> ${row.verdict}`);

const routed = seen.proxy.filter((r) => r.url.includes("/messages") || r.url.includes("/chat/completions"));
const smsAtProxy = seen.proxy.filter((r) => r.url === "/messages" && r.baseUrl === `http://${HOST}:${UPSTREAM_PORT}`);

console.log("\n=== what reached the proxy ===");
for (const r of routed) console.log(`  ${r.method} ${r.url}  x-headroom-base-url=${r.baseUrl ?? "null"}  original-path=${r.originalPath ?? "null"}`);
console.log("\n=== full headers seen by the proxy ===");
for (const r of seen.proxyRaw) console.log(`  ${r.url} :: ${JSON.stringify(r.headers)}`);

console.log("\n=== MessageBird-shaped requests that reached the proxy: " + smsAtProxy.length + " ===");
console.log("=== what reached the upstream directly ===");
for (const r of seen.upstream) console.log(`  ${r.method} ${r.url}  auth=${r.auth}  (no proxy headers: ${r.baseUrl === undefined})`);

await hooks.dispose(async () => {});
upstream.close();
proxy.close();

const failed = table.filter((row) => !row.pass);
console.log(`\n${failed.length === 0 ? "ALL CASES PASS" : failed.length + " CASE(S) FAILED"}`);
process.exit(failed.length === 0 ? 0 : 1);
