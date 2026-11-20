// Shared-loopback HTTP conformance for the generated TypeScript client.
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
const { Client } = await import(pathToFileURL(process.argv[2] + "/client.js"));
const { APIError, ProtocolError, validate } = await import(pathToFileURL(process.argv[2] + "/runtime.js"));
const client = new Client(process.env.HEADROOM_WIRE_TEST_URL);
let count = 0;
async function test(name, fn) { await fn(); count++; console.log(`PASS TS ${name}`); }
await test("POST, nullability, Unicode, opaque content and future fields", async () => {
  const value = await client.retrieve({hash: "ok", extra_request: {snake_case: false}});
  assert.equal(value.tool_name, null);
  assert.equal(value.original_content, '{"snake_case":"世界"}');
  assert.deepEqual(value.future_extension, {snake_case: "unchanged"});
});
await test("GET encodes the entire path segment and preserves base path", async () => {
  const hash = "a/世界 ?#+'!*()";
  assert.equal((await client.retrieveGet(hash)).hash, hash);
});
await test("HTTP errors retain status, raw bytes, and headers", async () => {
  await assert.rejects(client.retrieve({hash: "missing"}), e => {
    assert(e instanceof APIError); assert.equal(e.status, 404);
    assert.equal(e.headers.get("x-fixture"), "true");
    assert.match(new TextDecoder().decode(e.body), /Entry missing/);
    assert(!e.message.includes("Entry missing")); return true;
  });
});
await test("missing required-nullable field rejects", async () => {
  await assert.rejects(client.retrieve({hash: "missing_field"}), ProtocolError);
});
await test("wire type mismatch rejects", async () => {
  await assert.rejects(client.retrieve({hash: "wrong_type"}), ProtocolError);
});
await test("unsafe integer rejects instead of silent rounding", async () => {
  await assert.rejects(client.retrieve({hash: "unsafe_int"}), ProtocolError);
});
await test("non-JSON success rejects", async () => {
  await assert.rejects(client.retrieve({hash: "notjson"}), ProtocolError);
});
await test("valid JSON with a non-JSON media type rejects", async () => {
  await assert.rejects(client.retrieve({hash: "valid_nonjson"}), ProtocolError);
});
await test("null in a non-nullable response field rejects", async () => {
  await assert.rejects(client.retrieve({hash: "null_nonnullable"}), ProtocolError);
});
await test("redirects are not followed", async () => {
  await assert.rejects(client.retrieve({hash: "redirect"}), e => e instanceof APIError && e.status === 302);
});
await test("bounded response reads", async () => {
  const small = new Client(process.env.HEADROOM_WIRE_TEST_URL, {maxResponseBytes: 16});
  await assert.rejects(small.retrieve({hash: "ok"}), ProtocolError);
});
await test("caller cancellation is honored", async () => {
  const controller = new AbortController(); controller.abort();
  await assert.rejects(client.retrieve({hash: "ok"}, {signal: controller.signal}));
});
await test("timeout and in-flight cancellation are honored", async () => {
  const timed = new Client(process.env.HEADROOM_WIRE_TEST_URL, {timeoutMs: 50});
  await assert.rejects(timed.retrieve({hash: "slow"}));
  const controller = new AbortController();
  setTimeout(() => controller.abort(), 25);
  await assert.rejects(client.retrieve({hash: "slow"}, {signal: controller.signal}));
});
await test("closed connection is not retried", async () => {
  await assert.rejects(client.retrieve({hash: "retry_probe_typescript"}));
});
await test("optional undefined, null, zero and false retain distinctions", async () => {
  const schema = {type: "object", properties: {note: {anyOf: [{type: "string"}, {type: "null"}]}, enabled: {type: "boolean"}, count: {type: "integer"}}, required: []};
  for (const value of [{}, {note: undefined}, {note: null}, {note: ""}, {enabled: false, count: 0}]) validate(value, schema, {});
  assert.throws(() => validate({enabled: null}, schema, {}), ProtocolError);
});
await test("invalid base URLs and dot segments are rejected", async () => {
  for (const base of ["file:///tmp", "http://user:pass@localhost", "http://localhost/?secret=1"]) assert.throws(() => new Client(base));
  await assert.rejects(client.retrieveGet(".."));
});
console.log(`TypeScript: ${count} conformance tests passed`);
