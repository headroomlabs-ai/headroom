// From the repository root: npm --prefix docs ci
// Then: node scripts/verify_docs_anthropic_sdk.mjs
// Exercises the development SDK directly against an owned loopback server.
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { createRequire } from 'node:module';

const require = createRequire(new URL('../docs/package.json', import.meta.url));
const { Anthropic } = require('@anthropic-ai/sdk');
const requests = [];
const server = createServer(async (request, response) => {
  const chunks = [];
  for await (const chunk of request) chunks.push(chunk);
  requests.push({ path: request.url, headers: request.headers, body: JSON.parse(Buffer.concat(chunks).toString('utf8')) });
  response.writeHead(200, { 'content-type': 'application/json' });
  response.end(JSON.stringify({
    id: 'msg_loopback', type: 'message', role: 'assistant', model: 'claude-sonnet-4-6',
    content: [{ type: 'thinking', thinking: 'Local SDK compatibility check.', signature: 'loopback-signature' }, { type: 'text', text: 'loopback-ok' }],
    stop_reason: 'end_turn', stop_sequence: null,
    usage: { input_tokens: 10, output_tokens: 4 },
  }));
});
await new Promise((resolve, reject) => {
  server.once('error', reject);
  server.listen(0, '127.0.0.1', resolve);
});
try {
  const client = new Anthropic({ apiKey: 'loopback-only', baseURL: `http://127.0.0.1:${server.address().port}`, maxRetries: 0, timeout: 5000 });
  const input = {
    model: 'claude-sonnet-4-6', max_tokens: 4096,
    thinking: { type: 'enabled', budget_tokens: 1024 },
    system: [{ type: 'text', text: 'Local SDK check.', cache_control: { type: 'ephemeral' } }],
    tools: [{ name: 'lookup', description: 'Local fixture.', input_schema: { type: 'object', properties: { key: { type: 'string' } }, required: ['key'] }, cache_control: { type: 'ephemeral' } }],
    messages: [{ role: 'user', content: [{ type: 'text', text: 'Test the local fixture.' }] }],
    metadata: { user_id: 'docs-compatibility-check' },
  };
  const result = await client.messages.create(input);
  assert.equal(requests.length, 1);
  assert.equal(requests[0].path, '/v1/messages');
  assert.equal(requests[0].headers['anthropic-version'], '2026-10-21');
  assert.equal(requests[0].headers.authorization, 'Bearer loopback-only');
  assert.deepEqual(requests[0].body, input);
  assert.equal(result.content[0].type, 'thinking');
  assert.equal(result.content[1].text, 'loopback-ok');
  console.log(JSON.stringify({ passed: true, requests: 1, thinking: true, cachedSystemAndTool: true, metadata: true }));
} finally {
  await new Promise((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
}
