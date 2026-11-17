import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

const sdk = path.resolve(process.argv[2]);
const require = createRequire(path.join(sdk, 'package.json'));
const Anthropic = require('@anthropic-ai/sdk').default;
const { withHeadroom } = await import(pathToFileURL(path.join(sdk, 'dist/adapters/anthropic.js')));
assert.ok(['localhost', '127.0.0.1'].includes(new URL(process.env.HEADROOM_BASE_URL).hostname));
const calls = [];
const server = createServer(async (request, response) => {
  const chunks = [];
  for await (const chunk of request) chunks.push(chunk);
  calls.push({ body: JSON.parse(Buffer.concat(chunks).toString()), path: request.url });
  response.writeHead(200, { 'content-type': 'application/json' });
  response.end(JSON.stringify({
    id: 'msg-local-peer-proof', type: 'message', role: 'assistant',
    model: 'claude-sonnet-4-5-20250929',
    content: [{ type: 'text', text: 'fixture answer' }],
    stop_reason: 'end_turn', stop_sequence: null,
    usage: { input_tokens: 100, output_tokens: 8 },
  }));
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
try {
  const messages = [
    { role: 'user', content: [{ type: 'text', text: 'Read the fixture.', cache_control: { type: 'ephemeral' } }] },
    { role: 'assistant', content: [
      { type: 'thinking', thinking: 'Opaque fixture reasoning.', signature: 'opaque-fixture-signature' },
      { type: 'tool_use', id: 'tool-local-1', name: 'lookup', input: { query: 'fixture' } },
    ] },
    { role: 'user', content: [{
      type: 'tool_result', tool_use_id: 'tool-local-1', content: 'short fixture result',
      cache_control: { type: 'ephemeral' },
    }] },
  ];
  const system = [{ type: 'text', text: 'Return the fixture answer.', cache_control: { type: 'ephemeral' } }];
  const metadata = { user_id: 'local-fixture' };
  const client = withHeadroom(new Anthropic({
    apiKey: 'local-fixture-only',
    baseURL: `http://127.0.0.1:${server.address().port}`,
  }), { baseUrl: process.env.HEADROOM_BASE_URL, fallback: false, retries: 0 });
  const result = await client.messages.create({
    model: 'claude-sonnet-4-5-20250929', max_tokens: 2048,
    thinking: { type: 'enabled', budget_tokens: 1024 }, messages, system, metadata,
    tools: [{ name: 'lookup', input_schema: {
      type: 'object', properties: { query: { type: 'string' } }, required: ['query'],
    } }],
  });
  assert.equal(calls.length, 1);
  assert.equal(calls[0].path, '/v1/messages');
  assert.deepEqual(calls[0].body.messages, messages);
  assert.deepEqual(calls[0].body.system, system);
  assert.deepEqual(calls[0].body.metadata, metadata);
  assert.equal(result.content[0].text, 'fixture answer');
  console.log('PASS: actual Anthropic SDK client + Headroom adapter preserve thinking, cache markers, tool pairs, system and metadata on the wire');
} finally {
  server.closeAllConnections();
  await new Promise(resolve => server.close(resolve));
}
