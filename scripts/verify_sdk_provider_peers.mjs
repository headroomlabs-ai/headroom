import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

const sdk = path.resolve(process.argv[2]);
const require = createRequire(path.join(sdk, 'package.json'));
const { createAnthropic } = require('@ai-sdk/anthropic');
const { createOpenAI } = require('@ai-sdk/openai');
const OpenAI = require('openai').default;
const { withHeadroom } = await import(pathToFileURL(path.join(sdk, 'dist/adapters/vercel-ai.js')));
const { withHeadroom: withOpenAIHeadroom } = await import(pathToFileURL(path.join(sdk, 'dist/adapters/openai.js')));
assert.ok(['localhost', '127.0.0.1'].includes(new URL(process.env.HEADROOM_BASE_URL).hostname));
const calls = [];
const server = createServer(async (request, response) => {
  const chunks = [];
  for await (const chunk of request) chunks.push(chunk);
  const body = JSON.parse(Buffer.concat(chunks).toString());
  calls.push({ body, url: request.url, authorization: request.headers.authorization });
  response.writeHead(200, { 'content-type': 'application/json' });
  if (request.url.endsWith('/chat/completions')) {
    response.end(JSON.stringify({
      id: 'chatcmpl-local', object: 'chat.completion', created: 1, model: 'gpt-4o',
      choices: [{ index: 0, message: { role: 'assistant', content: 'fixture answer' }, finish_reason: 'stop' }],
      usage: { prompt_tokens: 12, completion_tokens: 2, total_tokens: 14 },
    }));
    return;
  }
  response.end(JSON.stringify({
    id: 'msg-local-structured-proof', type: 'message', role: 'assistant',
    model: 'claude-3-7-sonnet-20250219',
    content: [{ type: 'text', text: '{"name":"Ada"}' }],
    stop_reason: 'end_turn', stop_sequence: null,
    usage: { input_tokens: 12, output_tokens: 8 },
  }));
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
try {
  if (process.argv[3] !== 'openai-only') {
  const provider = createAnthropic({
    apiKey: 'local-fixture-only',
    baseURL: `http://127.0.0.1:${server.address().port}/v1`,
  });
  const model = withHeadroom(provider('claude-3-7-sonnet-20250219'), {
    baseUrl: process.env.HEADROOM_BASE_URL, fallback: false, retries: 0,
    model: 'claude-3-7-sonnet-20250219',
  });
  const result = await model.doGenerate({
    prompt: [{ role: 'user', content: [{ type: 'text', text: 'Return the name Ada as JSON.' }] }],
    maxOutputTokens: 2048,
    responseFormat: { type: 'json', schema: {
      type: 'object', properties: { name: { type: 'string' } },
      required: ['name'], additionalProperties: false,
    } },
    providerOptions: { anthropic: { thinking: { type: 'enabled', budgetTokens: 1024 } } },
  });
  assert.equal(calls.length, 1);
  assert.equal(calls[0].body.thinking.type, 'enabled');
  assert.ok(calls[0].body.tools?.some(tool => tool.name === 'json'));
  const text = result.content.filter(part => part.type === 'text').map(part => part.text).join('');
  assert.equal(text, '{"name":"Ada"}');
  console.log('PASS: real provider + Headroom middleware + real compression HTTP path preserve final JSON text without a tool call');
  }
  const beforeOpenAI = calls.length;
  let credentialCalls = 0;
  const openai = createOpenAI({
    apiKey: async () => `local-fixture-${++credentialCalls}`,
    baseURL: `http://127.0.0.1:${server.address().port}/v1`,
  });
  const wrapped = withHeadroom(openai.chat('gpt-4o'), {
    baseUrl: process.env.HEADROOM_BASE_URL, fallback: false, retries: 0,
  });
  for (let index = 1; index <= 2; index++) {
    const result = await wrapped.doGenerate({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'Say hello.' }] }],
      maxOutputTokens: 128,
    });
    assert.equal(result.content.find(part => part.type === 'text').text, 'fixture answer');
    assert.equal(calls[beforeOpenAI + index - 1].authorization, `Bearer local-fixture-${index}`);
  }
  assert.equal(credentialCalls, 2);
  console.log('PASS: real OpenAI provider resolves async credentials separately for both wrapped requests');
  const native = withOpenAIHeadroom(new OpenAI({
    apiKey: 'local-native-fixture', baseURL: `http://127.0.0.1:${server.address().port}/v1`,
  }), { baseUrl: process.env.HEADROOM_BASE_URL, fallback: false, retries: 0 });
  const completion = await native.chat.completions.create({
    model: 'gpt-4o', messages: [{ role: 'user', content: 'Say hello.' }],
  });
  assert.equal(completion.choices[0].message.content, 'fixture answer');
  assert.equal(calls.length, beforeOpenAI + 3);
  console.log('PASS: native OpenAI SDK chat client completes a real HTTP request through the Headroom adapter');
} finally {
  server.closeAllConnections();
  await new Promise(resolve => server.close(resolve));
}
