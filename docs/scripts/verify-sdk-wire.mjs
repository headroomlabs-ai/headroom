/** Credentialless HTTP qualification for the SDK versions locked by docs/package-lock.json.
 * Run after `cd docs && npm ci`: `node scripts/verify-sdk-wire.mjs`.
 * Exercises SDK wire/API compatibility only, not Headroom adapters or a live provider.
 */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import OpenAI from 'openai';
import { createOpenAI } from '@ai-sdk/openai';
import { generateText } from 'ai';

/** @type {{path: string | undefined, body: Record<string, any>, authorization: string | undefined}[]} */
const calls = [];
const server = createServer(async (request, response) => {
  const chunks = [];
  for await (const chunk of request) chunks.push(chunk);
  const body = JSON.parse(Buffer.concat(chunks).toString());
  calls.push({ path: request.url, body, authorization: request.headers.authorization });
  response.writeHead(200, { 'content-type': 'application/json' });
  if (request.url === '/v1/chat/completions') {
    response.end(JSON.stringify({
      id: 'chatcmpl-local', object: 'chat.completion', created: 1, model: 'gpt-4o',
      choices: [{ index: 0, message: { role: 'assistant', content: 'local chat answer' }, finish_reason: 'stop' }],
      usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
    }));
  } else {
    response.end(JSON.stringify({
      id: 'resp_local', object: 'response', created_at: 1, status: 'completed', model: 'gpt-4o',
      output: [{ id: 'msg_local', type: 'message', role: 'assistant', status: 'completed',
        content: [{ type: 'output_text', text: 'local responses answer', annotations: [] }] }],
      usage: { input_tokens: 5, output_tokens: 3, total_tokens: 8,
        input_tokens_details: { cached_tokens: 0 }, output_tokens_details: { reasoning_tokens: 0 } },
    }));
  }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', () => resolve(undefined)));
try {
  const address = server.address();
  assert.ok(address && typeof address === 'object');
  const baseURL = `http://127.0.0.1:${address.port}/v1`;
  const native = new OpenAI({ apiKey: 'local-native-fixture', baseURL, maxRetries: 0 });
  const completion = await native.chat.completions.create({
    model: 'gpt-4o', messages: [{ role: 'user', content: 'Say hello locally.' }],
  });
  assert.equal(completion.choices[0].message.content, 'local chat answer');
  assert.equal(calls[0].path, '/v1/chat/completions');
  assert.equal(calls[0].authorization, 'Bearer local-native-fixture');
  assert.deepEqual(calls[0].body.messages, [{ role: 'user', content: 'Say hello locally.' }]);

  const provider = createOpenAI({ apiKey: 'local-ai-fixture', baseURL });
  // The documented provider('gpt-4o') pattern selects the Responses API.
  const generated = await generateText({
    model: provider('gpt-4o'), prompt: 'Say hello locally.', maxRetries: 0,
  });
  assert.equal(generated.text, 'local responses answer');
  assert.equal(calls[1].path, '/v1/responses');
  assert.equal(calls[1].authorization, 'Bearer local-ai-fixture');
  assert.equal(calls[1].body.model, 'gpt-4o');
  assert.equal(calls[1].body.input[0].content[0].text, 'Say hello locally.');
  assert.equal(generated.usage.totalTokens, 8);
  assert.equal(calls.length, 2);
  console.log('PASS: native OpenAI chat request and ai.generateText + OpenAI Responses provider over loopback HTTP; no credentials or external provider required.');
} finally {
  server.closeAllConnections();
  await new Promise(resolve => server.close(() => resolve(undefined)));
}
