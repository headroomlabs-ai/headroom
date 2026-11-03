'use strict';
// From e2e/wrap/tools after installing its locked dependencies:
// node --test --test-concurrency=1 --test-reporter=tap dependency-regression.test.cjs
const path = require('node:path');
const { createRequire } = require('node:module');
const fromTools = createRequire(path.join(process.cwd(), 'package.json'));

if (process.argv[2] === '--serve') {
  const express = fromTools('express');
  const app = express();
  app.get('/health', (_req, res) => res.json({ ok: true }));
  app.post('/text', express.text({ verify() {} }), (req, res) => res.json(req.body));
  app.use((error, _req, res, _next) => res.status(error.status || 500).json({ type: error.type }));
  const server = app.listen(0, '127.0.0.1', () => process.send({ port: server.address().port }));
  process.on('message', (message) => {
    if (message === 'stop') { server.close(() => process.exit(0)); server.closeAllConnections(); }
  });
} else {
  const test = require('node:test');
  const assert = require('node:assert/strict');
  const http = require('node:http');
  const { fork } = require('node:child_process');

  test('Ajv rejects a percent-encoded scheme instead of aliasing an HTTPS schema', () => {
    const Ajv = fromTools('ajv');
    const ajv = new Ajv();
    ajv.addSchema({ $id: 'https://example.invalid/defs.json', type: 'integer' });
    const valid = ajv.compile({ $ref: 'https://example.invalid/defs.json' });
    assert.equal(valid(42), true);
    assert.equal(valid('42'), false);
    assert.throws(
      () => ajv.compile({ $ref: '%68ttps://example.invalid/defs.json' }),
      { message: 'URI scheme is malformed.' },
    );
  });

  test('unsupported charset with invalid gzip leaves the Express process healthy', async (t) => {
    const child = fork(__filename, ['--serve'], {
      cwd: process.cwd(), env: process.env, execArgv: [], silent: true,
    });
    let stderr = '';
    child.stdout.resume();
    child.stderr.on('data', (chunk) => { stderr += chunk; });
    t.after(async () => {
      t.diagnostic(JSON.stringify({ exitCode: child.exitCode, signal: child.signalCode, stderr }));
      if (child.exitCode !== null || child.signalCode !== null) return;
      await new Promise((resolve) => {
        const timer = setTimeout(() => { child.kill(); resolve(); }, 1000);
        child.once('exit', () => { clearTimeout(timer); resolve(); });
        if (child.connected) child.send('stop'); else child.kill();
      });
    });
    const port = await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('server startup timeout')), 3000);
      child.once('message', (msg) => { clearTimeout(timer); resolve(msg.port); });
      child.once('error', (err) => { clearTimeout(timer); reject(err); });
      child.once('exit', (code) => { clearTimeout(timer); reject(new Error('server exited: ' + code + stderr)); });
    });
    function request(route, body, headers = {}, method = 'POST') {
      return new Promise((resolve, reject) => {
        const req = http.request({ hostname: '127.0.0.1', port, path: route, method, agent: false,
          headers: { ...headers, 'Content-Length': Buffer.byteLength(body) } }, (res) => {
          const chunks = [];
          res.on('data', (chunk) => chunks.push(chunk));
          res.on('error', reject);
          res.on('end', () => resolve({ status: res.statusCode, body: Buffer.concat(chunks).toString() }));
        });
        req.setTimeout(2000, () => req.destroy(new Error('localhost request timeout')));
        req.on('error', reject);
        req.end(body);
      });
    }
    const response = await request('/text', 'not-gzip', {
      'Content-Type': 'text/plain; charset=not-a-real-charset', 'Content-Encoding': 'gzip',
    });
    assert.equal(response.status, 415);
    assert.equal(JSON.parse(response.body).type, 'charset.unsupported');
    // Let a pending zlib error surface before checking the child's health.
    await new Promise((resolve) => setTimeout(resolve, 80));
    assert.equal(child.exitCode, null, stderr);
    const health = await request('/health', '', {}, 'GET');
    assert.equal(health.status, 200);
    assert.deepEqual(JSON.parse(health.body), { ok: true });
  });
}
