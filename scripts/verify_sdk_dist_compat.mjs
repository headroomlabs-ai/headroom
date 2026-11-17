import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
const root = path.resolve(process.argv[2]);
const require = createRequire(import.meta.url);
const cjs = require(path.join(root, 'dist/index.cjs'));
const esm = await import(pathToFileURL(path.join(root, 'dist/index.js')));
for (const api of [cjs, esm]) {
  assert.equal(typeof api.HeadroomClient, 'function');
  assert.equal(typeof api.compress, 'function');
  const client = new api.HeadroomClient({ baseUrl: 'http://127.0.0.1:9' });
  assert.equal(typeof client.compress, 'function');
}
console.log(process.version + ': built ESM and CJS SDK entrypoints load and construct the client');
