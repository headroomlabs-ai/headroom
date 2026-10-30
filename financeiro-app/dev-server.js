// Servidor local para desenvolvimento: serve a pasta public e a mesma API usada na Vercel.
// Dados ficam na pasta .dados-dev (não vai para o Git). Defina ADMIN_SENHA para testar o login.
import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { tratar } from './api/_lib/router.js';
import { StoreArquivo } from './api/_lib/store-arquivo.js';
import { carregarDados } from './api/_lib/dados.js';
import { semearDemo } from './dev/demo.js';

const raiz = path.dirname(fileURLToPath(import.meta.url));
const pasta = process.env.DATA_DIR || path.join(raiz, process.env.DEMO ? '.dados-demo' : '.dados-dev');
const store = new StoreArquivo(pasta);
const MIME = { '.webmanifest': 'application/manifest+json', '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.png': 'image/png', '.svg': 'image/svg+xml' };

if (process.env.DEMO) {
  const d = await carregarDados(store);
  if (!d.lancamentos.length) { await semearDemo(store); console.log('Dados de demonstração criados.'); }
}

export const servidor = http.createServer(async (req, res) => {
  const url = new URL(req.url, 'http://localhost');
  if (url.pathname === '/api/app') {
    const chunks = [];
    for await (const c of req) chunks.push(c);
    let corpo = null;
    try { corpo = chunks.length ? JSON.parse(Buffer.concat(chunks).toString()) : null; } catch { /* corpo inválido */ }
    const query = Object.fromEntries(url.searchParams);
    const rota = query.rota || '';
    delete query.rota;
    const r = await tratar({ metodo: req.method, rota, query, corpo, cookie: req.headers.cookie, ip: req.socket.remoteAddress }, { store, senha: process.env.ADMIN_SENHA, semente: process.env.USUARIOS_SEMENTE });
    for (const [k, v] of Object.entries(r.cabecalhos || {})) res.setHeader(k, v);
    res.writeHead(r.status, { 'Content-Type': r.tipo || 'application/json; charset=utf-8', 'Cache-Control': 'no-store' });
    return res.end(r.tipo ? r.corpo : JSON.stringify(r.corpo));
  }
  const alvo = path.join(raiz, 'public', url.pathname === '/' ? 'index.html' : path.normalize(url.pathname));
  if (!alvo.startsWith(path.join(raiz, 'public'))) { res.writeHead(403); return res.end(); }
  try {
    res.writeHead(200, { 'Content-Type': MIME[path.extname(alvo)] || 'application/octet-stream', 'Cache-Control': 'no-store' });
    res.end(await fs.readFile(alvo));
  } catch { res.writeHead(404); res.end('Não encontrado'); }
});

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const porta = Number(process.env.PORT) || 3000;
  servidor.listen(porta, () => console.log(`Financeiro interno em http://localhost:${porta}${process.env.ADMIN_SENHA ? ' (com senha)' : ' (sem senha — só para desenvolvimento)'}`));
}
