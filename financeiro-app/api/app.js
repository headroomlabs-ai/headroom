// Função da Vercel: recebe todas as chamadas da API em /api/app?rota=...
import { tratar } from './_lib/router.js';
import { StoreBlob } from './_lib/store-blob.js';

const store = new StoreBlob();

export default async function handler(req, res) {
  const url = new URL(req.url, 'http://localhost');
  const query = Object.fromEntries(url.searchParams);
  const rota = query.rota || '';
  delete query.rota;
  let corpo = req.body;
  if (typeof corpo === 'string') { try { corpo = JSON.parse(corpo); } catch { corpo = null; } }
  const r = await tratar({
    metodo: req.method, rota, query, corpo, cookie: req.headers.cookie,
    ip: String(req.headers['x-forwarded-for'] || '').split(',')[0].trim(),
    https: req.headers['x-forwarded-proto'] === 'https',
  }, { store, senha: process.env.ADMIN_SENHA, semente: process.env.USUARIOS_SEMENTE, producao: !!process.env.VERCEL });
  for (const [k, v] of Object.entries(r.cabecalhos || {})) res.setHeader(k, v);
  res.setHeader('Cache-Control', 'no-store');
  res.setHeader('Content-Type', r.tipo || 'application/json; charset=utf-8');
  res.status(r.status).send(r.tipo ? r.corpo : JSON.stringify(r.corpo));
}
