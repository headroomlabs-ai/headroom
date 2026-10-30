// Leitura e gravação dos registros. Lê só o que mudou (compara o etag da listagem) e devolve cópias isoladas.
import { criarDados } from './finance.js';
import { MARCADOR, MARCADOR_SERVICOS, registrosDeServicos, registrosDoPlano } from './plano.js';

const PREFIXO = 'dados/';
const caches = new WeakMap(); // store -> Map(caminho -> { etag, obj })
const cacheDe = (store) => { if (!caches.has(store)) caches.set(store, new Map()); return caches.get(store); };
const caminhoDe = (colecao, id) => `${PREFIXO}${colecao}/${id}.json`;

async function emLotes(itens, tamanho, fn) {
  for (let i = 0; i < itens.length; i += tamanho) await Promise.all(itens.slice(i, i + tamanho).map(fn));
}

export async function gravar(store, colecao, obj) {
  const caminho = caminhoDe(colecao, obj.id);
  const etag = await store.gravar(caminho, obj);
  cacheDe(store).set(caminho, { etag, obj: structuredClone(obj) });
}
export async function excluir(store, colecao, id) {
  const caminho = caminhoDe(colecao, id);
  await store.excluir(caminho);
  cacheDe(store).delete(caminho);
}

async function semearPlano(store) {
  const { categorias, centros } = registrosDoPlano();
  await Promise.all([...categorias.map((c) => gravar(store, 'categorias', c)), ...centros.map((c) => gravar(store, 'centros', c))]);
  await store.gravar(MARCADOR, { criado_em: new Date().toISOString() });
}

async function semearServicos(store) {
  await Promise.all(registrosDeServicos().map((s) => gravar(store, 'servicos', s)));
  await store.gravar(MARCADOR_SERVICOS, { criado_em: new Date().toISOString() });
}

export async function carregarDados(store) {
  const cache = cacheDe(store);
  let lista = await store.listar(PREFIXO);
  if (!lista.some((x) => x.caminho === MARCADOR)) { await semearPlano(store); lista = await store.listar(PREFIXO); }
  if (!lista.some((x) => x.caminho === MARCADOR_SERVICOS)) { await semearServicos(store); lista = await store.listar(PREFIXO); }
  const vivos = new Set(lista.map((x) => x.caminho));
  for (const k of [...cache.keys()]) if (!vivos.has(k)) cache.delete(k);
  await emLotes(lista.filter((x) => cache.get(x.caminho)?.etag !== x.etag), 24, async (x) => {
    const obj = await store.ler(x.caminho);
    if (obj) cache.set(x.caminho, { etag: x.etag, obj }); else cache.delete(x.caminho);
  });
  const colecoes = {};
  for (const [caminho, { obj }] of cache) {
    const col = caminho.split('/')[1];
    if (col !== 'meta') (colecoes[col] ||= []).push(structuredClone(obj));
  }
  return criarDados(colecoes);
}
