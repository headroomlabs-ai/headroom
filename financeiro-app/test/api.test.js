import test from 'node:test';
import assert from 'node:assert/strict';
import { tratar } from '../api/_lib/router.js';
import { StoreMemoria } from '../api/_lib/store-memoria.js';
import { StoreBlob } from '../api/_lib/store-blob.js';
import { carregarDados } from '../api/_lib/dados.js';
import { criarSessao, sessaoValida, senhaConfere } from '../api/_lib/auth.js';

const HOJE = '2026-06-15';
const chamar = (store, metodo, rota, { query, corpo, senha, cookie } = {}) =>
  tratar({ metodo, rota, query, corpo, cookie, ip: '1.1.1.1' }, { store, senha, hoje: () => HOJE, agora: () => '2026-06-15T10:00:00.000Z' });
const j = async (store, ...a) => { const r = await chamar(store, ...a); return r; };

test('primeiro acesso cria o plano de contas padrão uma única vez', async () => {
  const store = new StoreMemoria();
  const cats = await j(store, 'GET', 'cadastros/categorias');
  assert.equal(cats.status, 200);
  assert.ok(cats.corpo.length >= 20 && cats.corpo.every((c) => c.grupo_dre));
  await j(store, 'GET', 'cadastros/categorias');
  assert.equal((await j(store, 'GET', 'cadastros/categorias')).corpo.length, cats.corpo.length);
});

test('fluxo completo pela API: conta, pessoa, lançamento, baixa, estorno e exclusão', async () => {
  const store = new StoreMemoria();
  const conta = (await j(store, 'POST', 'cadastros/contas', { corpo: { nome: 'Inter', saldo_inicial_cents: 100000 } })).corpo;
  const pessoa = (await j(store, 'POST', 'cadastros/pessoas', { corpo: { nome: 'Cliente', tipo: 'cliente' } })).corpo;
  const cat = (await j(store, 'GET', 'cadastros/categorias')).corpo.find((c) => c.tipo === 'receita');
  const novo = await j(store, 'POST', 'lancamentos', { corpo: { tipo: 'receita', nome: 'Projeto X', valor_total_cents: 90000, parcelas: 3,
    primeiro_vencimento: '2026-06-20', categoria_id: cat.id, pessoa_id: pessoa.id, conta_id: conta.id } });
  assert.equal(novo.status, 201);
  const lista = (await j(store, 'GET', 'parcelas', { query: { tipo: 'receita', de: '2026-01-01', ate: '2026-12-31' } })).corpo;
  assert.equal(lista.itens.length, 3);
  assert.equal(lista.itens[0].nome, 'Projeto X 1/3');
  const pid = lista.itens[0].id;
  assert.equal((await j(store, 'POST', `parcelas/${pid}/baixa`, { corpo: { data: '2026-06-20', conta_id: conta.id } })).status, 200);
  assert.equal((await j(store, 'GET', 'contas-saldos')).corpo[0].saldo_cents, 130000);
  assert.equal((await j(store, 'POST', `parcelas/${pid}/baixa`, { corpo: { conta_id: conta.id } })).status, 400);
  assert.equal((await j(store, 'POST', `parcelas/${pid}/estorno`)).status, 200);
  assert.equal((await j(store, 'GET', 'contas-saldos')).corpo[0].saldo_cents, 100000);
  assert.equal((await j(store, 'DELETE', `cadastros/contas/${conta.id}`)).status, 409); // em uso
  assert.equal((await j(store, 'DELETE', `lancamentos/${novo.corpo.id}`)).status, 200);
  assert.equal((await j(store, 'DELETE', `cadastros/contas/${conta.id}`)).status, 200);
  const csv = await j(store, 'GET', 'parcelas.csv', { query: { de: '2026-01-01', ate: '2026-12-31' } });
  assert.match(csv.tipo, /csv/);
});

test('erros de validação viram 400 e rota inexistente 404', async () => {
  const store = new StoreMemoria();
  assert.equal((await j(store, 'POST', 'lancamentos', { corpo: { tipo: 'receita', nome: '' } })).status, 400);
  assert.equal((await j(store, 'POST', 'cadastros/categorias', { corpo: { nome: 'x', tipo: 'despesa', grupo_dre: 'receita_bruta' } })).status, 400);
  assert.equal((await j(store, 'PUT', 'cadastros/contas/inexistente', { corpo: { nome: 'x' } })).status, 404);
  assert.equal((await j(store, 'GET', 'nada')).status, 404);
});

test('senha: sem sessão bloqueia tudo, login certo libera, errado nega, e a sessão expira', async () => {
  const store = new StoreMemoria();
  const senha = 'uma-senha-bem-forte-123';
  assert.equal((await j(store, 'GET', 'resumo', { senha })).status, 401);
  assert.equal((await j(store, 'GET', 'sessao', { senha })).corpo.autenticado, false);
  const errado = await j(store, 'POST', 'login', { senha, corpo: { senha: 'errada' } });
  assert.equal(errado.status, 401);
  const certo = await j(store, 'POST', 'login', { senha, corpo: { senha } });
  assert.equal(certo.status, 200);
  const cookie = certo.cabecalhos['Set-Cookie'].split(';')[0];
  assert.match(certo.cabecalhos['Set-Cookie'], /HttpOnly/);
  assert.equal((await j(store, 'GET', 'resumo', { senha, cookie })).status, 200);
  assert.equal((await j(store, 'GET', 'resumo', { senha, cookie: 'sessao=1.abc' })).status, 401);
  const velho = criarSessao(senha, Date.now() - 13 * 3600 * 1000);
  assert.equal(sessaoValida(velho, senha), false);
  assert.equal(sessaoValida(criarSessao(senha), 'outra-senha'), false);
  assert.ok(senhaConfere(senha, senha) && !senhaConfere('x', senha) && !senhaConfere('x', undefined));
}, { timeout: 10000 });

test('produção sem ADMIN_SENHA fica bloqueada (não abre por engano)', async () => {
  const r = await tratar({ metodo: 'GET', rota: 'resumo' }, { store: new StoreMemoria(), senha: undefined, producao: true });
  assert.equal(r.status, 503);
});

test('só relê o que mudou: segunda leitura não baixa registros de novo', async () => {
  const store = new StoreMemoria();
  await carregarDados(store);
  let leituras = 0;
  const ler = store.ler.bind(store);
  store.ler = async (c) => { leituras++; return ler(c); };
  await carregarDados(store);
  assert.equal(leituras, 0);
  await j(store, 'POST', 'cadastros/centros', { corpo: { nome: 'Novo' } });
  await carregarDados(store);
  assert.equal(leituras, 0); // o que acabou de ser gravado já está no cache
});

test('StoreBlob usa Blob privado, sem sufixo aleatório, com sobrescrita e sem cache', async () => {
  const chamadas = [];
  const arquivos = new Map();
  const sdk = {
    put: async (p, corpo, op) => { chamadas.push(['put', p, op]); arquivos.set(p, corpo); return { etag: 'x1' }; },
    get: async (p, op) => { chamadas.push(['get', p, op]); return arquivos.has(p) ? { statusCode: 200, stream: new Response(arquivos.get(p)).body } : null; },
    list: async (op) => { chamadas.push(['list', op.prefix]); return { blobs: [...arquivos.keys()].filter((k) => k.startsWith(op.prefix)).map((pathname) => ({ pathname, etag: 'x1' })), hasMore: false }; },
    del: async (p) => { chamadas.push(['del', p]); arquivos.delete(p); },
  };
  const store = new StoreBlob(sdk);
  await store.gravar('dados/a/1.json', { id: '1' });
  assert.deepEqual(await store.ler('dados/a/1.json'), { id: '1' });
  assert.equal(await store.ler('dados/a/2.json'), null);
  assert.deepEqual(await store.listar('dados/'), [{ caminho: 'dados/a/1.json', etag: 'x1' }]);
  await store.excluir('dados/a/1.json');
  const put = chamadas.find((c) => c[0] === 'put')[2];
  assert.equal(put.access, 'private'); assert.equal(put.addRandomSuffix, false); assert.equal(put.allowOverwrite, true);
  assert.deepEqual(chamadas.find((c) => c[0] === 'get')[2], { access: 'private', useCache: false });
});

test('cadastro de cliente: código automático, serviços padrão e lista de clientes', async () => {
  const store = new StoreMemoria();
  const servicos = (await j(store, 'GET', 'cadastros/servicos')).corpo;
  assert.ok(servicos.length >= 5);
  assert.equal((await j(store, 'GET', 'cadastros/servicos')).corpo.length, servicos.length); // semeado uma vez
  const cli = (await j(store, 'POST', 'cadastros/pessoas', { corpo: { nome: 'Maria', tipo: 'cliente', cidade: 'Goiânia' } })).corpo;
  assert.equal(cli.codigo, 'CLI-0001');
  const forn = (await j(store, 'POST', 'cadastros/pessoas', { corpo: { nome: 'Eng', tipo: 'fornecedor' } })).corpo;
  assert.equal(forn.codigo, null);
  assert.equal((await j(store, 'GET', 'sugestoes-codigo')).corpo.cliente, 'CLI-0002');
  const proj = await j(store, 'POST', 'cadastros/contratos', { corpo: { codigo: 'CA260601', nome: 'Casa', pessoa_id: cli.id, area_m2: 150,
    servicos: [{ servico_id: servicos[0].id, nome: servicos[0].nome, valor_cents: 500000 }] } });
  assert.equal(proj.status, 201);
  assert.equal(proj.corpo.valor_total_cents, 500000);
  assert.equal((await j(store, 'POST', 'cadastros/contratos', { corpo: { codigo: 'ca260601', nome: 'Outro' } })).status, 409);
  const lista = (await j(store, 'GET', 'clientes')).corpo;
  assert.equal(lista.itens.length, 1);
  assert.deepEqual(lista.itens[0].servicos, [servicos[0].nome]);
  assert.equal((await j(store, 'DELETE', `cadastros/servicos/${servicos[0].id}`)).status, 409);
});

test('usuários por e-mail: criar com senha provisória, entrar, trocar senha, excluir e semente do ambiente', async () => {
  const store = new StoreMemoria();
  const SENHA = 'senha-admin-1234';
  const cookieDe = (r) => r.cabecalhos['Set-Cookie'].split(';')[0];
  const admin = await chamar(store, 'POST', 'login', { corpo: { senha: SENHA }, senha: SENHA });
  const ck = cookieDe(admin);
  const novo = await chamar(store, 'POST', 'usuarios', { corpo: { email: ' Financeiro@Cariati.com.br ', nome: 'Financeiro' }, senha: SENHA, cookie: ck });
  assert.equal(novo.status, 201);
  assert.equal(novo.corpo.usuario.email, 'financeiro@cariati.com.br');
  assert.equal(novo.corpo.usuario.senha_hash, undefined);
  const prov = novo.corpo.senha_provisoria;
  assert.ok(prov.length >= 12);
  assert.equal((await chamar(store, 'POST', 'usuarios', { corpo: { email: 'financeiro@cariati.com.br', nome: 'x' }, senha: SENHA, cookie: ck })).status, 409);
  assert.equal((await chamar(store, 'POST', 'usuarios', { corpo: { nome: 'x' }, senha: SENHA })).status, 401); // sem sessão
  // login do usuário: senha errada, e certa
  assert.equal((await chamar(store, 'POST', 'login', { corpo: { email: 'financeiro@cariati.com.br', senha: 'errada' }, senha: SENHA })).status, 401);
  const r = await chamar(store, 'POST', 'login', { corpo: { email: 'FINANCEIRO@cariati.com.br', senha: prov }, senha: SENHA });
  assert.equal(r.status, 200);
  const ckU = cookieDe(r);
  const sess = await chamar(store, 'GET', 'sessao', { senha: SENHA, cookie: ckU });
  assert.equal(sess.corpo.autenticado, true); assert.equal(sess.corpo.usuario.email, 'financeiro@cariati.com.br');
  assert.equal((await chamar(store, 'GET', 'cadastros/contas', { senha: SENHA, cookie: ckU })).status, 200); // mesmo acesso do admin
  assert.equal((await chamar(store, 'GET', 'usuarios', { senha: SENHA, cookie: ckU })).corpo.length, 1);
  // trocar a própria senha
  assert.equal((await chamar(store, 'POST', 'minha-senha', { corpo: { atual: 'x', nova: 'nova-senha-123' }, senha: SENHA, cookie: ckU })).status, 400);
  assert.equal((await chamar(store, 'POST', 'minha-senha', { corpo: { atual: prov, nova: 'curta' }, senha: SENHA, cookie: ckU })).status, 400);
  assert.equal((await chamar(store, 'POST', 'minha-senha', { corpo: { atual: prov, nova: 'nova-senha-123' }, senha: SENHA, cookie: ckU })).status, 200);
  assert.equal((await chamar(store, 'POST', 'login', { corpo: { email: 'financeiro@cariati.com.br', senha: prov }, senha: SENHA })).status, 401);
  assert.equal((await chamar(store, 'POST', 'login', { corpo: { email: 'financeiro@cariati.com.br', senha: 'nova-senha-123' }, senha: SENHA })).status, 200);
  // redefinir pelo admin e excluir: sessão do excluído deixa de valer
  const id = novo.corpo.usuario.id;
  assert.equal((await chamar(store, 'POST', `usuarios/${id}/senha`, { senha: SENHA, cookie: ck })).status, 200);
  assert.equal((await chamar(store, 'DELETE', `usuarios/${id}`, { senha: SENHA, cookie: ckU })).status, 409); // não exclui a si mesmo
  assert.equal((await chamar(store, 'DELETE', `usuarios/${id}`, { senha: SENHA, cookie: ck })).status, 200);
  assert.equal((await chamar(store, 'GET', 'cadastros/contas', { senha: SENHA, cookie: ckU })).status, 401);
  // semente do ambiente: cria uma vez e não sobrescreve a senha trocada depois
  const loja2 = new StoreMemoria();
  const semente = JSON.stringify([{ email: 'a@b.com', nome: 'A', senha: 'senha-semente-1' }]);
  const comSemente = (corpo) => tratar({ metodo: 'POST', rota: 'login', corpo, ip: '2.2.2.2' }, { store: loja2, senha: SENHA, semente, hoje: () => HOJE });
  assert.equal((await comSemente({ email: 'a@b.com', senha: 'senha-semente-1' })).status, 200);
  assert.equal((await comSemente({ email: 'a@b.com', senha: 'outra' })).status, 401);
  // excluído depois de semeado, não volta
  const idSem = (await tratar({ metodo: 'GET', rota: 'usuarios', cookie: `sessao=${criarSessao(SENHA)}`, ip: '3.3.3.3' }, { store: loja2, senha: SENHA, hoje: () => HOJE })).corpo[0].id;
  await tratar({ metodo: 'DELETE', rota: `usuarios/${idSem}`, cookie: `sessao=${criarSessao(SENHA)}`, ip: '3.3.3.3' }, { store: loja2, senha: SENHA, hoje: () => HOJE });
  assert.equal((await comSemente({ email: 'a@b.com', senha: 'senha-semente-1' })).status, 401);
});
