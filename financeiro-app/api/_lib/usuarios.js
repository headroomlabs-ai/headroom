// Usuários do sistema (login por e-mail). Ficam fora de dados/ e nunca saem da API com o hash da senha.
import { ErroValidacao, novoId } from './finance.js';
import { gerarSenha, hashConfere, hashSenha } from './auth.js';

const PREFIXO = 'usuarios/';
const caminho = (id) => `${PREFIXO}${id}.json`;
const EMAIL = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
export const normalizarEmail = (e) => String(e || '').trim().toLowerCase();
export const publico = ({ senha_hash, ...u }) => u;

export async function listarUsuarios(store) {
  const itens = await store.listar(PREFIXO);
  const lista = (await Promise.all(itens.map((x) => store.ler(x.caminho)))).filter(Boolean);
  return lista.sort((a, b) => a.nome.localeCompare(b.nome));
}
export async function acharPorEmail(store, email) {
  const e = normalizarEmail(email);
  return (await listarUsuarios(store)).find((u) => u.email === e) ?? null;
}

// Cria o usuário com uma senha provisória (devolvida uma única vez) ou com a senha informada.
export async function criarUsuario(store, { email, nome, senha }, agora = new Date().toISOString()) {
  const e = normalizarEmail(email);
  if (!EMAIL.test(e)) throw new ErroValidacao('Informe um e-mail válido.');
  if (!String(nome || '').trim()) throw new ErroValidacao('Informe o nome.');
  if (await acharPorEmail(store, e)) throw new ErroValidacao('Já existe um usuário com este e-mail.', 409);
  const provisoria = senha ? null : gerarSenha();
  const u = { id: novoId(), email: e, nome: String(nome).trim(), perfil: 'financeiro', ativo: 1, criado_em: agora, senha_hash: hashSenha(senha || provisoria) };
  await store.gravar(caminho(u.id), u);
  return { usuario: publico(u), senha_provisoria: provisoria };
}
export async function definirSenha(store, id, senha) {
  const u = await store.ler(caminho(id));
  if (!u) throw new ErroValidacao('Usuário não encontrado.', 404);
  const nova = senha || gerarSenha();
  if (String(nova).length < 8) throw new ErroValidacao('A senha deve ter pelo menos 8 caracteres.');
  await store.gravar(caminho(id), { ...u, senha_hash: hashSenha(nova) });
  return senha ? null : nova;
}
export async function excluirUsuario(store, id) {
  if (!await store.ler(caminho(id))) throw new ErroValidacao('Usuário não encontrado.', 404);
  await store.excluir(caminho(id));
}
export async function entrar(store, email, senha) {
  const u = await acharPorEmail(store, email);
  return u && u.ativo && hashConfere(senha, u.senha_hash) ? u : null;
}
// Usuários iniciais vindos do ambiente (USUARIOS_SEMENTE = JSON [{email, nome, senha}]). Cada e-mail é criado uma única vez,
// então trocar a senha ou excluir o usuário pela tela continua valendo.
export async function semear(store, json) {
  let lista;
  try { lista = JSON.parse(json || '[]'); } catch { return; }
  for (const x of Array.isArray(lista) ? lista : []) {
    if (!x?.email || !x?.senha) continue;
    // Marca cada e-mail como semeado: um usuário excluído depois não volta sozinho.
    const marca = `usuarios-meta/semente-${Buffer.from(normalizarEmail(x.email)).toString('hex')}.json`;
    if (await store.ler(marca)) continue;
    if (!await acharPorEmail(store, x.email)) await criarUsuario(store, { email: x.email, nome: x.nome || x.email, senha: x.senha });
    await store.gravar(marca, { criado_em: new Date().toISOString() });
  }
}
