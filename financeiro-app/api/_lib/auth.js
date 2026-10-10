// Senha única do dono. A senha vem da variável de ambiente ADMIN_SENHA e é conferida sempre no servidor.
import crypto from 'node:crypto';

const sha = (s) => crypto.createHash('sha256').update(String(s)).digest();
export const senhaConfere = (recebida, esperada) => !!esperada && crypto.timingSafeEqual(sha(recebida ?? ''), sha(esperada));

const DURACAO_S = 12 * 3600;
const assinatura = (senha, dados) => crypto.createHmac('sha256', sha(`sessao:${senha}`)).update(String(dados)).digest('hex');

// Token da sessão: <validade>.<usuário em base64url, ou vazio para o administrador>.<assinatura>.
// A assinatura usa a ADMIN_SENHA como segredo; sem ela ninguém consegue forjar uma sessão.
export function criarSessao(senha, agoraMs = Date.now(), usuario = '') {
  const exp = Math.floor(agoraMs / 1000) + DURACAO_S;
  const sub = usuario ? Buffer.from(usuario).toString('base64url') : '';
  return `${exp}.${sub}.${assinatura(senha, `${exp}.${sub}`)}`;
}
// Devolve { usuario } (e-mail, ou '' para o administrador) se a sessão é válida; senão null.
export function lerSessao(token, senha, agoraMs = Date.now()) {
  if (!token || !senha) return null;
  const [exp, sub = '', sig] = String(token).split('.');
  if (!/^\d+$/.test(exp || '') || !sig || Number(exp) < agoraMs / 1000) return null;
  const esperada = assinatura(senha, `${exp}.${sub}`);
  if (sig.length !== esperada.length || !crypto.timingSafeEqual(Buffer.from(sig), Buffer.from(esperada))) return null;
  return { usuario: sub ? Buffer.from(sub, 'base64url').toString() : '' };
}
export const sessaoValida = (token, senha, agoraMs) => !!lerSessao(token, senha, agoraMs);

// Senhas dos usuários: scrypt com sal próprio. Só o hash é guardado.
export function hashSenha(senha) {
  const sal = crypto.randomBytes(16).toString('hex');
  return `scrypt$${sal}$${crypto.scryptSync(String(senha), sal, 32).toString('hex')}`;
}
export function hashConfere(senha, guardado) {
  const [tipo, sal, hash] = String(guardado || '').split('$');
  if (tipo !== 'scrypt' || !sal || !hash) return false;
  const calc = crypto.scryptSync(String(senha ?? ''), sal, 32);
  const esperado = Buffer.from(hash, 'hex');
  return calc.length === esperado.length && crypto.timingSafeEqual(calc, esperado);
}
export const gerarSenha = () => crypto.randomBytes(12).toString('base64url');
export function lerCookie(cabecalho, nome) {
  for (const parte of String(cabecalho || '').split(';')) {
    const [k, ...v] = parte.trim().split('=');
    if (k === nome) return v.join('=');
  }
  return null;
}
export const cookieSessao = (token, seguro) =>
  `sessao=${token}; HttpOnly; SameSite=Strict; Path=/; Max-Age=${DURACAO_S}${seguro ? '; Secure' : ''}`;
export const cookieLimpo = (seguro) => `sessao=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0${seguro ? '; Secure' : ''}`;
