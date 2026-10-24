// Todas as rotas da API. Usado pela função da Vercel e pelo servidor local.
import * as F from './finance.js';
import { carregarDados, excluir, gravar } from './dados.js';
import { cookieLimpo, cookieSessao, criarSessao, hashConfere, lerCookie, lerSessao, senhaConfere } from './auth.js';
import * as U from './usuarios.js';
import { hojeISO } from './dates.js';

const ok = (corpo, status = 200) => ({ status, corpo });
const csv = (nome, linhas) => {
  const esc = (v) => `"${String(v ?? '').replaceAll('"', '""')}"`;
  return { status: 200, tipo: 'text/csv; charset=utf-8', cabecalhos: { 'Content-Disposition': `attachment; filename="${nome}"` },
    corpo: '﻿' + linhas.map((l) => l.map(esc).join(';')).join('\r\n') };
};
const reais = (c) => (c / 100).toFixed(2).replace('.', ',');
const STATUS_TXT = { pago: 'Pago', vencido: 'Vencido', vence_hoje: 'Vence hoje', a_vencer: 'A vencer' };
const dormir = (ms) => new Promise((r) => setTimeout(r, ms));

// Contagem simples de erros de login por endereço (vale por instância; a senha forte é a defesa principal).
const falhas = new Map();
const JANELA_MS = 10 * 60 * 1000;

function filtrosParcela(q) {
  return { tipo: q.tipo, campo: q.campo, de: q.de, ate: q.ate, status: q.status, busca: q.busca, categoria_id: q.categoria_id,
    pessoa_id: q.pessoa_id, contrato_id: q.contrato_id, centro_custo_id: q.centro_custo_id, page: Number(q.page) || undefined,
    pageSize: Number(q.pageSize) || undefined, ordem: q.ordem };
}
const ano = (q) => Number(q.ano) || new Date().getFullYear();

export async function tratar(req, ctx) {
  const { store, senha, semente = '', producao = false, hoje = hojeISO, agora = () => new Date().toISOString() } = ctx;
  const { metodo, query = {}, corpo = {}, cookie, ip = '' } = req;
  const seguro = req.https || producao;
  const partes = String(req.rota || '').split('/').filter(Boolean);
  const [a, b, c] = partes;
  const body = corpo && typeof corpo === 'object' ? corpo : {};
  const H = hoje();

  try {
    if (producao && !senha) return ok({ erro: 'Sistema sem senha configurada (ADMIN_SENHA). Acesso bloqueado.' }, 503);
    const exigeSenha = !!senha;
    const sessao = lerSessao(lerCookie(cookie, 'sessao'), senha);
    // Sessão de usuário só vale enquanto o usuário existir e estiver ativo.
    const usuario = sessao?.usuario ? await U.acharPorEmail(store, sessao.usuario) : null;
    const autenticado = !exigeSenha || (!!sessao && (!sessao.usuario || !!usuario?.ativo));

    if (a === 'sessao' && metodo === 'GET') return ok({ autenticado, exige_senha: exigeSenha, usuario: autenticado && usuario ? U.publico(usuario) : null });
    if (a === 'login' && metodo === 'POST') {
      if (!exigeSenha) return ok({ ok: true });
      const f = falhas.get(ip);
      if (f && f.n >= 8 && Date.now() - f.desde < JANELA_MS) return ok({ erro: 'Muitas tentativas. Aguarde alguns minutos.' }, 429);
      let entrou = null; // '' = administrador; senão o e-mail do usuário
      if (body.email && String(body.email).trim()) {
        if (semente) await U.semear(store, semente);
        const u = await U.entrar(store, body.email, body.senha);
        if (u) entrou = u.email;
      } else if (senhaConfere(body.senha, senha)) entrou = '';
      if (entrou === null) {
        const atual = f && Date.now() - f.desde < JANELA_MS ? f : { n: 0, desde: Date.now() };
        falhas.set(ip, { n: atual.n + 1, desde: atual.desde });
        await dormir(800);
        return ok({ erro: body.email ? 'E-mail ou senha incorretos.' : 'Senha incorreta.' }, 401);
      }
      falhas.delete(ip);
      return { ...ok({ ok: true }), cabecalhos: { 'Set-Cookie': cookieSessao(criarSessao(senha, Date.now(), entrou), seguro) } };
    }
    if (a === 'logout' && metodo === 'POST') return { ...ok({ ok: true }), cabecalhos: { 'Set-Cookie': cookieLimpo(seguro) } };
    if (!autenticado) return ok({ erro: 'Sessão expirada. Entre novamente.' }, 401);

    // ----- usuários (todos têm o mesmo acesso do administrador) -----
    if (a === 'usuarios') {
      if (metodo === 'GET' && !b) return ok((await U.listarUsuarios(store)).map(U.publico));
      if (metodo === 'POST' && !b) return ok(await U.criarUsuario(store, body, agora()), 201);
      if (metodo === 'POST' && b && c === 'senha') return ok({ senha_provisoria: await U.definirSenha(store, b, null) });
      if (metodo === 'DELETE' && b) {
        if (usuario?.id === b) throw new F.ErroValidacao('Você não pode excluir o próprio usuário.', 409);
        await U.excluirUsuario(store, b);
        return ok({ ok: true });
      }
    }
    if (a === 'minha-senha' && metodo === 'POST') {
      if (!usuario) throw new F.ErroValidacao('Só usuários com e-mail podem trocar a senha por aqui. A senha de administrador é trocada na Vercel.', 400);
      if (!hashConfere(body.atual, usuario.senha_hash)) throw new F.ErroValidacao('A senha atual está incorreta.', 400);
      await U.definirSenha(store, usuario.id, String(body.nova || ''));
      return ok({ ok: true });
    }

    const d = await carregarDados(store);

    // ----- cadastros -----
    if (a === 'cadastros' && F.TIPOS_CADASTRO.includes(b)) {
      if (metodo === 'GET' && !c) {
        const lista = [...d[b]];
        if (b === 'contratos') return ok(lista.sort((x, y) => y.codigo.localeCompare(x.codigo)).map((x) => ({ ...x, pessoa_nome: d.mapa.pessoas.get(x.pessoa_id)?.nome ?? null })));
        if (b === 'categorias') return ok(lista.sort((x, y) => x.tipo.localeCompare(y.tipo) || x.nome.localeCompare(y.nome)));
        return ok(lista.sort((x, y) => x.nome.localeCompare(y.nome)));
      }
      if (metodo === 'POST' && !c) {
        const reg = { ...F.validarCadastro(d, b, body), id: F.novoId() };
        if (b === 'pessoas') {
          reg.criado_em = agora();
          if (!reg.codigo && reg.tipo !== 'fornecedor') reg.codigo = F.sugestoesCodigo(d, H).cliente;
        }
        await gravar(store, b, reg);
        return ok(reg, 201);
      }
      if (metodo === 'PUT' && c) {
        const existente = d.mapa[b].get(c);
        if (!existente) throw new F.ErroValidacao('Registro não encontrado.', 404);
        const reg = { ...existente, ...F.validarCadastro(d, b, body, existente), id: c };
        await gravar(store, b, reg);
        return ok(reg);
      }
      if (metodo === 'DELETE' && c) {
        if (!d.mapa[b].get(c)) throw new F.ErroValidacao('Registro não encontrado.', 404);
        if (F.emUso(d, b, c)) throw new F.ErroValidacao('Este registro está em uso e não pode ser excluído.', 409);
        await excluir(store, b, c);
        return ok({ ok: true });
      }
    }

    // ----- lançamentos e parcelas -----
    if (a === 'parcelas' && !b && metodo === 'GET') return ok(F.listarParcelas(d, filtrosParcela(query), H));
    if (a === 'parcelas.csv' && metodo === 'GET') {
      const r = F.listarParcelas(d, { ...filtrosParcela(query), todos: true }, H);
      return csv('lancamentos.csv', [
        ['Nome', 'Valor', 'Valor pago', 'Em aberto', 'Vencimento', 'Data de pagamento', 'Pessoa', 'Categoria', 'Projeto', 'Nota fiscal', 'Status'],
        ...r.itens.map((p) => [p.nome, reais(p.valor_cents), reais(p.valor_pago_cents), reais(p.aberto_cents), p.vencimento, p.data_pagamento || '',
          p.pessoa_nome, p.categoria_nome, p.contrato_codigo, p.nota_fiscal, STATUS_TXT[p.status]])]);
    }
    if (a === 'lancamentos' && !b && metodo === 'POST') {
      const l = F.criarLancamento(d, body, agora());
      await gravar(store, 'lancamentos', l);
      return ok({ id: l.id }, 201);
    }
    if (a === 'lancamentos' && b && metodo === 'DELETE') {
      if (!d.mapa.lancamentos.get(b)) throw new F.ErroValidacao('Lançamento não encontrado.', 404);
      await excluir(store, 'lancamentos', b);
      return ok({ ok: true });
    }
    if (a === 'parcelas' && b && metodo === 'POST' && ['baixa', 'estorno'].includes(c)) {
      const l = c === 'baixa' ? F.baixarParcela(d, b, body) : F.estornarParcela(d, b);
      await gravar(store, 'lancamentos', l);
      return ok({ ok: true });
    }
    if (a === 'parcelas' && b && metodo === 'PUT') {
      await gravar(store, 'lancamentos', F.editarParcela(d, b, body));
      return ok({ ok: true });
    }

    // ----- contas e transferências -----
    if (a === 'contas-saldos' && metodo === 'GET') return ok(F.saldoContas(d, null, H));
    if (a === 'contas' && b && c === 'extrato' && metodo === 'GET') return ok(F.extratoConta(d, b, H));
    if (a === 'transferencias' && !b && metodo === 'GET') return ok(F.listarTransferencias(d, query));
    if (a === 'transferencias' && !b && metodo === 'POST') {
      const t = F.criarTransferencia(d, body, agora());
      await gravar(store, 'transferencias', t);
      return ok({ id: t.id }, 201);
    }
    if (a === 'transferencias' && b && metodo === 'DELETE') {
      await excluir(store, 'transferencias', b);
      return ok({ ok: true });
    }

    // ----- dashboard e relatórios -----
    if (metodo === 'GET') {
      if (a === 'a-receber') return ok(F.aReceber(d, query, H));
      if (a === 'clientes') return ok(F.clientes(d, query, H));
      if (a === 'sugestoes-codigo') return ok(F.sugestoesCodigo(d, H));
      if (a === 'resumo') return ok(F.resumo(d, query, H));
      if (a === 'pagamentos-cliente') return ok(F.pagamentosCliente(d, query, H));
      if (a === 'relatorios' && b === 'fluxo-caixa') return ok(F.fluxoCaixa(d, ano(query), H));
      if (a === 'relatorios' && b === 'dre') return ok(F.dre(d, ano(query), { regime: query.regime }, H));
      if (a === 'relatorios' && b === 'resultados') return ok(F.resultadosGerais(d, query, H));
      if (a === 'relatorios' && b === 'resultados-projeto') return ok(F.resultadosPorProjeto(d, query, H));
      if (a === 'relatorios' && b === 'outros') return ok(F.outrosRelatorios(d, query, H));
      if (a === 'relatorios' && b === 'outros.csv') {
        const r = F.outrosRelatorios(d, query, H);
        return csv('relatorio.csv', [['Nome', 'Lançamentos', 'Valor', 'Pago', 'Em aberto'],
          ...r.itens.map((i) => [i.nome, i.qtd, reais(i.valor_cents), reais(i.pago_cents), reais(i.aberto_cents)]),
          ['Total', r.total.qtd, reais(r.total.valor_cents), reais(r.total.pago_cents), reais(r.total.aberto_cents)]]);
      }
    }
    return ok({ erro: 'Rota não encontrada.' }, 404);
  } catch (e) {
    if (e instanceof F.ErroValidacao) return ok({ erro: e.message }, e.status);
    console.error('erro na API', metodo, req.rota, e);
    return ok({ erro: 'Erro interno.' }, 500);
  }
}
