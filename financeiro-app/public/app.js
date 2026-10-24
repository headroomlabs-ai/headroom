import { $, $$, api, esc, brl, dataBR, setPeriodo, toast } from './util.js';
import { estadoErro, icon } from './ui.js';
import { formLancamento, formTransferencia, formCadastro } from './forms.js';
import { modal } from './util.js';
import * as V from './views.js';
import { clientesView, formCliente } from './clientes.js';
import { aReceberView } from './areceber.js';
import { novoUsuario, usuariosView } from './usuarios.js';

// ---------- telas ----------
const novoMenu = () => abrirNovo();
const ROTAS = {
  resumo: { titulo: 'Resumo', grupo: 'Financeiro', icone: 'home', sub: 'Caixa, contas e próximos vencimentos num só olhar.', fn: (el) => V.resumo(el), cta: { rotulo: 'Novo lançamento', icone: 'plus', acao: novoMenu } },
  areceber: { titulo: 'A receber', grupo: 'Financeiro', icone: 'alvo', sub: 'Quem está devendo, o que vence e como isso afeta o caixa.', fn: (el) => aReceberView(el), cta: { rotulo: 'Nova receita', icone: 'plus', acao: () => formLancamento('receita', renderAgora) } },
  clientes: { titulo: 'Clientes', grupo: 'Financeiro', icone: 'usuarios', sub: 'Quem são os clientes, seus códigos, projetos e serviços contratados.', fn: (el) => clientesView(el), cta: { rotulo: 'Novo cliente', icone: 'plus', acao: () => formCliente(renderAgora) } },
  pagamentos: { titulo: 'Pagamentos do cliente', grupo: 'Financeiro', icone: 'contrato', sub: 'Contratos, parcelas e quanto ainda falta receber.', fn: (el) => V.pagamentos(el), cta: { rotulo: 'Nova receita de contrato', icone: 'plus', acao: () => formLancamento('receita', renderAgora) } },
  receitas: { titulo: 'Receitas', grupo: 'Financeiro', icone: 'receitas', sub: 'Tudo o que entra: vencimentos, recebimentos e atrasos.', fn: (el, q) => V.lista(el, 'receita', q), cta: { rotulo: 'Nova receita', icone: 'plus', acao: () => formLancamento('receita', renderAgora) } },
  despesas: { titulo: 'Despesas', grupo: 'Financeiro', icone: 'despesas', sub: 'Tudo o que sai: contas a pagar, pagas e atrasadas.', fn: (el, q) => V.lista(el, 'despesa', q), cta: { rotulo: 'Nova despesa', icone: 'plus', acao: () => formLancamento('despesa', renderAgora) } },
  transferencias: { titulo: 'Transferências', grupo: 'Financeiro', icone: 'transfer', sub: 'Movimentos entre as suas próprias contas.', fn: (el) => V.transferencias(el), cta: { rotulo: 'Nova transferência', icone: 'plus', violeta: true, acao: () => formTransferencia(renderAgora) } },
  contas: { titulo: 'Contas e extratos', grupo: 'Financeiro', icone: 'banco', sub: 'Saldo de cada conta e o extrato dia a dia.', fn: (el, q) => V.contas(el, q), cta: { rotulo: 'Nova conta', icone: 'plus', violeta: true, acao: () => formCadastro('contas', null, renderAgora) } },
  fluxo: { titulo: 'Fluxo de caixa', grupo: 'Relatórios', icone: 'fluxo', sub: 'Previsto × realizado, mês a mês.', fn: (el) => V.fluxo(el), cta: { rotulo: 'Novo lançamento', icone: 'plus', acao: novoMenu } },
  dre: { titulo: 'DRE gerencial', grupo: 'Relatórios', icone: 'pizza', sub: 'Resultado do escritório por competência ou por caixa.', fn: (el) => V.dre(el), cta: { rotulo: 'Novo lançamento', icone: 'plus', acao: novoMenu } },
  resultados: { titulo: 'Resultados', grupo: 'Relatórios', icone: 'tendencia', sub: 'Quanto sobrou: geral e por projeto.', fn: (el, q) => V.resultados(el, q), cta: { rotulo: 'Novo lançamento', icone: 'plus', acao: novoMenu } },
  outros: { titulo: 'Outros relatórios', grupo: 'Relatórios', icone: 'camadas', sub: 'Totais por cliente, fornecedor, categoria, centro de custo ou projeto.', fn: (el) => V.outros(el), cta: { rotulo: 'Novo lançamento', icone: 'plus', acao: novoMenu } },
  usuarios: { titulo: 'Usuários', grupo: 'Configurações', icone: 'usuarios', sub: 'Quem entra no sistema, com e-mail e senha própria.', fn: (el) => usuariosView(el), cta: { rotulo: 'Novo usuário', icone: 'plus', acao: () => novoUsuario(renderAgora) } },
  cadastros: { titulo: 'Cadastros', grupo: 'Configurações', icone: 'ajustes', sub: 'Contas, categorias, centros de custo, pessoas e projetos.', fn: (el, q) => V.cadastrosView(el, q),
    cta: { rotulo: 'Novo cadastro', icone: 'plus', acao: () => formCadastro(V.cadastrosView.aba || 'contas', null, renderAgora) } },
};
const GRUPOS_NAV = ['Financeiro', 'Relatórios', 'Configurações'];
const BOTTOM = [['voltar', 'Voltar', 'voltar'], ['resumo', 'Início', 'home'], ['receitas', 'Receitas', 'receitas'], ['despesas', 'Despesas', 'despesas'], ['menu', 'Módulos', 'modulos']];

const ler = (k, padrao = null) => { try { return localStorage.getItem(k) ?? padrao; } catch { return padrao; } };
const gravar = (k, v) => { try { localStorage.setItem(k, v); } catch { /* sem storage */ } };

// ---------- ícones fixos do HTML ----------
$$('[data-ic]').forEach((e) => { e.innerHTML = icon(e.dataset.ic); });
$('#menu-btn').innerHTML = icon('menu');

// ---------- navegação lateral ----------
let favoritos = [];
try { favoritos = JSON.parse(ler('favoritos', '[]')) || []; } catch { favoritos = []; }
function montarNav() {
  const item = (k) => { const r = ROTAS[k]; const fav = favoritos.includes(k);
    return `<div class="nav-item"><a class="nav-link" href="#/${k}" data-k="${k}" title="${esc(r.titulo)}">${icon(r.icone)}<span class="rot">${esc(r.titulo)}</span></a>
      <button class="fav${fav ? ' on' : ''}" data-fav="${k}" aria-pressed="${fav}" aria-label="${fav ? 'Remover dos favoritos' : 'Adicionar aos favoritos'}: ${esc(r.titulo)}">${icon('star')}</button></div>`; };
  $('#nav').innerHTML = (favoritos.length ? `<div class="nav-grupo">Favoritos</div>${favoritos.filter((k) => ROTAS[k]).map(item).join('')}` : '')
    + GRUPOS_NAV.map((g) => `<div class="nav-grupo">${g}</div>${Object.keys(ROTAS).filter((k) => ROTAS[k].grupo === g).map(item).join('')}`).join('');
  marcarAtivo();
}
$('#nav').addEventListener('click', (e) => {
  const b = e.target.closest('[data-fav]');
  if (b) { e.preventDefault(); const k = b.dataset.fav; favoritos = favoritos.includes(k) ? favoritos.filter((x) => x !== k) : [...favoritos, k]; gravar('favoritos', JSON.stringify(favoritos)); montarNav(); return; }
  if (e.target.closest('.nav-link')) fecharGaveta();
});
let chaveAtual = 'resumo';
function marcarAtivo() {
  $$('.nav-link').forEach((a) => { const on = a.dataset.k === chaveAtual; a.classList.toggle('ativo', on); if (on) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current'); });
  $$('.bn-item').forEach((b) => b.classList.toggle('ativo', b.dataset.bn === chaveAtual));
}
// recolher o menu (desktop)
const app = $('#app');
function aplicarRecolhido(v) {
  app.classList.toggle('recolhido', v);
  $('#colapsar').innerHTML = icon(v ? 'expandir' : 'recolher');
  $('#colapsar').setAttribute('aria-label', v ? 'Expandir menu' : 'Recolher menu');
}
aplicarRecolhido(ler('menuRecolhido') === '1');
$('#colapsar').onclick = () => { const v = !app.classList.contains('recolhido'); aplicarRecolhido(v); gravar('menuRecolhido', v ? '1' : '0'); };
// gaveta (mobile)
const gaveta = (abrir) => { document.body.classList.toggle('gaveta-aberta', abrir); $('#side').classList.toggle('aberto', abrir); $('#scrim').hidden = !abrir; $('#menu-btn').setAttribute('aria-expanded', String(abrir)); };
const fecharGaveta = () => gaveta(false);
$('#menu-btn').onclick = () => gaveta(!$('#side').classList.contains('aberto'));
$('#scrim').onclick = fecharGaveta;
$('#sair').onclick = async () => { await api('logout', { method: 'POST' }).catch(() => {}); telaLogin(); };
$('#side-novo').onclick = () => { fecharGaveta(); abrirNovo(); };

// barra inferior (mobile)
$('#bottomnav').innerHTML = BOTTOM.map(([k, rot, ic]) => `<button class="bn-item" data-bn="${k}" type="button" aria-label="${rot}">${icon(ic)}<span>${rot}</span></button>`).join('');
$('#bottomnav').onclick = (e) => {
  const b = e.target.closest('[data-bn]');
  if (!b) return;
  const k = b.dataset.bn;
  if (k === 'voltar') { if (history.length > 1) history.back(); else location.hash = '#/resumo'; } else if (k === 'menu') gaveta(true); else location.hash = `#/${k}`;
};

// ---------- tema ----------
function aplicarTema(t) {
  document.documentElement.setAttribute('data-theme', t);
  $('#tema').innerHTML = icon(t === 'dark' ? 'sun' : 'moon');
  $('#tema').setAttribute('aria-label', t === 'dark' ? 'Mudar para o tema claro' : 'Mudar para o tema escuro');
  $('meta[name=theme-color]').setAttribute('content', t === 'light' ? '#EEF1F8' : '#0B0D14');
}
const alternarTema = () => { const t = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark'; gravar('tema', t); aplicarTema(t); };
aplicarTema(document.documentElement.getAttribute('data-theme') || 'dark');
$('#tema').onclick = alternarTema;

// ---------- novo lançamento (folha de escolha) ----------
function abrirNovo() {
  const m = modal('Novo lançamento', `<div class="atalhos" style="grid-template-columns:1fr">
    <button class="atalho" type="button" data-novo="cliente">${icon('usuarios')}<span><strong>Novo cliente</strong><small>Código, dados, projeto e serviços contratados</small></span></button>
    <button class="atalho" type="button" data-novo="receita">${icon('receitas')}<span><strong>Nova receita</strong><small>Entrada de dinheiro: projeto, consultoria, gestão de obra…</small></span></button>
    <button class="atalho" type="button" data-novo="despesa">${icon('despesas')}<span><strong>Nova despesa</strong><small>Saída de dinheiro: terceiros, aluguel, impostos…</small></span></button>
    <button class="atalho" type="button" data-novo="transferencia">${icon('transfer')}<span><strong>Nova transferência</strong><small>Entre as suas contas, sem afetar o resultado</small></span></button></div>`,
  { rodape: '<button type="button" class="btn btn-ghost" data-fechar>Cancelar</button>', pequeno: true });
  m.dlg.addEventListener('click', (e) => {
    const b = e.target.closest('[data-novo]');
    if (!b) return;
    m.fechar();
    (b.dataset.novo === 'cliente' ? formCliente(renderAgora) : b.dataset.novo === 'transferencia' ? formTransferencia(renderAgora) : formLancamento(b.dataset.novo, renderAgora)).catch((err) => toast(err.message, 'erro'));
  });
}

// ---------- cabeçalho, migalhas e botão principal ----------
const cta = $('#cta');
const desktop = window.matchMedia('(min-width: 769px)');
function posicionarCta() {
  const slot = $('#cta-slot');
  if (desktop.matches && slot) slot.append(cta); else document.body.append(cta);
}
desktop.addEventListener('change', posicionarCta);
function pintarCrumbs(lista) {
  const el = $('.crumbs');
  if (!el) return;
  el.innerHTML = lista.map((t, i) => {
    const ultimo = i === lista.length - 1;
    const conteudo = ultimo ? `<span aria-current="page">${esc(t)}</span>` : i === 1 ? `<a href="#/${chaveAtual}">${esc(t)}</a>` : `<span>${esc(t)}</span>`;
    return `${i ? '<span class="sep" aria-hidden="true">/</span>' : ''}${conteudo}`;
  }).join('');
}
window.addEventListener('crumbs', (e) => pintarCrumbs(e.detail));

function rota() {
  const [caminho, query = ''] = (location.hash.slice(2) || 'resumo').split('?');
  return { chave: ROTAS[caminho] ? caminho : 'resumo', query: Object.fromEntries(new URLSearchParams(query)) };
}
let seq = 0, ultimaChave = null;
async function render() {
  const { chave, query } = rota();
  const R = ROTAS[chave];
  chaveAtual = chave;
  document.body.classList.remove('login');
  document.title = `${R.titulo} · Financeiro Cariati`;
  marcarAtivo();
  fecharGaveta();
  const raiz = $('#conteudo');
  if (chave !== ultimaChave || !$('#vista')) {
    ultimaChave = chave;
    raiz.innerHTML = `<div class="page-head"><div><nav class="crumbs" aria-label="Você está aqui"></nav><h1 id="titulo">${esc(R.titulo)}</h1><p class="page-sub">${esc(R.sub)}</p></div><div class="page-cta" id="cta-slot"></div></div><div id="vista" aria-live="polite"></div>`;
    pintarCrumbs([R.grupo, R.titulo]);
    window.scrollTo({ top: 0 });
  }
  cta.hidden = !R.cta;
  if (R.cta) {
    cta.className = `btn cta ${R.cta.violeta ? 'btn-violet violeta' : 'btn-primary'}`;
    cta.innerHTML = `${icon(R.cta.icone)}<span>${esc(R.cta.rotulo)}</span>`;
    cta.onclick = () => { Promise.resolve(R.cta.acao()).catch((e) => toast(e.message, 'erro')); };
  }
  posicionarCta();
  const el = $('#vista');
  const minha = ++seq;
  try {
    await R.fn(el, query);
  } catch (e) {
    if (minha !== seq) return;
    el.innerHTML = estadoErro({ texto: e.message });
    $('#estado-refazer')?.addEventListener('click', () => render());
  }
}
const renderAgora = () => render();
window.addEventListener('hashchange', render);
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') fecharGaveta(); });

// ---------- busca global (Ctrl/Cmd + K) ----------
const norm = (s) => String(s).normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase();
function abrirPaleta() {
  if ($('dialog.paleta')) return;
  const dlg = document.createElement('dialog');
  dlg.className = 'paleta';
  dlg.innerHTML = `<div class="folha"><div class="paleta-campo">${icon('search')}<input id="paleta-in" type="text" placeholder="Buscar telas, ações ou lançamentos…" aria-label="Buscar" autocomplete="off" role="combobox" aria-expanded="true" aria-controls="paleta-lista"><kbd>Esc</kbd></div>
    <div class="paleta-lista" id="paleta-lista" role="listbox" aria-label="Resultados"></div><div class="paleta-rodape"><span>↑↓ navegar</span><span>↵ abrir</span><span>Esc fechar</span></div></div>`;
  document.body.append(dlg);
  const fechar = () => { dlg.close(); dlg.remove(); };
  dlg.addEventListener('cancel', (e) => { e.preventDefault(); fechar(); });
  dlg.addEventListener('click', (e) => { if (e.target === dlg) fechar(); });
  dlg.showModal();
  const entrada = $('#paleta-in', dlg), lista = $('#paleta-lista', dlg);
  const acoes = [
    { grupo: 'Ações', icone: 'usuarios', rot: 'Novo cliente', exec: () => formCliente(renderAgora) },
    { grupo: 'Ações', icone: 'receitas', rot: 'Nova receita', exec: () => formLancamento('receita', renderAgora) },
    { grupo: 'Ações', icone: 'despesas', rot: 'Nova despesa', exec: () => formLancamento('despesa', renderAgora) },
    { grupo: 'Ações', icone: 'transfer', rot: 'Nova transferência', exec: () => formTransferencia(renderAgora) },
    { grupo: 'Ações', icone: 'sun', rot: 'Alternar tema claro/escuro', exec: alternarTema },
  ];
  if (!$('#sair').hidden) acoes.push({ grupo: 'Ações', icone: 'sair', rot: 'Sair', exec: () => $('#sair').click() });
  const telas = Object.entries(ROTAS).map(([k, r]) => ({ grupo: 'Telas', icone: r.icone, rot: r.titulo, sub: r.grupo, exec: () => { location.hash = `#/${k}`; } }));
  let achados = [], sel = 0, itens = [], tmr, seqBusca = 0;
  const desenhar = () => {
    const q = norm(entrada.value.trim());
    const base = [...telas, ...acoes].filter((i) => !q || norm(`${i.rot} ${i.sub || ''}`).includes(q));
    itens = [...achados, ...base];
    sel = Math.min(sel, Math.max(itens.length - 1, 0));
    let ultimo = '';
    lista.innerHTML = itens.length ? itens.map((i, n) => {
      const cab = i.grupo !== ultimo ? `<div class="paleta-grupo" role="presentation">${i.grupo}</div>` : '';
      ultimo = i.grupo;
      return `${cab}<button class="paleta-item" role="option" id="po-${n}" data-n="${n}" aria-selected="${n === sel}" type="button">${icon(i.icone)}<span style="flex:1;min-width:0"><span>${esc(i.rot)}</span>${i.sub ? `<small>${esc(i.sub)}</small>` : ''}</span></button>`;
    }).join('') : `<div class="estado" style="padding:var(--s6)"><p>Nada encontrado para “${esc(entrada.value)}”.</p></div>`;
    entrada.setAttribute('aria-activedescendant', itens.length ? `po-${sel}` : '');
  };
  const abrir = (n) => { const i = itens[n]; if (!i) return; fechar(); i.exec(); };
  entrada.addEventListener('input', () => {
    sel = 0; achados = []; desenhar();
    clearTimeout(tmr);
    const q = entrada.value.trim();
    if (q.length < 2) return;
    const minha = ++seqBusca;
    tmr = setTimeout(async () => {
      try {
        const r = await api(`parcelas?${new URLSearchParams({ busca: q, de: '2000-01-01', ate: '2100-12-31', pageSize: 6 })}`);
        if (minha !== seqBusca) return;
        achados = r.itens.map((p) => ({ grupo: 'Lançamentos', icone: p.tipo === 'receita' ? 'receitas' : 'despesas', rot: `${p.nome} · ${brl(p.valor_cents)}`, sub: `${p.tipo === 'receita' ? 'Receita' : 'Despesa'} · vence ${dataBR(p.vencimento)}${p.pessoa_nome ? ' · ' + p.pessoa_nome : ''}`,
          exec: () => { const ano = p.vencimento.slice(0, 4); setPeriodo({ de: `${ano}-01-01`, ate: `${ano}-12-31` }); location.hash = `#/${p.tipo === 'receita' ? 'receitas' : 'despesas'}?status=&busca=${encodeURIComponent(p.lancamento_nome)}`; } }));
        desenhar();
      } catch { /* sem resultados ao vivo */ }
    }, 250);
  });
  entrada.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown') { e.preventDefault(); sel = Math.min(sel + 1, itens.length - 1); desenhar(); $(`#po-${sel}`, dlg)?.scrollIntoView({ block: 'nearest' }); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); sel = Math.max(sel - 1, 0); desenhar(); $(`#po-${sel}`, dlg)?.scrollIntoView({ block: 'nearest' }); }
    else if (e.key === 'Enter') { e.preventDefault(); abrir(sel); }
  });
  lista.addEventListener('click', (e) => { const b = e.target.closest('[data-n]'); if (b) abrir(Number(b.dataset.n)); });
  desenhar();
  entrada.focus();
}
$('#busca-abrir').onclick = abrirPaleta;
document.addEventListener('keydown', (e) => { if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); if (!document.body.classList.contains('login')) abrirPaleta(); } });

// ---------- fundo com leve paralaxe ----------
if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches && document.documentElement.getAttribute('data-fx') !== 'low') {
  let rx = 0, sy = 0, pend = false;
  const aplicar = () => { pend = false; document.documentElement.style.setProperty('--px', `${rx.toFixed(1)}px`); document.documentElement.style.setProperty('--py', `${(-sy * 0.05).toFixed(1)}px`); };
  const agendar = () => { if (!pend) { pend = true; requestAnimationFrame(aplicar); } };
  window.addEventListener('mousemove', (e) => { rx = (e.clientX / window.innerWidth - 0.5) * 36; agendar(); }, { passive: true });
  window.addEventListener('scroll', () => { sy = window.scrollY; agendar(); }, { passive: true });
}

// ---------- acesso por senha ----------
function telaLogin() {
  document.body.classList.add('login');
  cta.hidden = true;
  ultimaChave = null;
  $('#conteudo').innerHTML = `<form class="login-card glass glass-forte mira reveal" id="login-form">
    <img class="login-logo" src="logo-cariati.png" alt="Cariati Arquitetura &amp; Gestão"><h2>Financeiro interno</h2><p class="suave">Acesso restrito. Entre com seu e-mail e senha.</p>
    <label class="f" style="text-align:left">E-mail<input class="campo" type="email" name="email" autocomplete="username" placeholder="seu@email.com.br"><span class="dica">Administrador: deixe o e-mail em branco e use a senha de administrador.</span></label>
    <label class="f" style="text-align:left">Senha<input class="campo" type="password" name="senha" autocomplete="current-password" required></label>
    <div class="erro" role="alert" style="justify-content:center"></div><button class="btn btn-primary" type="submit" style="width:100%">Entrar</button></form>`;
  $('#login-form').email.value = ler('ultimoEmail', '');
  $((ler('ultimoEmail', '') ? '[name=senha]' : '[name=email]'), $('#login-form')).focus();
  $('#login-form').onsubmit = async (e) => {
    e.preventDefault();
    const botao = $('button', e.target);
    botao.disabled = true;
    try {
      await api('login', { method: 'POST', body: { email: e.target.email.value.trim(), senha: e.target.senha.value } });
      gravar('ultimoEmail', e.target.email.value.trim());
      document.body.classList.remove('login');
      render();
    } catch (err) {
      $('.erro', e.target).innerHTML = `${icon('alert')}<span>${esc(err.message)}</span>`;
      botao.disabled = false;
      e.target.senha.select();
    }
  };
}
window.addEventListener('sem-sessao', () => { $$('dialog').forEach((d) => d.remove()); telaLogin(); });

montarNav();
(async () => {
  try {
    const s = await api('sessao');
    $('#sair').hidden = !s.exige_senha;
    if (!s.autenticado) return telaLogin();
  } catch (e) {
    $('#conteudo').innerHTML = estadoErro({ titulo: 'Não foi possível conectar ao sistema', texto: e.message });
    $('#estado-refazer')?.addEventListener('click', () => location.reload());
    return;
  }
  render();
})();

// PWA: guarda só os arquivos estáticos (a API nunca é guardada)
if ('serviceWorker' in navigator) window.addEventListener('load', () => { navigator.serviceWorker.register('/sw.js').catch(() => {}); });
