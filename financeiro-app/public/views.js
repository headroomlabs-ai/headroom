import { $, $$, MESES, api, brl, confirmar, controlePeriodo, dataBR, esc, excluirComDesfazer, getPeriodo, ligarPeriodo, qs, rotuloMes, toast, urlApi } from './util.js';
import { CORES, animarContadores, chipStatus, estadoVazio, gauge, grafico, icon, kpi, linha, revelar, rosca, skeletonPagina, spark } from './ui.js';
import { CADASTROS, GRUPOS, formBaixa, formCadastro, formEditarParcela, formLancamento, formTransferencia, limparCache } from './forms.js';

export const setCrumbs = (lista) => window.dispatchEvent(new CustomEvent('crumbs', { detail: lista }));
const pronto = (el) => { revelar(el); animarContadores(el); };
const sv = (c) => `<span class="num ${c < 0 ? 'vermelho' : 'verde'}">${brl(c)}</span>`;
const pct = (a, b) => (b > 0 ? (a / b) * 100 : 0);
const fmtPct = (v) => `${v.toFixed(1).replace('.', ',')}%`;
const tabs = (lista, ativa) => `<div class="tabs" role="tablist">${lista.map(([k, r]) => `<button class="tab${k === ativa ? ' ativa' : ''}" data-tab="${k}" role="tab" aria-selected="${k === ativa}">${esc(r)}</button>`).join('')}</div>`;
const anoAtual = new Date().getFullYear();
const seletorAno = (ano) => `<div class="pilula-vidro">${icon('calendario')}<select class="campo" name="ano" aria-label="Ano">${[anoAtual - 2, anoAtual - 1, anoAtual, anoAtual + 1].map((a) => `<option${a === ano ? ' selected' : ''}>${a}</option>`).join('')}</select></div>`;
const iniciais = (n) => (String(n || '?').trim().split(/\s+/).slice(0, 2).map((x) => x[0]).join('') || '?').toUpperCase();
const orbDe = (n, cls = '') => `<span class="orb ${cls}" style="width:44px;height:44px"><b style="font-size:.9rem">${esc(iniciais(n))}</b></span>`;

// ---------- Resumo ----------
export async function resumo(el) {
  const p = getPeriodo();
  el.innerHTML = skeletonPagina();
  const [r, res, desp, pagos] = await Promise.all([
    api(`resumo?${qs(p)}`),
    api(`relatorios/resultados?${qs(p)}`).catch(() => null),
    api(`relatorios/outros?${qs({ ...p, agrupar: 'categoria', tipo: 'despesa', campo: 'vencimento' })}`).catch(() => null),
    api(`parcelas?${qs({ ...p, status: 'pago', pageSize: 100, ordem: 'desc' })}`).catch(() => null),
  ]);
  const totalRec = r.receitas.realizado.valor + r.receitas.em_aberto.valor + r.receitas.vencido.valor;
  const perc = pct(r.receitas.realizado.valor, totalRec);
  const recentes = (pagos?.itens || []).filter((i) => i.data_pagamento).sort((a, b) => b.data_pagamento.localeCompare(a.data_pagamento)).slice(0, 6);
  const CORES_ROSCA = ['#FF9F1C', '#8B5CF6', '#22D3EE', '#D946EF', '#34D399', '#FF6B35'];
  const top = (desp?.itens || []).slice(0, 5);
  const outras = (desp?.itens || []).slice(5).reduce((s, i) => s + i.valor_cents, 0);
  const partes = [...top.map((i, k) => ({ rotulo: i.nome, valor: i.valor_cents, cor: CORES_ROSCA[k] })), ...(outras ? [{ rotulo: 'Outras', valor: outras, cor: '#6F7890' }] : [])];
  const prox = (lista, tipo) => lista.length ? `<ul class="lista">${lista.map((i) => `<li><span class="esq">${icon(i.status === 'vencido' ? 'alert' : 'relogio')}<span>${esc(i.nome)}<small>${esc(i.pessoa || '')}</small></span></span>
    <span class="dir"><b class="num">${brl(i.valor_cents)}</b><small class="${i.status === 'vencido' ? 'vermelho' : ''}">${dataBR(i.vencimento)}${i.status === 'vencido' ? ' · vencido' : ''}</small></span></li>`).join('')}</ul>`
    : `<p class="suave" style="padding:var(--s4) 0">Nenhuma ${tipo} em aberto no período. Tudo em dia.</p>`;
  const serie = res?.grafico?.map((m) => m.valor) || [];
  el.innerHTML = `
    <div class="filtros">${controlePeriodo(p)}</div>
    <div class="grade g-resumo">
      <section class="glass painel hero mira reveal" aria-label="Resultado do período">
        <div class="destaque">
          ${gauge({ pct: perc, rotulo: 'Resultado', valor: `<span data-count="${r.balanco_cents}" data-fmt="brl">${brl(r.balanco_cents)}</span>`, sub: `${fmtPct(perc)} das receitas recebidas`, cor: r.balanco_cents < 0 ? 'red' : 'amber', ariaLabel: `Resultado do período ${brl(r.balanco_cents)}, ${fmtPct(perc)} das receitas já recebidas` })}
          <div class="empilha">
            <div><h3>Recebido − pago</h3><p class="suave" style="margin-top:6px">Quanto já entrou no caixa menos o que já saiu no período escolhido.</p></div>
            ${spark(serie, r.balanco_cents < 0 ? 'var(--err)' : 'var(--amber)')}
            <div class="legenda"><span><i style="background:var(--ok)"></i>Recebido ${brl(r.receitas.realizado.valor)}</span><span><i style="background:var(--err)"></i>Pago ${brl(r.despesas.realizado.valor)}</span></div>
            <div class="progresso ok" aria-hidden="true"><i style="width:${perc}%"></i></div>
          </div>
        </div>
      </section>
      <section class="glass painel reveal" aria-label="Contas bancárias">
        <h3>Contas <button class="btn btn-ghost btn-sm" data-ir="contas">Extratos ${icon('dir')}</button></h3>
        <div class="suave" style="font-size:.78rem">Saldo atual</div>
        <div class="kpi-val ${r.saldo_total_cents < 0 ? 'vermelho' : ''}" style="font-size:clamp(1.6rem,1.2rem+1.5vw,2.2rem);margin:2px 0 var(--s3)" data-count="${r.saldo_total_cents}" data-fmt="brl">${brl(r.saldo_total_cents)}</div>
        <ul class="lista">${r.contas.map((c) => `<li><span class="esq">${orbDe(c.nome, 'cyan')}<span>${esc(c.nome)}<small>${esc(c.banco || '')}</small></span></span>${sv(c.saldo_cents)}</li>`).join('') || '<li class="suave">Nenhuma conta ainda. Crie em Cadastros → Contas bancárias.</li>'}</ul>
      </section>
    </div>
    <div class="grade-kpi tres" style="margin-top:var(--s4)">
      ${kpi({ rotulo: 'Receitas em aberto', icone: 'receitas', valor: r.receitas.em_aberto.valor, qtd: r.receitas.em_aberto.qtd, cor: 'laranja', destino: 'receitas?status=em_aberto' })}
      ${kpi({ rotulo: 'Receitas vencidas', icone: 'alert', valor: r.receitas.vencido.valor, qtd: r.receitas.vencido.qtd, cor: r.receitas.vencido.qtd ? 'vermelho' : '', destino: 'receitas?status=vencido' })}
      ${kpi({ rotulo: 'Recebido', icone: 'check', valor: r.receitas.realizado.valor, qtd: r.receitas.realizado.qtd, cor: 'verde', destino: 'receitas?status=pago' })}
      ${kpi({ rotulo: 'Despesas em aberto', icone: 'despesas', valor: r.despesas.em_aberto.valor, qtd: r.despesas.em_aberto.qtd, cor: 'laranja', destino: 'despesas?status=em_aberto' })}
      ${kpi({ rotulo: 'Despesas vencidas', icone: 'alert', valor: r.despesas.vencido.valor, qtd: r.despesas.vencido.qtd, cor: r.despesas.vencido.qtd ? 'vermelho' : '', destino: 'despesas?status=vencido' })}
      ${kpi({ rotulo: 'Pago', icone: 'check', valor: r.despesas.realizado.valor, qtd: r.despesas.realizado.qtd, cor: 'verde', destino: 'despesas?status=pago' })}
    </div>
    <div class="grade g2" style="margin-top:var(--s4)">
      <section class="glass painel reveal"><h3>Atalhos</h3>
        <div class="atalhos">
          <button class="atalho" data-novo="receita">${icon('receitas')}<span><strong>Nova receita</strong><small>Registrar entrada</small></span></button>
          <button class="atalho" data-novo="despesa">${icon('despesas')}<span><strong>Nova despesa</strong><small>Registrar saída</small></span></button>
          <button class="atalho" data-novo="transferencia">${icon('transfer')}<span><strong>Transferir</strong><small>Entre contas</small></span></button>
          <button class="atalho" data-ir="despesas?status=vencido">${icon('alert')}<span><strong>Contas vencidas</strong><small>${r.despesas.vencido.qtd + r.receitas.vencido.qtd} para revisar</small></span></button>
        </div>
        <hr class="divisor"><h3>Atividade recente</h3>
        ${recentes.length ? `<ul class="lista">${recentes.map((i) => `<li><span class="esq">${icon(i.tipo === 'receita' ? 'receitas' : 'despesas')}<span>${esc(i.nome)}<small>${esc(i.pessoa_nome || i.categoria_nome || '')}</small></span></span>
          <span class="dir"><b class="num ${i.tipo === 'receita' ? 'verde' : 'vermelho'}">${i.tipo === 'receita' ? '+' : '−'} ${brl(i.valor_pago_cents)}</b><small>${dataBR(i.data_pagamento)}</small></span></li>`).join('')}</ul>`
        : '<p class="suave" style="padding:var(--s3) 0">Os últimos recebimentos e pagamentos aparecem aqui.</p>'}
      </section>
      <section class="glass painel reveal"><h3>Despesas por categoria</h3>
        ${partes.length ? rosca(partes, { centro: 'Despesas' }) : '<p class="suave" style="padding:var(--s4) 0">Sem despesas no período.</p>'}
      </section>
    </div>
    <div class="grade g2" style="margin-top:var(--s4)">
      <section class="glass painel reveal"><h3>Próximas receitas <button class="btn btn-ghost btn-sm" data-ir="receitas">Ver todas ${icon('dir')}</button></h3>${prox(r.proximas_receitas, 'receita')}</section>
      <section class="glass painel reveal"><h3>Próximas despesas <button class="btn btn-ghost btn-sm" data-ir="despesas">Ver todas ${icon('dir')}</button></h3>${prox(r.proximas_despesas, 'despesa')}</section>
    </div>`;
  ligarPeriodo(el, () => resumo(el));
  el.onclick = (e) => {
    const b = e.target.closest('[data-ir],[data-novo]');
    if (!b) return;
    if (b.dataset.ir) location.hash = `#/${b.dataset.ir}`;
    else if (b.dataset.novo === 'transferencia') formTransferencia(() => resumo(el)); else formLancamento(b.dataset.novo, () => resumo(el));
  };
  pronto(el);
}

// ---------- Receitas e Despesas ----------
const estadoLista = { receita: { status: '', busca: '', page: 1, ord: null }, despesa: { status: '', busca: '', page: 1, ord: null } };

export async function lista(el, tipo, query = {}) {
  const e = estadoLista[tipo];
  if (query.status !== undefined) { e.status = query.status; e.page = 1; }
  if (query.busca !== undefined) { e.busca = query.busca; e.page = 1; }
  const p = getPeriodo();
  const rec = tipo === 'receita';
  if (!el.dataset.pronto || el.dataset.tela !== tipo) el.innerHTML = skeletonPagina();
  const r = await api(`parcelas?${qs({ tipo, ...p, status: e.status, busca: e.busca, page: e.page, pageSize: 25 })}`);
  el.dataset.pronto = '1'; el.dataset.tela = tipo;
  const f = r.faixas;
  const perc = pct(f.pagos.valor, f.total.valor);
  const fx = [['vencidos', 'Vencidos', 'vencido', 'vermelho', 'alert'], ['vence_hoje', 'Vence hoje', 'vence_hoje', 'laranja', 'relogio'], ['a_vencer', 'A vencer', 'a_vencer', 'laranja', 'calendario'],
    ['pagos', rec ? 'Recebidos' : 'Pagos', 'pago', 'verde', 'check'], ['total', 'Total do período', '', 'azul', 'camadas']];
  const paginas = Math.ceil(r.total_itens / r.pageSize);
  let itens = [...r.itens];
  if (e.ord) {
    const { k, dir } = e.ord;
    itens.sort((a, b) => (k === 'valor' ? a.valor_cents - b.valor_cents : String(a[k] ?? '').localeCompare(String(b[k] ?? ''))) * dir);
  }
  const th = (k, rot, cls = '') => `<th class="ord ${cls}" data-ord="${k}" ${e.ord?.k === k ? `aria-sort="${e.ord.dir > 0 ? 'ascending' : 'descending'}"` : ''} tabindex="0" role="columnheader">${rot}<span class="seta">${e.ord?.k === k ? (e.ord.dir > 0 ? '▲' : '▼') : '↕'}</span></th>`;
  const linhas = itens.map((i) => `<tr data-lanc="${i.lancamento_id}">
    <td class="nome" data-label="Nome"><strong>${esc(i.nome)}</strong>${i.recorrente ? ` <span class="chip" title="Recorrente">${icon('repetir')}Mensal</span>` : ''}<small>${(i.etiquetas || '').split(',').filter(Boolean).map((t) => `<span class="chip tag">${esc(t.trim())}</span>`).join(' ')}</small></td>
    <td class="num ${rec ? 'verde' : 'vermelho'}" data-label="Valor"><b>${brl(i.valor_cents)}</b>${i.valor_pago_cents && i.status !== 'pago' ? `<small class="suave" style="display:block">pago ${brl(i.valor_pago_cents)}</small>` : ''}</td>
    <td data-label="Vencimento">${dataBR(i.vencimento)}</td><td data-label="${rec ? 'Recebido em' : 'Pago em'}">${dataBR(i.data_pagamento)}</td>
    <td data-label="${rec ? 'Cliente' : 'Fornecedor'}">${esc(i.pessoa_nome || '-')}</td><td data-label="Categoria">${esc(i.categoria_nome || '-')}</td>
    <td data-label="Projeto">${esc(i.contrato_codigo || '-')}</td><td data-label="Status">${chipStatus(i.status)}</td>
    <td class="acoes" data-label=""><div class="acoes-linha">${i.status !== 'pago' ? `<button class="btn btn-sm btn-ok" data-acao="baixar" data-id="${i.id}">${icon('check')}${rec ? 'Receber' : 'Pagar'}</button>` : `<button class="btn btn-sm" data-acao="estornar" data-id="${i.id}">${icon('desfazer')}Estornar</button>`}
      ${i.valor_pago_cents === 0 ? `<button class="btn btn-sm" data-acao="editar" data-id="${i.id}" aria-label="Editar parcela">${icon('editar')}</button>` : ''}
      <button class="btn btn-sm btn-danger" data-acao="excluir" data-id="${i.id}" data-lanc="${i.lancamento_id}" aria-label="Excluir lançamento inteiro" title="Exclui o lançamento inteiro, com todas as parcelas">${icon('lixo')}</button></div></td></tr>`).join('');
  const filtrado = e.status || e.busca;
  el.innerHTML = `
    <div class="filtros">${controlePeriodo(p)}
      <label class="pilula-vidro busca-campo">${icon('search')}<input class="campo" type="search" name="busca" placeholder="Nome, cliente ou nota fiscal" value="${esc(e.busca)}" aria-label="Pesquisar lançamentos"></label>
      <span class="espaco"></span><a class="btn btn-ghost" href="${urlApi(`parcelas.csv?${qs({ tipo, ...p, status: e.status, busca: e.busca })}`)}">${icon('baixar')}Exportar planilha</a></div>
    <div class="grade g-destaque">
      <section class="glass painel hero mira reveal" aria-label="${rec ? 'Recebido' : 'Pago'} no período" style="display:grid;place-items:center">
        ${gauge({ pct: perc, rotulo: rec ? 'Recebido' : 'Pago', valor: `<span data-count="${perc}" data-fmt="pct">${fmtPct(perc)}</span>`, sub: `${brl(f.pagos.valor)} de ${brl(f.total.valor)}`, cor: rec ? 'amber' : 'violet', tamanho: 220, ariaLabel: `${fmtPct(perc)} ${rec ? 'recebido' : 'pago'} do total do período` })}</section>
      <div class="grade-kpi" style="align-content:center" role="group" aria-label="Filtrar por situação">${fx.map(([k, rot, st, cor, ic]) => kpi({ rotulo: rot, icone: ic, valor: f[k].valor, qtd: f[k].qtd, cor, status: st, ativo: e.status === st })).join('')}</div>
    </div>
    <section class="glass painel reveal" style="margin-top:var(--s4)"><h3>${rec ? 'Receitas' : 'Despesas'} por mês de vencimento</h3>
      ${grafico(r.grafico.map((m) => ({ rotulo: rotuloMes(m.mes), partes: [{ valor: m.pago, cor: CORES.pago }, { valor: m.atrasado, cor: CORES.atrasado }, { valor: m.previsto, cor: CORES.previsto }] })))}
      <div class="legenda"><span><i style="background:var(--ok)"></i>${rec ? 'Recebido' : 'Pago'}</span><span><i style="background:var(--err)"></i>Atrasado</span><span><i style="background:var(--amber)"></i>Previsto</span></div></section>
    ${itens.length ? `<section class="glass reveal" style="margin-top:var(--s4)"><div class="tabela-wrap"><table class="tbl"><thead><tr>${th('nome', 'Nome')}${th('valor', 'Valor', 'num')}${th('vencimento', 'Vencimento')}<th>${rec ? 'Recebido em' : 'Pago em'}</th><th>${rec ? 'Cliente' : 'Fornecedor'}</th><th>Categoria</th><th>Projeto</th><th>Status</th><th></th></tr></thead><tbody>${linhas}</tbody></table></div></section>`
    : filtrado ? estadoVazio({ titulo: 'Nada encontrado com estes filtros', texto: 'Tente outro período, limpe a busca ou escolha "Total do período".', acaoRotulo: 'Limpar filtros', acaoId: 'limpar-filtros' })
      : estadoVazio({ titulo: rec ? 'Nenhuma receita neste período' : 'Nenhuma despesa neste período', texto: rec ? 'Registre a primeira entrada para ver o resumo, o gráfico e a DRE ganharem vida.' : 'Registre a primeira saída para acompanhar custos e resultado.', acaoRotulo: rec ? 'Nova receita' : 'Nova despesa', acaoId: 'estado-novo' })}
    ${paginas > 1 ? `<div class="paginacao">${Array.from({ length: paginas }, (_, i) => `<button class="btn btn-sm${i + 1 === r.page ? ' ativa' : ''}" data-pagina="${i + 1}" aria-label="Página ${i + 1}"${i + 1 === r.page ? ' aria-current="page"' : ''}>${i + 1}</button>`).join('')}</div>` : ''}`;
  const recarregar = () => lista(el, tipo);
  ligarPeriodo(el, () => { e.page = 1; recarregar(); });
  let t;
  $('[name=busca]', el).oninput = (ev) => { clearTimeout(t); t = setTimeout(() => { e.busca = ev.target.value; e.page = 1; recarregar().then(() => { const b = $('[name=busca]', el); b.focus(); b.setSelectionRange(b.value.length, b.value.length); }); }, 350); };
  el.onclick = async (ev) => {
    const th2 = ev.target.closest('th[data-ord]');
    if (th2) { const k = th2.dataset.ord; e.ord = e.ord?.k === k ? { k, dir: -e.ord.dir } : { k, dir: 1 }; return recarregar(); }
    if (ev.target.closest('#estado-novo')) return formLancamento(tipo, recarregar);
    if (ev.target.closest('#limpar-filtros')) { e.status = ''; e.busca = ''; return recarregar(); }
    const k = ev.target.closest('.kpi[data-status]');
    if (k) { e.status = e.status === k.dataset.status ? '' : k.dataset.status; e.page = 1; return recarregar(); }
    const b = ev.target.closest('button');
    if (!b) return;
    if (b.dataset.pagina) { e.page = Number(b.dataset.pagina); return recarregar(); }
    const item = r.itens.find((i) => String(i.id) === b.dataset.id);
    try {
      if (b.dataset.acao === 'baixar') formBaixa(item, recarregar);
      else if (b.dataset.acao === 'editar') formEditarParcela(item, recarregar);
      else if (b.dataset.acao === 'estornar') { await api(`parcelas/${item.id}/estorno`, { method: 'POST' }); toast('Pagamento estornado. Você pode registrar de novo quando quiser.', { tipo: 'info' }); recarregar(); }
      else if (b.dataset.acao === 'excluir' && await confirmar(`Excluir "${item.lancamento_nome}" e todas as suas parcelas?`)) {
        const linhasDoLanc = () => $$(`tr[data-lanc="${b.dataset.lanc}"]`, el);
        excluirComDesfazer({ aviso: 'Lançamento excluído.', ocultar: () => linhasDoLanc().forEach((x) => x.classList.add('linha-oculta')), restaurar: () => linhasDoLanc().forEach((x) => x.classList.remove('linha-oculta')),
          executar: (keepalive) => api(`lancamentos/${b.dataset.lanc}`, { method: 'DELETE', keepalive }), depois: recarregar });
      }
    } catch (err) { toast(err.message, 'erro'); }
  };
  pronto(el);
}

// ---------- Pagamentos do cliente ----------
export async function pagamentos(el) {
  const p = getPeriodo();
  const filtro = pagamentos.filtro ||= { busca: '', status: '' };
  if (!el.dataset.pronto || el.dataset.tela !== 'pag') el.innerHTML = skeletonPagina();
  const r = await api(`pagamentos-cliente?${qs({ ...p, ...filtro })}`);
  el.dataset.pronto = '1'; el.dataset.tela = 'pag';
  const tot = r.reduce((s, x) => ({ t: s.t + x.total_cents, p: s.p + x.pago_cents, a: s.a + x.aberto_cents }), { t: 0, p: 0, a: 0 });
  const perc = pct(tot.p, tot.t);
  const linhas = r.map((x) => `<tr data-lanc="${x.id}"><td class="nome" data-label="Pagamento"><button class="toggle" data-exp="${x.id}" aria-expanded="false" aria-label="Mostrar parcelas de ${esc(x.nome)}">${icon('dir')}</button><strong>${esc(x.nome)}</strong><small>${esc(x.cliente || '-')}</small></td>
    <td data-label="Projeto">${esc(x.contrato_codigo)}</td><td data-label="Competência">${dataBR(x.competencia)}</td><td class="num" data-label="Valor total"><b>${brl(x.total_cents)}</b></td><td class="num verde" data-label="Valor pago">${brl(x.pago_cents)}</td><td class="num laranja" data-label="Em aberto">${brl(x.aberto_cents)}</td>
    <td data-label="Status">${chipStatus(x.status)}</td></tr>
    <tr class="tbl-det" hidden data-det="${x.id}"><td colspan="7"><table>${x.parcelas.map((q) => `<tr><td>${esc(q.nome)}</td><td>${dataBR(q.vencimento)}</td><td class="num">${brl(q.valor_cents)}</td><td>${q.data_pagamento ? 'pago em ' + dataBR(q.data_pagamento) : ''}</td><td>${chipStatus(q.status)}</td></tr>`).join('')}</table></td></tr>`).join('');
  el.innerHTML = `
    <div class="filtros">${controlePeriodo(p)}<span class="suave" style="font-size:.8rem">por data de competência</span>
      <select class="campo" name="status" aria-label="Filtrar por status"><option value="">Todos os status</option><option value="em_aberto">Em aberto</option><option value="vencido">Com parcela vencida</option><option value="pago">Pagos</option></select>
      <label class="pilula-vidro busca-campo">${icon('search')}<input class="campo" type="search" name="busca" placeholder="Pesquisar pagamento ou cliente" value="${esc(filtro.busca)}" aria-label="Pesquisar"></label></div>
    <div class="grade g-destaque">
      <section class="glass painel hero mira reveal" style="display:grid;place-items:center">${gauge({ pct: perc, rotulo: 'Recebido', valor: `<span data-count="${perc}" data-fmt="pct">${fmtPct(perc)}</span>`, sub: `${brl(tot.p)} de ${brl(tot.t)}`, cor: 'amber', ariaLabel: `${fmtPct(perc)} do total contratado já recebido` })}</section>
      <div class="grade-kpi" style="align-content:center">${kpi({ rotulo: 'Total contratado', icone: 'contrato', valor: tot.t, qtd: r.length, cor: 'azul' })}${kpi({ rotulo: 'Recebido', icone: 'check', valor: tot.p, cor: 'verde' })}${kpi({ rotulo: 'Em aberto', icone: 'relogio', valor: tot.a, cor: 'laranja' })}</div>
    </div>
    ${r.length ? `<section class="glass reveal" style="margin-top:var(--s4)"><div class="tabela-wrap"><table class="tbl"><thead><tr><th>Pagamento</th><th>Projeto</th><th>Competência</th><th class="num">Valor total</th><th class="num">Valor pago</th><th class="num">Em aberto</th><th>Status</th></tr></thead><tbody>${linhas}</tbody></table></div></section>`
    : estadoVazio({ titulo: 'Nenhum pagamento de contrato neste período', texto: 'Crie uma receita ligada a um projeto para acompanhar parcelas, valor pago e o que falta receber.', acaoRotulo: 'Nova receita de contrato', acaoId: 'estado-novo' })}`;
  $('[name=status]', el).value = filtro.status;
  ligarPeriodo(el, () => pagamentos(el));
  $('[name=status]', el).onchange = (e) => { filtro.status = e.target.value; pagamentos(el); };
  let t; $('[name=busca]', el).oninput = (e) => { clearTimeout(t); t = setTimeout(() => { filtro.busca = e.target.value; pagamentos(el).then(() => { const b = $('[name=busca]', el); b.focus(); b.setSelectionRange(b.value.length, b.value.length); }); }, 350); };
  el.onclick = (e) => {
    if (e.target.closest('#estado-novo')) return formLancamento('receita', () => pagamentos(el));
    const b = e.target.closest('.toggle');
    if (!b) return;
    const det = $(`[data-det="${b.dataset.exp}"]`, el);
    det.hidden = !det.hidden; b.setAttribute('aria-expanded', String(!det.hidden));
  };
  pronto(el);
}

// ---------- Transferências ----------
export async function transferencias(el) {
  const p = getPeriodo();
  el.innerHTML = skeletonPagina();
  const r = await api(`transferencias?${qs(p)}`);
  const total = r.reduce((s, t) => s + t.valor_cents, 0);
  el.innerHTML = `<div class="filtros">${controlePeriodo(p)}</div>
    <div class="grade g2">
      <section class="glass painel reveal" style="display:flex;align-items:center;gap:var(--s4)"><span class="orb">${icon('transfer')}</span><div><h3>Transferências no período</h3><div class="kpi-val" data-count="${r.length}" data-fmt="int">${r.length}</div></div></section>
      <section class="glass painel reveal" style="display:flex;align-items:center;gap:var(--s4)"><span class="orb cyan">${icon('banco')}</span><div><h3>Total movimentado</h3><div class="kpi-val" data-count="${total}" data-fmt="brl">${brl(total)}</div></div></section>
    </div>
    ${r.length ? `<section class="glass reveal" style="margin-top:var(--s4)"><div class="tabela-wrap"><table class="tbl"><thead><tr><th>Descrição</th><th class="num">Valor</th><th>De</th><th>Para</th><th>Data</th><th></th></tr></thead><tbody>${r.map((t) => `<tr data-tr="${t.id}"><td class="nome" data-label="Descrição"><strong>${esc(t.descricao)}</strong></td><td class="num" data-label="Valor"><b>${brl(t.valor_cents)}</b></td><td data-label="De">${esc(t.origem_nome)}</td><td data-label="Para">${esc(t.destino_nome)}</td><td data-label="Data">${dataBR(t.data)}</td><td class="acoes" data-label=""><div class="acoes-linha"><button class="btn btn-sm btn-danger" data-del="${t.id}" aria-label="Excluir transferência">${icon('lixo')}</button></div></td></tr>`).join('')}</tbody></table></div></section>`
    : estadoVazio({ titulo: 'Nenhuma transferência no período', texto: 'Registre aqui as movimentações internas entre suas contas. Elas mudam o saldo de cada conta, mas não entram no resultado.', acaoRotulo: 'Nova transferência', acaoId: 'estado-novo' })}`;
  ligarPeriodo(el, () => transferencias(el));
  el.onclick = async (e) => {
    if (e.target.closest('#estado-novo')) return formTransferencia(() => transferencias(el));
    const b = e.target.closest('[data-del]');
    if (!b || !await confirmar('Excluir esta transferência?')) return;
    const linhaTr = $(`tr[data-tr="${b.dataset.del}"]`, el);
    excluirComDesfazer({ aviso: 'Transferência excluída.', ocultar: () => linhaTr.classList.add('linha-oculta'), restaurar: () => linhaTr.classList.remove('linha-oculta'),
      executar: (keepalive) => api(`transferencias/${b.dataset.del}`, { method: 'DELETE', keepalive }), depois: () => transferencias(el) });
  };
  pronto(el);
}

// ---------- Contas e extratos ----------
export async function contas(el, query = {}) {
  el.innerHTML = skeletonPagina();
  const lista = await api('contas-saldos');
  const sel = query.conta || lista[0]?.id;
  const ex = sel ? await api(`contas/${sel}/extrato`) : null;
  const total = lista.reduce((s, c) => s + c.saldo_cents, 0);
  const atual = lista.find((c) => c.id === sel);
  setCrumbs(['Financeiro', 'Contas e extratos', ...(atual ? [atual.nome] : [])]);
  const perc = atual && total > 0 ? Math.max(0, pct(atual.saldo_cents, total)) : 0;
  el.innerHTML = `
    ${lista.length ? `<div class="grade g-destaque">
      <section class="glass painel hero mira reveal" style="display:grid;place-items:center">${gauge({ pct: perc, rotulo: atual?.nome || 'Conta', valor: `<span data-count="${atual?.saldo_cents || 0}" data-fmt="brl">${brl(atual?.saldo_cents || 0)}</span>`, sub: `${fmtPct(perc)} do saldo total`, cor: 'violet', ariaLabel: `${atual?.nome}: ${fmtPct(perc)} do saldo total` })}</section>
      <div class="grade-kpi" style="align-content:center">${lista.map((c) => `<button class="kpi reveal${c.id === sel ? ' ativa' : ''}" data-conta="${c.id}" type="button"><span class="kpi-topo"><span class="kpi-rot">${icon('banco')}${esc(c.nome)}</span></span><span class="kpi-val ${c.saldo_cents < 0 ? 'vermelho' : ''}" data-count="${c.saldo_cents}" data-fmt="brl">${brl(c.saldo_cents)}</span><span class="kpi-var suave">${esc(c.banco || '')}</span></button>`).join('')}</div></div>`
    : estadoVazio({ titulo: 'Nenhuma conta cadastrada', texto: 'Cadastre suas contas bancárias com o saldo inicial para ver saldos e extratos.', acaoRotulo: 'Nova conta', acaoId: 'estado-novo' })}
    ${ex ? `<h2 style="margin:var(--s6) 0 var(--s3)">Extrato · ${esc(ex.conta.nome)}</h2>${ex.movimentos.length ? '' : '<p class="suave" style="margin-bottom:var(--s3)">Ainda não há movimentos nesta conta.</p>'}<section class="glass reveal"><div class="tabela-wrap"><table class="tbl"><thead><tr><th>Data</th><th>Descrição</th><th>Pessoa</th><th class="num">Valor</th><th class="num">Saldo</th></tr></thead><tbody>
      <tr><td data-label="Data"></td><td class="nome suave" data-label="Descrição">Saldo inicial</td><td data-label="Pessoa"></td><td class="num" data-label="Valor"></td><td class="num" data-label="Saldo"><b>${brl(ex.saldo_inicial_cents)}</b></td></tr>
      ${[...ex.movimentos].reverse().map((m) => `<tr><td data-label="Data">${dataBR(m.data)}</td><td class="nome" data-label="Descrição"><strong>${esc(m.descricao)}</strong>${m.origem === 'transferencia' ? ` <span class="chip">${icon('transfer')}transferência</span>` : ''}</td><td data-label="Pessoa">${esc(m.pessoa || '')}</td><td class="num" data-label="Valor">${sv(m.valor_cents)}</td><td class="num" data-label="Saldo">${brl(m.saldo_cents)}</td></tr>`).join('')}</tbody></table></div></section>` : ''}`;
  el.onclick = (e) => {
    if (e.target.closest('#estado-novo')) return formCadastro('contas', null, () => { limparCache(); contas(el); });
    const b = e.target.closest('[data-conta]');
    if (b) location.hash = `#/contas?conta=${b.dataset.conta}`;
  };
  pronto(el);
}

// ---------- Fluxo de caixa ----------
const abertos = new Set(['receitas', 'despesas']);
export async function fluxo(el) {
  const ano = fluxo.ano ||= anoAtual;
  el.innerHTML = skeletonPagina();
  const r = await api(`relatorios/fluxo-caixa?ano=${ano}`);
  const cel = (v, cor) => `<td class="num ${v < 0 ? 'vermelho' : cor || ''}">${brl(v)}</td>`;
  const dupla = (d, sinal = 1, cor = '') => d.previsto.map((_, m) => cel(sinal * d.previsto[m], cor) + cel(sinal * d.realizado[m], cor)).join('');
  const lin = (rot, d, { cls = '', sinal = 1, cor = '', toggle } = {}) =>
    `<tr class="${cls}"><td class="fixa">${toggle ? `<button class="toggle" data-t="${toggle}" aria-expanded="${abertos.has(toggle)}" aria-label="Mostrar categorias" style="transform:${abertos.has(toggle) ? 'rotate(90deg)' : 'none'}">${icon('dir')}</button>` : ''}${esc(rot)}</td>${dupla(d, sinal, cor)}</tr>`;
  const resultado = { previsto: r.receitas.total.previsto.map((v, i) => v - r.despesas.total.previsto[i]), realizado: r.receitas.total.realizado.map((v, i) => v - r.despesas.total.realizado[i]) };
  const somaPrev = r.receitas.total.previsto.reduce((a, b) => a + b, 0), somaReal = r.receitas.total.realizado.reduce((a, b) => a + b, 0);
  const perc = pct(somaReal, somaPrev);
  const ultimo = ano === anoAtual ? new Date().getMonth() + 1 : ano < anoAtual ? 12 : 0;
  const rotulos = MESES.map((m) => m.slice(0, 3).toLowerCase());
  const series = [{ nome: 'Saldo previsto', valores: r.saldo_final.previsto, cor: 'var(--violet)' }, ...(ultimo ? [{ nome: 'Saldo realizado', valores: r.saldo_final.realizado.slice(0, ultimo), cor: 'var(--amber)' }] : [])];
  el.innerHTML = `<div class="filtros">${seletorAno(ano)}<span class="suave" style="font-size:.8rem">Previsto = por vencimento · Realizado = por data de pagamento · o saldo final de cada mês é o saldo inicial do seguinte</span></div>
    <div class="grade g-destaque">
      <section class="glass painel hero mira reveal" style="display:grid;place-items:center">${gauge({ pct: perc, rotulo: 'Receitas realizadas', valor: `<span data-count="${perc}" data-fmt="pct">${fmtPct(perc)}</span>`, sub: `${brl(somaReal)} de ${brl(somaPrev)} previstos`, cor: 'amber', ariaLabel: `${fmtPct(perc)} das receitas previstas do ano já realizadas` })}</section>
      <section class="glass painel reveal"><h3>Saldo ao longo do ano</h3>${linha(series, rotulos)}<div class="legenda"><span><i style="background:var(--violet)"></i>Saldo previsto</span><span><i style="background:var(--amber)"></i>Saldo realizado</span></div></section>
    </div>
    <section class="glass reveal" style="margin-top:var(--s4)"><div class="tabela-wrap"><table class="hier"><thead><tr><th class="fixa" rowspan="2">Descrição</th>${MESES.map((m) => `<th colspan="2" class="num" style="text-align:center">${m}</th>`).join('')}</tr>
    <tr>${MESES.map(() => '<th class="num">Previsto</th><th class="num">Realizado</th>').join('')}</tr></thead><tbody>
    ${lin('Saldo inicial', r.saldo_inicial, { cls: 'sub' })}
    ${lin('Total de receitas', r.receitas.total, { cls: 'sub', cor: 'verde', toggle: 'receitas' })}
    ${abertos.has('receitas') ? r.receitas.categorias.map((c) => lin(c.nome, c, { cls: 'cat' })).join('') : ''}
    ${lin('Total de despesas', r.despesas.total, { cls: 'sub', sinal: -1, toggle: 'despesas' })}
    ${abertos.has('despesas') ? r.despesas.categorias.map((c) => lin(c.nome, c, { cls: 'cat', sinal: -1 })).join('') : ''}
    ${lin('Resultado do mês', resultado, { cls: 'sub' })}${lin('Saldo final', r.saldo_final, { cls: 'sub' })}</tbody></table></div></section>`;
  $('[name=ano]', el).onchange = (e) => { fluxo.ano = Number(e.target.value); fluxo(el); };
  el.onclick = (e) => { const b = e.target.closest('[data-t]'); if (b) { abertos.has(b.dataset.t) ? abertos.delete(b.dataset.t) : abertos.add(b.dataset.t); fluxo(el); } };
  pronto(el);
}

// ---------- DRE gerencial ----------
const abertosDre = new Set();
export async function dre(el) {
  const ano = dre.ano ||= anoAtual;
  const regime = dre.regime ||= 'competencia';
  el.innerHTML = skeletonPagina();
  const r = await api(`relatorios/dre?${qs({ ano, regime })}`);
  const L = Object.fromEntries(r.linhas.map((l) => [l.chave, l]));
  const margemAno = pct(L.resultado_operacional.total, L.receita_bruta.total) * (L.resultado_operacional.total < 0 ? -1 : 1);
  const pctc = (v) => (v === null ? '—' : `${v.toFixed(2).replace('.', ',')}%`);
  const celulas = (vals, margem) => vals.map((v) => (margem ? `<td class="num ${v !== null && v < 0 ? 'vermelho' : ''}">${pctc(v)}</td>` : `<td class="num ${v < 0 ? 'vermelho' : v > 0 ? 'verde' : 'suave'}">${brl(v)}</td>`)).join('');
  const linhas = r.linhas.map((l) => {
    if (l.tipo === 'margem') return `<tr class="margem"><td class="fixa">${esc(l.nome)}</td>${celulas(l.valores, true)}<td class="num"></td></tr>`;
    const exp = l.tipo === 'grupo' && l.categorias.length;
    let h = `<tr class="${l.tipo === 'subtotal' ? 'sub' : ''}"><td class="fixa">${exp ? `<button class="toggle" data-g="${l.chave}" aria-expanded="${abertosDre.has(l.chave)}" aria-label="Mostrar categorias" style="transform:${abertosDre.has(l.chave) ? 'rotate(90deg)' : 'none'}">${icon('dir')}</button>` : ''}${esc(l.nome)}</td>${celulas(l.valores)}<td class="num ${l.total < 0 ? 'vermelho' : ''}"><b>${brl(l.total)}</b></td></tr>`;
    if (exp && abertosDre.has(l.chave)) h += l.categorias.map((c) => `<tr class="cat"><td class="fixa">${esc(c.nome)}</td>${celulas(c.valores)}<td class="num">${brl(c.total)}</td></tr>`).join('');
    return h;
  }).join('');
  el.innerHTML = `<div class="filtros">${seletorAno(ano)}<div class="pilula-vidro">${icon('camadas')}<select class="campo" name="regime" aria-label="Regime"><option value="competencia"${regime === 'competencia' ? ' selected' : ''}>Regime de competência</option><option value="caixa"${regime === 'caixa' ? ' selected' : ''}>Regime de caixa</option></select></div>
      <span class="suave" style="font-size:.8rem">Margens calculadas sobre a receita operacional bruta.</span></div>
    <div class="grade g-destaque">
      <section class="glass painel hero mira reveal" style="display:grid;place-items:center">${gauge({ pct: Math.abs(margemAno), rotulo: 'Margem operacional', valor: `<span data-count="${pct(L.resultado_operacional.total, L.receita_bruta.total)}" data-fmt="pct">${fmtPct(pct(L.resultado_operacional.total, L.receita_bruta.total))}</span>`, sub: `Resultado ${brl(L.resultado_operacional.total)}`, cor: L.resultado_operacional.total < 0 ? 'red' : 'amber', ariaLabel: `Margem operacional do ano: ${pctc(pct(L.resultado_operacional.total, L.receita_bruta.total))}` })}</section>
      <div class="grade-kpi" style="align-content:center">${kpi({ rotulo: 'Receita bruta', icone: 'receitas', valor: L.receita_bruta.total, cor: 'verde' })}${kpi({ rotulo: 'Resultado bruto', icone: 'tendencia', valor: L.resultado_bruto.total, cor: L.resultado_bruto.total < 0 ? 'vermelho' : 'azul' })}${kpi({ rotulo: 'Resultado líquido', icone: 'alvo', valor: L.resultado_liquido.total, cor: L.resultado_liquido.total < 0 ? 'vermelho' : 'verde' })}</div>
    </div>
    <section class="glass reveal" style="margin-top:var(--s4)"><div class="tabela-wrap"><table class="hier"><thead><tr><th class="fixa">Demonstração do resultado</th>${MESES.map((m) => `<th class="num">${m}</th>`).join('')}<th class="num">Total</th></tr></thead><tbody>${linhas}</tbody></table></div></section>
    <p class="suave" style="margin-top:var(--s3);font-size:.8rem">Cada categoria pertence a um grupo da DRE (${Object.values(GRUPOS).join('; ')}). Ajuste em Cadastros → Categorias.</p>`;
  $('[name=ano]', el).onchange = (e) => { dre.ano = Number(e.target.value); dre(el); };
  $('[name=regime]', el).onchange = (e) => { dre.regime = e.target.value; dre(el); };
  el.onclick = (e) => { const b = e.target.closest('[data-g]'); if (b) { abertosDre.has(b.dataset.g) ? abertosDre.delete(b.dataset.g) : abertosDre.add(b.dataset.g); dre(el); } };
  pronto(el);
}

// ---------- Resultados ----------
const NOMES_ABA = { gerais: 'Resultados gerais', consolidado: 'Consolidado por projeto', previsto: 'Previsto por projeto' };
export async function resultados(el, query = {}) {
  const aba = resultados.aba = query.aba || resultados.aba || 'gerais';
  const p = getPeriodo();
  const campo = resultados.campo ||= 'vencimento';
  setCrumbs(['Relatórios', 'Resultados', NOMES_ABA[aba]]);
  el.innerHTML = skeletonPagina();
  const seletor = `<div class="filtros">${controlePeriodo(p)}${aba === 'gerais' ? `<div class="pilula-vidro">${icon('calendario')}<select class="campo" name="campo" aria-label="Tipo de data"><option value="vencimento">Vencimento das parcelas</option><option value="pagamento">Pagamento das parcelas</option><option value="competencia">Competência</option></select></div>` : ''}</div>`;
  let corpo;
  if (aba === 'gerais') {
    const r = await api(`relatorios/resultados?${qs({ ...p, campo })}`);
    const perc = pct(r.resultado.valor, r.receitas.valor);
    corpo = `<div class="grade g-destaque">
      <section class="glass painel hero mira reveal" style="display:grid;place-items:center">${gauge({ pct: Math.abs(perc), rotulo: 'Margem do período', valor: `<span data-count="${perc}" data-fmt="pct">${fmtPct(perc)}</span>`, sub: `Resultado ${brl(r.resultado.valor)}`, cor: r.resultado.valor < 0 ? 'red' : 'amber', ariaLabel: `Resultado de ${brl(r.resultado.valor)}, ${fmtPct(perc)} das receitas` })}</section>
      <div class="grade-kpi" style="align-content:center">${kpi({ rotulo: 'Receitas', icone: 'receitas', valor: r.receitas.valor, qtd: r.receitas.qtd, cor: 'verde' })}${kpi({ rotulo: 'Despesas', icone: 'despesas', valor: r.despesas.valor, qtd: r.despesas.qtd, cor: 'vermelho' })}${kpi({ rotulo: 'Resultado', icone: 'tendencia', valor: r.resultado.valor, qtd: r.resultado.qtd, cor: r.resultado.valor < 0 ? 'vermelho' : 'azul', sparkHtml: spark(r.grafico.map((m) => m.valor), 'var(--cyan)') })}</div></div>
      <section class="glass painel reveal" style="margin-top:var(--s4)"><h3>Resultado por mês (recebido − pago)</h3>${grafico(r.grafico.map((m) => ({ rotulo: rotuloMes(m.mes), partes: [{ valor: m.valor, cor: m.valor < 0 ? CORES.vermelho : CORES.azul }] })))}</section>`;
  } else {
    const pre = aba === 'previsto';
    const r = await api(`relatorios/resultados-projeto?${qs({ ...p, previsto: pre })}`);
    const perc = pct(r.resultado_cents, r.receitas_cents);
    corpo = `<div class="grade g-destaque">
      <section class="glass painel hero mira reveal" style="display:grid;place-items:center">${gauge({ pct: Math.abs(perc), rotulo: pre ? 'Margem prevista' : 'Margem dos projetos', valor: `<span data-count="${perc}" data-fmt="pct">${fmtPct(perc)}</span>`, sub: `${r.itens.length} projeto(s)`, cor: r.resultado_cents < 0 ? 'red' : 'violet', ariaLabel: `Margem dos projetos: ${fmtPct(perc)}` })}</section>
      <div class="grade-kpi" style="align-content:center">${kpi({ rotulo: pre ? 'Previsão de receitas' : 'Receitas', icone: 'receitas', valor: r.receitas_cents, cor: 'verde' })}${kpi({ rotulo: pre ? 'Previsão de despesas' : 'Despesas', icone: 'despesas', valor: r.despesas_cents, cor: 'vermelho' })}${kpi({ rotulo: pre ? 'Resultado previsto' : 'Resultado', icone: 'tendencia', valor: r.resultado_cents, qtd: r.itens.length, cor: r.resultado_cents < 0 ? 'vermelho' : 'azul' })}</div></div>
      ${r.itens.length ? `<section class="glass reveal" style="margin-top:var(--s4)"><div class="tabela-wrap"><table class="tbl"><thead><tr><th>Projeto</th><th>Cliente</th><th class="num">${pre ? 'Prev. receitas' : 'Receitas'}</th><th class="num">${pre ? 'Prev. despesas' : 'Despesas'}</th><th class="num">${pre ? 'Resultado previsto' : 'Resultado'}</th><th class="num">Margem</th></tr></thead><tbody>
      ${r.itens.map((i) => `<tr><td class="nome" data-label="Projeto"><strong>${esc(i.projeto)}</strong></td><td data-label="Cliente">${esc(i.cliente || '-')}</td><td class="num verde" data-label="Receitas">${brl(i.receitas_cents)}</td><td class="num vermelho" data-label="Despesas">${brl(i.despesas_cents)}</td><td class="num" data-label="Resultado"><b>${sv(i.resultado_cents)}</b></td><td class="num" data-label="Margem">${i.receitas_cents ? fmtPct((i.resultado_cents / i.receitas_cents) * 100) : '—'}</td></tr>`).join('')}</tbody></table></div></section>`
      : estadoVazio({ titulo: 'Nenhum projeto com movimento neste período', texto: 'Ligue receitas e despesas a um projeto (contrato) para ver a rentabilidade de cada um.' })}`;
  }
  el.innerHTML = tabs(Object.entries(NOMES_ABA), aba) + seletor + corpo;
  const c = $('[name=campo]', el); if (c) { c.value = campo; c.onchange = (e) => { resultados.campo = e.target.value; resultados(el); }; }
  ligarPeriodo(el, () => resultados(el));
  el.onclick = (e) => { const b = e.target.closest('[data-tab]'); if (b) { resultados.aba = b.dataset.tab; resultados(el); } };
  pronto(el);
}

// ---------- Outros relatórios ----------
const NOMES_AGRUPAR = { cliente: 'Clientes', fornecedor: 'Fornecedores', categoria: 'Categorias', centro_custo: 'Centros de custo', projeto: 'Projetos' };
export async function outros(el) {
  const f = outros.f ||= { agrupar: 'cliente', tipo: 'receita', campo: 'competencia', status: '' };
  const p = getPeriodo();
  setCrumbs(['Relatórios', 'Outros relatórios', NOMES_AGRUPAR[f.agrupar]]);
  el.innerHTML = skeletonPagina();
  const r = await api(`relatorios/outros?${qs({ ...f, ...p })}`);
  const perc = pct(r.total.pago_cents, r.total.valor_cents);
  const sel = (nome, ops) => `<select class="campo" name="${nome}" aria-label="${nome}" style="width:auto">${Object.entries(ops).map(([v, t]) => `<option value="${v}"${f[nome] === v ? ' selected' : ''}>${t}</option>`).join('')}</select>`;
  el.innerHTML = `${tabs(Object.entries(NOMES_AGRUPAR), f.agrupar)}<div class="filtros">${controlePeriodo(p)}
    ${sel('campo', { competencia: 'Competência', vencimento: 'Vencimento', pagamento: 'Pagamento' })}${sel('tipo', { receita: 'Receitas', despesa: 'Despesas' })}${sel('status', { '': 'Todos os status', em_aberto: 'Em aberto', vencido: 'Vencidas', pago: 'Pagas' })}<span class="espaco"></span>
    <a class="btn btn-ghost" href="${urlApi(`relatorios/outros.csv?${qs({ ...f, ...p })}`)}">${icon('baixar')}Exportar planilha</a></div>
    <div class="grade g-destaque">
      <section class="glass painel hero mira reveal" style="display:grid;place-items:center">${gauge({ pct: perc, rotulo: f.tipo === 'receita' ? 'Recebido' : 'Pago', valor: `<span data-count="${perc}" data-fmt="pct">${fmtPct(perc)}</span>`, sub: `${brl(r.total.pago_cents)} de ${brl(r.total.valor_cents)}`, cor: 'cyan', ariaLabel: `${fmtPct(perc)} do total já ${f.tipo === 'receita' ? 'recebido' : 'pago'}` })}</section>
      <div class="grade-kpi" style="align-content:center">${kpi({ rotulo: 'Valor total', icone: 'camadas', valor: r.total.valor_cents, qtd: r.total.qtd, cor: 'azul' })}${kpi({ rotulo: f.tipo === 'receita' ? 'Recebido' : 'Pago', icone: 'check', valor: r.total.pago_cents, cor: 'verde' })}${kpi({ rotulo: 'Em aberto', icone: 'relogio', valor: r.total.aberto_cents, cor: 'laranja' })}</div></div>
    ${r.itens.length ? `<section class="glass reveal" style="margin-top:var(--s4)"><div class="tabela-wrap"><table class="tbl"><thead><tr><th>Nome</th><th class="num">Lançamentos</th><th class="num">Valor</th><th class="num">Pago</th><th class="num">Em aberto</th><th>Participação</th></tr></thead><tbody>
    ${r.itens.map((i) => `<tr><td class="nome" data-label="Nome"><strong>${esc(i.nome)}</strong></td><td class="num" data-label="Lançamentos">${i.qtd}</td><td class="num" data-label="Valor"><b>${brl(i.valor_cents)}</b></td><td class="num verde" data-label="Pago">${brl(i.pago_cents)}</td><td class="num laranja" data-label="Em aberto">${brl(i.aberto_cents)}</td><td data-label="Participação" style="min-width:120px"><div class="progresso violet" role="img" aria-label="${fmtPct(pct(i.valor_cents, r.total.valor_cents))} do total"><i style="width:${pct(i.valor_cents, r.total.valor_cents)}%"></i></div></td></tr>`).join('')}
    <tr class="sub"><td class="nome" data-label="Nome"><strong>Total</strong></td><td class="num" data-label="Lançamentos">${r.total.qtd}</td><td class="num" data-label="Valor"><b>${brl(r.total.valor_cents)}</b></td><td class="num" data-label="Pago">${brl(r.total.pago_cents)}</td><td class="num" data-label="Em aberto">${brl(r.total.aberto_cents)}</td><td></td></tr></tbody></table></div></section>`
    : estadoVazio({ titulo: 'Nada encontrado com estes filtros', texto: 'Amplie o período, troque o tipo de data ou escolha "Todos os status".' })}`;
  for (const n of ['campo', 'tipo', 'status']) $(`[name=${n}]`, el).onchange = (e) => { f[n] = e.target.value; outros(el); };
  ligarPeriodo(el, () => outros(el));
  el.onclick = (e) => {
    const b = e.target.closest('[data-tab]');
    if (!b) return;
    f.agrupar = b.dataset.tab;
    if (b.dataset.tab === 'fornecedor') f.tipo = 'despesa'; else if (b.dataset.tab === 'cliente') f.tipo = 'receita';
    outros(el);
  };
  pronto(el);
}

// ---------- Cadastros ----------
export async function cadastrosView(el, query = {}) {
  const aba = cadastrosView.aba = query.aba || cadastrosView.aba || 'contas';
  const cfg = CADASTROS[aba];
  setCrumbs(['Configurações', 'Cadastros', cfg.titulo]);
  el.innerHTML = skeletonPagina();
  const dados = await api(`cadastros/${aba}`);
  const fmt = (r, [k, , t]) => {
    const v = r[k];
    if (t === 'money') return `<td class="num" data-label="${esc(cfg.colunas.find((c) => c[0] === k)[1])}">${brl(v || 0)}</td>`;
    const rot = esc(cfg.colunas.find((c) => c[0] === k)[1]);
    if (t === 'grupo') return `<td data-label="${rot}">${esc(GRUPOS[v] || v)}</td>`;
    if (k === 'tipo' && aba === 'categorias') return `<td data-label="${rot}"><span class="chip tipo-${esc(v)}">${esc(v)}</span></td>`;
    if (k === 'nome' || k === 'codigo') return `<td class="${k === 'nome' && aba !== 'contratos' ? 'nome' : ''}" data-label="${rot}"><strong>${esc(v ?? '')}</strong></td>`;
    return `<td data-label="${rot}">${esc(v ?? '')}</td>`;
  };
  el.innerHTML = `${tabs(Object.entries(CADASTROS).map(([k, c]) => [k, c.titulo]), aba)}
    <div class="grade"><section class="glass painel reveal" style="display:flex;align-items:center;gap:var(--s4)"><span class="orb">${icon(aba === 'contas' ? 'banco' : aba === 'pessoas' ? 'usuarios' : aba === 'contratos' ? 'contrato' : aba === 'servicos' ? 'tendencia' : 'camadas')}<b></b></span>
      <div><h3>${esc(cfg.titulo)}</h3><div class="kpi-val" data-count="${dados.length}" data-fmt="int">${dados.length}</div></div></section></div>
    ${dados.length ? `<section class="glass reveal" style="margin-top:var(--s4)"><div class="tabela-wrap"><table class="tbl"><thead><tr>${cfg.colunas.map((c) => `<th${c[2] === 'money' ? ' class="num"' : ''}>${c[1]}</th>`).join('')}<th></th></tr></thead><tbody>
    ${dados.map((r) => `<tr data-reg="${r.id}">${cfg.colunas.map((c) => fmt(r, c)).join('')}<td class="acoes" data-label=""><div class="acoes-linha"><button class="btn btn-sm" data-edit="${r.id}" aria-label="Editar">${icon('editar')}Editar</button><button class="btn btn-sm btn-danger" data-del="${r.id}" aria-label="Excluir">${icon('lixo')}</button></div></td></tr>`).join('')}</tbody></table></div></section>`
    : estadoVazio({ titulo: 'Nada cadastrado ainda', texto: `Cadastre ${cfg.singular} para usar nos lançamentos e relatórios.`, acaoRotulo: `Novo(a) ${cfg.singular}`, acaoId: 'estado-novo' })}`;
  el.onclick = async (e) => {
    if (e.target.closest('#estado-novo')) return formCadastro(aba, null, () => cadastrosView(el));
    const b = e.target.closest('button');
    if (!b) return;
    const recarregar = () => cadastrosView(el);
    try {
      if (b.dataset.tab) { cadastrosView.aba = b.dataset.tab; return recarregar(); }
      if (b.dataset.edit) return formCadastro(aba, dados.find((x) => String(x.id) === b.dataset.edit), recarregar);
      if (b.dataset.del && await confirmar('Excluir este cadastro?')) {
        const lin = $(`tr[data-reg="${b.dataset.del}"]`, el);
        excluirComDesfazer({ aviso: 'Cadastro excluído.', ocultar: () => lin.classList.add('linha-oculta'), restaurar: () => lin.classList.remove('linha-oculta'),
          executar: (keepalive) => api(`cadastros/${aba}/${b.dataset.del}`, { method: 'DELETE', keepalive }).then(() => limparCache()), depois: recarregar });
      }
    } catch (err) { toast(err.message, 'erro'); }
  };
  pronto(el);
}

export { formCadastro };
