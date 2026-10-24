// A receber: quem está devendo, o que vence nos próximos dias e o efeito disso no caixa.
import { $, $$, api, brl, dataBR, esc, qs } from './util.js';
import { animarContadores, chipStatus, estadoVazio, gauge, icon, kpi, revelar, skeletonPagina } from './ui.js';
import { formBaixa } from './forms.js';

const fmtPct = (v) => `${v.toFixed(1).replace('.', ',')}%`;
const soDigitos = (v) => String(v || '').replace(/\D/g, '');
const atrasoTxt = (n) => `${n} dia${n === 1 ? '' : 's'} de atraso`;

function mensagem(c) {
  const venc = c.parcelas.filter((p) => p.status === 'vencido');
  const lista = venc.map((p) => `• ${p.nome} — ${brl(p.aberto_cents)} (venc. ${dataBR(p.vencimento)})`).join('\n');
  return `Olá, ${c.nome.split(' ')[0]}! Tudo bem? Passando para lembrar dos pagamentos em aberto com a Cariati Arquitetura & Gestão:\n${lista}\nQualquer dúvida, é só falar. Obrigado!`;
}
function acaoContato(c) {
  const tel = soDigitos(c.telefone);
  const zap = tel.length >= 10 ? `<a class="btn btn-sm" target="_blank" rel="noopener" href="https://wa.me/${tel.startsWith('55') ? tel : '55' + tel}?text=${encodeURIComponent(mensagem(c))}">${icon('raio')}Cobrar no WhatsApp</a>` : '';
  const mail = c.email ? `<a class="btn btn-sm" href="mailto:${esc(c.email)}?subject=${encodeURIComponent('Pagamentos em aberto - Cariati')}&body=${encodeURIComponent(mensagem(c))}">E-mail</a>` : '';
  return zap + mail;
}

export async function aReceberView(el) {
  const filtro = aReceberView.filtro ||= { busca: '', soAtraso: false };
  window.dispatchEvent(new CustomEvent('crumbs', { detail: ['Financeiro', 'A receber'] }));
  if (el.dataset.tela !== 'ar') el.innerHTML = skeletonPagina();
  const r = await api(`a-receber?${qs({ busca: filtro.busca })}`);
  el.dataset.tela = 'ar';
  const t = r.totais;
  const perc = t.em_aberto.valor_cents ? (t.vencido.valor_cents / t.em_aberto.valor_cents) * 100 : 0;
  const lista = filtro.soAtraso ? r.clientes.filter((c) => c.vencido_cents > 0) : r.clientes;
  const maxAging = Math.max(1, ...r.aging.map((a) => a.valor_cents));
  const neg = r.caixa.projecao.find((p) => p.saldo_projetado_cents < 0);

  const cartao = (c, i) => `<article class="glass painel reveal cobranca${c.vencido_cents ? ' em-atraso' : ''}" data-pessoa="${i}">
    <header class="cobranca-topo"><button class="toggle" type="button" data-exp="${i}" aria-expanded="${c.vencido_cents ? 'true' : 'false'}" aria-label="Mostrar parcelas de ${esc(c.nome)}">${icon('dir')}</button>
      <div class="cobranca-nome"><strong>${esc(c.nome)}</strong><small>${esc([c.codigo, c.telefone].filter(Boolean).join(' · '))}</small></div>
      <div class="cobranca-valores">
        ${c.vencido_cents ? `<span class="chip s-vencido">${icon('alert')}${brl(c.vencido_cents)} vencido · ${atrasoTxt(c.max_atraso_dias)}</span>` : ''}
        ${c.a_vencer_cents ? `<span class="chip s-a_vencer">${icon('relogio')}${brl(c.a_vencer_cents)} a vencer${c.proximo_vencimento ? ' · próx. ' + dataBR(c.proximo_vencimento) : ''}</span>` : ''}
      </div></header>
    <div class="cobranca-det"${c.vencido_cents ? '' : ' hidden'}>
      <table class="tbl-mini">${c.parcelas.map((p, j) => `<tr><td>${esc(p.nome)}${p.contrato_codigo ? ` <span class="suave">· ${esc(p.contrato_codigo)}</span>` : ''}</td><td>${dataBR(p.vencimento)}</td>
        <td class="num"><b>${brl(p.aberto_cents)}</b></td><td>${p.status === 'vencido' ? `<span class="chip s-vencido">${atrasoTxt(p.dias_atraso)}</span>` : chipStatus(p.status)}</td>
        <td><button type="button" class="btn btn-sm" data-baixa="${i}:${j}">${icon('check')}Receber</button></td></tr>`).join('')}</table>
      ${c.vencido_cents ? `<div class="cobranca-acoes">${acaoContato(c)}</div>` : ''}</div></article>`;

  el.innerHTML = `
    <div class="filtros"><label class="pilula-vidro busca-campo">${icon('search')}<input class="campo" type="search" name="busca" placeholder="Pesquisar cliente, projeto ou parcela" value="${esc(filtro.busca)}" aria-label="Pesquisar"></label>
      <label class="switch" style="min-height:40px"><input type="checkbox" name="soAtraso"${filtro.soAtraso ? ' checked' : ''}><span class="trilho" aria-hidden="true"></span><span>Só clientes em atraso</span></label></div>
    <div class="grade g-destaque">
      <section class="glass painel hero mira reveal" style="display:grid;place-items:center">${gauge({ pct: perc, rotulo: 'Em atraso', valor: `<span data-count="${perc}" data-fmt="pct">${fmtPct(perc)}</span>`, sub: `${brl(t.vencido.valor_cents)} de ${brl(t.em_aberto.valor_cents)} a receber`, cor: perc > 25 ? 'red' : 'amber', ariaLabel: `${fmtPct(perc)} do que há a receber está vencido` })}</section>
      <div class="grade-kpi" style="align-content:center">
        ${kpi({ rotulo: 'Vencido', icone: 'alert', valor: t.vencido.valor_cents, qtd: t.vencido.qtd, cor: 'vermelho', sub: `${t.clientes_em_atraso} cliente(s) em atraso` })}
        ${kpi({ rotulo: 'Vence hoje', icone: 'relogio', valor: t.vence_hoje.valor_cents, qtd: t.vence_hoje.qtd, cor: 'laranja' })}
        ${kpi({ rotulo: 'Próximos 7 dias', icone: 'calendario', valor: t.proximos_7.valor_cents, qtd: t.proximos_7.qtd, cor: 'azul' })}
        ${kpi({ rotulo: 'Próximos 30 dias', icone: 'calendario', valor: t.proximos_30.valor_cents, qtd: t.proximos_30.qtd, cor: 'violet' })}
      </div></div>
    <div class="grade g2" style="margin-top:var(--s4)">
      <section class="glass painel reveal"><h3>Compensação no caixa</h3>
        <p class="suave" style="margin-bottom:var(--s3)">Saldo atual das contas: <b class="num">${brl(r.caixa.saldo_cents)}</b>. Considera o que entra e o que sai em cada prazo (sem contar os atrasados).</p>
        <div style="overflow-x:auto"><table class="tbl-mini"><thead><tr><th>Prazo</th><th class="num">Entra</th><th class="num">Sai</th><th class="num">Saldo</th></tr></thead><tbody>
          ${r.caixa.projecao.map((p) => `<tr><td>${p.dias} dias</td><td class="num verde">${brl(p.entra_cents)}</td><td class="num vermelho">${brl(p.sai_cents)}</td><td class="num"><b class="${p.saldo_projetado_cents < 0 ? 'vermelho' : ''}">${brl(p.saldo_projetado_cents)}</b></td></tr>`).join('')}</tbody></table></div>
        ${neg ? `<p class="aviso-caixa">${icon('alert')}<span>Atenção: o caixa fica negativo em até ${neg.dias} dias. Receber os ${brl(t.vencido.valor_cents)} em atraso cobriria ${t.vencido.valor_cents >= -neg.saldo_projetado_cents ? 'a diferença' : 'parte da diferença'}.</span></p>`
          : t.vencido.valor_cents ? `<p class="suave" style="margin-top:var(--s3)">Receber os ${brl(t.vencido.valor_cents)} em atraso reforçaria o caixa dos próximos dias.</p>` : ''}
        ${r.caixa.vencido_pagar_cents ? `<p class="suave" style="margin-top:var(--s2)">Você também tem ${brl(r.caixa.vencido_pagar_cents)} de despesas vencidas a pagar.</p>` : ''}</section>
      <section class="glass painel reveal"><h3>Tempo de atraso</h3>
        ${r.aging.map((a) => `<div class="aging"><div class="aging-topo"><span>${a.rotulo}</span><b class="num">${brl(a.valor_cents)}</b><span class="suave">${a.qtd} parcela(s)</span></div>
          <div class="progresso"><i style="width:${(a.valor_cents / maxAging) * 100}%;background:var(--grad-err, linear-gradient(90deg,#F87171,#FB923C))"></i></div></div>`).join('')}</section>
    </div>
    <h2 style="margin:var(--s6) 0 var(--s3)">Clientes com pagamentos pendentes</h2>
    ${lista.length ? `<div class="empilha">${lista.map(cartao).join('')}</div>`
    : estadoVazio({ titulo: filtro.soAtraso || filtro.busca ? 'Nenhum cliente encontrado' : 'Nada pendente', texto: filtro.soAtraso || filtro.busca ? 'Ajuste o filtro para ver outros clientes.' : 'Todos os pagamentos estão em dia. Bom trabalho!', acaoRotulo: '', acaoId: '' })}
    ${r.agenda.length ? `<h2 style="margin:var(--s6) 0 var(--s3)">Agenda: próximos vencimentos</h2><section class="glass reveal"><div class="tabela-wrap"><table class="tbl"><thead><tr><th>Vencimento</th><th>Parcela</th><th>Cliente</th><th class="num">Valor</th><th>Status</th></tr></thead><tbody>
      ${r.agenda.map((p) => `<tr><td data-label="Vencimento">${dataBR(p.vencimento)}</td><td class="nome" data-label="Parcela"><strong>${esc(p.nome)}</strong></td><td data-label="Cliente">${esc(p.pessoa_nome || '—')}</td><td class="num" data-label="Valor"><b>${brl(p.aberto_cents)}</b></td><td data-label="Status">${chipStatus(p.status)}</td></tr>`).join('')}</tbody></table></div></section>` : ''}`;

  let tmr;
  $('[name=busca]', el).oninput = (e) => { clearTimeout(tmr); tmr = setTimeout(() => { filtro.busca = e.target.value; aReceberView(el).then(() => { const b = $('[name=busca]', el); b.focus(); b.setSelectionRange(b.value.length, b.value.length); }); }, 350); };
  $('[name=soAtraso]', el).onchange = (e) => { filtro.soAtraso = e.target.checked; aReceberView(el); };
  el.onclick = (e) => {
    const x = e.target.closest('[data-exp]');
    if (x) { const det = $('.cobranca-det', x.closest('.cobranca')); det.hidden = !det.hidden; x.setAttribute('aria-expanded', String(!det.hidden)); return; }
    const b = e.target.closest('[data-baixa]');
    if (b) { const [i, j] = b.dataset.baixa.split(':').map(Number); formBaixa(lista[i].parcelas[j], () => aReceberView(el)); }
  };
  revelar(el); animarContadores(el);
}
