// Componentes visuais: ícones de linha fina, medidor radial, gráficos neon, esqueletos e estados.

const P = {
  home: '<path d="M3 10.5 12 3l9 7.5"/><path d="M5 9.5V21h14V9.5"/><path d="M10 21v-6h4v6"/>',
  receitas: '<circle cx="12" cy="12" r="9"/><path d="M12 7v10"/><path d="m8 13 4 4 4-4"/>',
  despesas: '<circle cx="12" cy="12" r="9"/><path d="M12 17V7"/><path d="m8 11 4-4 4 4"/>',
  contrato: '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/><path d="M9 13h6M9 17h6"/>',
  transfer: '<path d="M7 7h13"/><path d="m16 3 4 4-4 4"/><path d="M17 17H4"/><path d="m8 13-4 4 4 4"/>',
  banco: '<path d="M3 10 12 4l9 6"/><path d="M5 10v8M9.5 10v8M14.5 10v8M19 10v8"/><path d="M3 21h18"/>',
  fluxo: '<path d="M3 3v18h18"/><path d="m7 15 4-5 3 3 5-7"/>',
  pizza: '<path d="M21 12a9 9 0 1 1-9-9v9z"/><path d="M15.5 3.6A9 9 0 0 1 20.4 8.5H15.5z"/>',
  tendencia: '<path d="m3 17 6-6 4 4 8-8"/><path d="M15 7h6v6"/>',
  camadas: '<path d="M12 3 3 8l9 5 9-5z"/><path d="m3 13 9 5 9-5"/>',
  ajustes: '<path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0"/><circle cx="16" cy="6" r="2"/><circle cx="10" cy="12" r="2"/><circle cx="18" cy="18" r="2"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  search: '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
  moon: '<path d="M20 14.5A8.5 8.5 0 0 1 9.5 4 8.5 8.5 0 1 0 20 14.5z"/>',
  star: '<path d="m12 3 2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1L3.2 9.5l6.1-.9z"/>',
  esq: '<path d="m15 6-6 6 6 6"/>',
  dir: '<path d="m9 6 6 6-6 6"/>',
  baixo: '<path d="m6 9 6 6 6-6"/>',
  menu: '<path d="M4 7h16M4 12h16M4 17h16"/>',
  sair: '<path d="M9 4H5a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h4"/><path d="m16 8 4 4-4 4"/><path d="M20 12H9"/>',
  relogio: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  modulos: '<rect x="4" y="4" width="7" height="7" rx="1.5"/><rect x="13" y="4" width="7" height="7" rx="1.5"/><rect x="4" y="13" width="7" height="7" rx="1.5"/><rect x="13" y="13" width="7" height="7" rx="1.5"/>',
  voltar: '<path d="M19 12H5"/><path d="m11 6-6 6 6 6"/>',
  alert: '<path d="M12 3 2 20h20z"/><path d="M12 10v5M12 18h.01"/>',
  check: '<path d="m5 12 5 5 9-10"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>',
  baixar: '<path d="M12 4v11"/><path d="m7 11 5 5 5-5"/><path d="M5 20h14"/>',
  lixo: '<path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13"/>',
  editar: '<path d="M4 20h4L19 9l-4-4L4 16z"/>',
  desfazer: '<path d="M9 14 4 9l5-5"/><path d="M4 9h10a6 6 0 0 1 0 12h-3"/>',
  x: '<path d="M6 6l12 12M18 6 6 18"/>',
  calendario: '<rect x="3.5" y="5" width="17" height="15.5" rx="2.5"/><path d="M8 3v4M16 3v4M3.5 10h17"/>',
  repetir: '<path d="M17 3l3 3-3 3"/><path d="M4 11V9a3 3 0 0 1 3-3h13"/><path d="M7 21l-3-3 3-3"/><path d="M20 13v2a3 3 0 0 1-3 3H4"/>',
  recolher: '<path d="m14 7-5 5 5 5"/><path d="M19 4v16"/>',
  expandir: '<path d="m10 7 5 5-5 5"/><path d="M5 4v16"/>',
  usuarios: '<circle cx="9" cy="8" r="3.5"/><path d="M2.5 20a6.5 6.5 0 0 1 13 0"/><path d="M16 4.5a3.5 3.5 0 0 1 0 7M18 14a6.5 6.5 0 0 1 3.5 6"/>',
  cartao: '<rect x="3" y="5" width="18" height="14" rx="2.5"/><path d="M3 10h18M7 15h3"/>',
  raio: '<path d="M13 2 4 14h7l-1 8 9-12h-7z"/>',
  alvo: '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
};
P.calendar = P.calendario;

export const icon = (nome, cls = '') => `<svg class="ic ${cls}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">${P[nome] || ''}</svg>`;
// ícones com nomes antigos usados em outros arquivos
P.logout = P.sair; P.plusx = P.plus;

const fmtBrl = (c) => (c / 100).toLocaleString('pt-BR', { style: 'currency', currency: 'BRL' });
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
export const reduzido = () => window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// ---------- medidor radial luminoso (elemento central de destaque) ----------
export function gauge({ pct, rotulo = '', valor = '', sub = '', cor = 'amber', tamanho = 230, ariaLabel }) {
  const p = Math.max(0, Math.min(100, Number.isFinite(pct) ? pct : 0));
  const R = 84, C = 2 * Math.PI * R, to = C * (1 - p / 100);
  const gradiente = { amber: 'gAmber', violet: 'gViolet', cyan: 'gCyan', red: 'gRed' }[cor] || 'gAmber';
  let marcas = '';
  for (let i = 0; i < 60; i++) {
    const a = (i * 6 * Math.PI) / 180, forte = i % 5 === 0, r1 = 93, r2 = forte ? 99 : 96;
    marcas += `<line class="marca${forte ? ' forte' : ''}" x1="${(100 + r1 * Math.cos(a)).toFixed(2)}" y1="${(100 + r1 * Math.sin(a)).toFixed(2)}" x2="${(100 + r2 * Math.cos(a)).toFixed(2)}" y2="${(100 + r2 * Math.sin(a)).toFixed(2)}"/>`;
  }
  const len = String(valor).replace(/<[^>]+>/g, '').length;
  const fs = len <= 5 ? 0.17 : len <= 8 ? 0.125 : len <= 11 ? 0.095 : len <= 13 ? 0.082 : 0.072;
  return `<div class="gauge ${cor}" style="--g:${tamanho}px;--fs:${fs};--c:${C.toFixed(2)};--to:${to.toFixed(2)}" role="img" aria-label="${esc(ariaLabel || `${rotulo}: ${Math.round(p)}%`)}">
    <svg viewBox="0 0 200 200" aria-hidden="true">${marcas}<circle class="anel-fino" cx="100" cy="100" r="64"/><circle class="anel-tracejado" cx="100" cy="100" r="71"/>
      <circle class="trilho-g" cx="100" cy="100" r="${R}"/><circle class="prog" cx="100" cy="100" r="${R}" stroke="url(#${gradiente})"/></svg>
    <div class="gauge-c"><span class="gauge-rot">${esc(rotulo)}</span><span class="gauge-val">${valor}</span><span class="gauge-sub">${sub}</span></div></div>`;
}

// ---------- micrográfico (sparkline) ----------
export function spark(valores, cor = 'var(--amber)') {
  const v = (valores || []).filter((x) => Number.isFinite(x));
  if (v.length < 2) return '';
  const min = Math.min(...v), max = Math.max(...v), d = max - min || 1;
  const pts = v.map((x, i) => [(i / (v.length - 1)) * 100, 27 - ((x - min) / d) * 24]);
  const linha = pts.map((p) => p.map((n) => n.toFixed(1)).join(',')).join(' ');
  return `<svg class="spark" viewBox="0 0 100 30" preserveAspectRatio="none" style="color:${cor}" aria-hidden="true">
    <polygon class="area-s" points="0,30 ${linha} 100,30" fill="currentColor" fill-opacity=".14"/><polyline class="linha-s" pathLength="1" points="${linha}" stroke="currentColor" vector-effect="non-scaling-stroke"/></svg>`;
}

// ---------- gráficos ----------
export const CORES = { pago: 'url(#gBarPago)', atrasado: 'url(#gBarAtraso)', previsto: 'url(#gBarPrev)', azul: 'url(#gBarAzul)', verde: 'url(#gBarPago)', vermelho: 'url(#gBarAtraso)' };
const curto = (c) => { const v = c / 100, a = Math.abs(v); return `${v < 0 ? '-' : ''}${a >= 1e6 ? (a / 1e6).toFixed(1).replace('.', ',') + ' mi' : a >= 1e3 ? Math.round(a / 1e3) + ' mil' : Math.round(a)}`; };

// Barras (empilhadas; aceita valores negativos). barras = [{ rotulo, partes: [{ valor, cor }] }]
export function grafico(barras, { altura = 250 } = {}) {
  const W = 940, H = altura, pl = 64, pr = 8, pt = 24, pb = 28;
  const pos = barras.map((b) => b.partes.filter((p) => p.valor > 0).reduce((s, p) => s + p.valor, 0));
  const neg = barras.map((b) => b.partes.filter((p) => p.valor < 0).reduce((s, p) => s + p.valor, 0));
  const max = Math.max(1, ...pos), min = Math.min(0, ...neg);
  const y = (v) => pt + ((max - v) / (max - min)) * (H - pt - pb);
  const larg = (W - pl - pr) / Math.max(barras.length, 1);
  const bw = Math.min(44, larg * 0.6);
  let svg = `<svg class="grafico" viewBox="0 0 ${W} ${H}" role="img" aria-label="Gráfico de barras por mês">`;
  [0, 1, 2, 3].map((i) => min + ((max - min) * i) / 3).forEach((t) => { svg += `<line class="grade-y" x1="${pl}" x2="${W - pr}" y1="${y(t)}" y2="${y(t)}"/><text x="${pl - 8}" y="${y(t) + 4}" text-anchor="end">${curto(t)}</text>`; });
  barras.forEach((b, i) => {
    const x = pl + larg * i + (larg - bw) / 2;
    let acima = 0, abaixo = 0;
    for (const p of b.partes) {
      if (!p.valor) continue;
      const topo = p.valor > 0 ? acima + p.valor : abaixo;
      const base = p.valor > 0 ? acima : abaixo + p.valor;
      svg += `<rect class="barra${p.valor < 0 ? ' neg' : ''}" style="--i:${i}" x="${x}" y="${y(topo)}" width="${bw}" height="${Math.max(2, y(base) - y(topo))}" rx="5" fill="${p.cor}"><title>${esc(b.rotulo)}: ${fmtBrl(p.valor)}</title></rect>`;
      if (p.valor > 0) acima += p.valor; else abaixo += p.valor;
    }
    const total = acima + abaixo;
    if (total) svg += `<text x="${x + bw / 2}" y="${(total >= 0 ? y(acima) : y(abaixo)) + (total >= 0 ? -7 : 15)}" text-anchor="middle">${curto(total)}</text>`;
    svg += `<text x="${x + bw / 2}" y="${H - 8}" text-anchor="middle">${esc(b.rotulo)}</text>`;
  });
  return svg + '</svg>';
}

// Linhas luminosas com área. series = [{ nome, valores, cor }], rotulos = ['jan', ...]
export function linha(series, rotulos, { altura = 250 } = {}) {
  const W = 940, H = altura, pl = 64, pr = 14, pt = 18, pb = 28;
  const todos = series.flatMap((s) => s.valores);
  const max = Math.max(1, ...todos), min = Math.min(0, ...todos);
  const x = (i) => pl + (i / Math.max(rotulos.length - 1, 1)) * (W - pl - pr);
  const y = (v) => pt + ((max - v) / (max - min)) * (H - pt - pb);
  let svg = `<svg class="grafico" viewBox="0 0 ${W} ${H}" role="img" aria-label="Gráfico de linhas por mês">`;
  [0, 1, 2, 3].map((i) => min + ((max - min) * i) / 3).forEach((t) => { svg += `<line class="grade-y" x1="${pl}" x2="${W - pr}" y1="${y(t)}" y2="${y(t)}"/><text x="${pl - 8}" y="${y(t) + 4}" text-anchor="end">${curto(t)}</text>`; });
  rotulos.forEach((r, i) => { svg += `<text x="${x(i)}" y="${H - 8}" text-anchor="middle">${esc(r)}</text>`; });
  series.forEach((s, k) => {
    const pts = s.valores.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
    svg += `<g style="color:${s.cor}">${k === 0 ? `<polygon class="area-g" points="${x(0)},${y(Math.max(min, 0))} ${pts} ${x(s.valores.length - 1)},${y(Math.max(min, 0))}" fill="currentColor" fill-opacity=".12"/>` : ''}
      <polyline class="linha-g" pathLength="1" points="${pts}" stroke="currentColor"${s.tracejada ? ' stroke-dasharray="0.02 0.02"' : ''}/>`;
    s.valores.forEach((v, i) => { svg += `<circle class="ponto" cx="${x(i).toFixed(1)}" cy="${y(v).toFixed(1)}" r="3.4" stroke="currentColor"><title>${esc(s.nome)} · ${esc(rotulos[i])}: ${fmtBrl(v)}</title></circle>`; });
    svg += '</g>';
  });
  return svg + '</svg>';
}

// Rosca (donut). partes = [{ rotulo, valor, cor }]
export function rosca(partes, { centro = '' } = {}) {
  const total = partes.reduce((s, p) => s + p.valor, 0);
  if (!total) return '';
  const R = 60, C = 2 * Math.PI * R;
  let acum = 0;
  const arcos = partes.map((p) => {
    const dash = (p.valor / total) * C, off = -acum;
    acum += dash;
    return `<circle cx="80" cy="80" r="${R}" stroke="${p.cor}" style="--c:${C.toFixed(1)}" stroke-dasharray="${Math.max(dash - 3, 0.5).toFixed(1)} ${C.toFixed(1)}" stroke-dashoffset="${off.toFixed(1)}" stroke-linecap="round"><title>${esc(p.rotulo)}: ${fmtBrl(p.valor)}</title></circle>`;
  }).join('');
  return `<div class="rosca-wrap"><div style="position:relative;width:150px;height:150px"><svg class="rosca" viewBox="0 0 160 160" role="img" aria-label="Distribuição por categoria"><circle cx="80" cy="80" r="${R}" stroke="var(--glass-2)" style="animation:none" stroke-dasharray="${C.toFixed(1)}"/>${arcos}</svg>
    <div style="position:absolute;inset:0;display:grid;place-content:center;text-align:center"><span class="gauge-rot">${esc(centro)}</span></div></div>
    <ul class="rosca-leg">${partes.map((p) => `<li><span class="nm"><i style="background:${p.cor}"></i>${esc(p.rotulo)}</span><b class="num">${Math.round((p.valor / total) * 100)}%</b></li>`).join('')}</ul></div>`;
}

// ---------- KPI de vidro ----------
export function kpi({ rotulo, icone = '', valor, qtd, cor = '', destino = '', status, ativo = false, sparkHtml = '', sub = '', formato = 'brl', tag = 'button' }) {
  const attrs = destino ? ` data-ir="${esc(destino)}"` : status !== undefined ? ` data-status="${esc(status)}"` : '';
  const t = attrs ? tag : 'div';
  return `<${t} class="kpi reveal${ativo ? ' ativa' : ''}"${attrs}${t === 'button' ? ' type="button"' : ''}>
    <span class="kpi-topo"><span class="kpi-rot">${icone ? icon(icone) : ''}${esc(rotulo)}</span>${qtd !== undefined ? `<span class="kpi-qtd">${qtd}</span>` : ''}</span>
    <span class="kpi-val ${cor}" data-count="${valor}" data-fmt="${formato}">${formato === 'brl' ? fmtBrl(valor) : valor}</span>${sub ? `<span class="kpi-var">${sub}</span>` : ''}${sparkHtml}</${t}>`;
}

export function chipStatus(status, textos) {
  const t = textos || { pago: 'Pago', vencido: 'Vencido', vence_hoje: 'Vence hoje', a_vencer: 'A vencer', em_aberto: 'Em aberto' };
  const ic = { pago: 'check', vencido: 'alert' }[status] || 'relogio';
  return `<span class="chip s-${status}">${icon(ic)}${esc(t[status] || status)}</span>`;
}

// ---------- animações ----------
export function animarContadores(raiz = document) {
  const lista = [...raiz.querySelectorAll('[data-count]')];
  if (reduzido()) return;
  lista.forEach((el) => {
    const alvo = Number(el.dataset.count);
    if (!Number.isFinite(alvo)) return;
    const fmt = el.dataset.fmt;
    const mostrar = (v) => { el.textContent = fmt === 'brl' ? fmtBrl(v) : fmt === 'pct' ? `${v.toFixed(1).replace('.', ',')}%` : Math.round(v).toLocaleString('pt-BR'); };
    const dur = 900, ini = performance.now();
    mostrar(0);
    const passo = (t) => {
      const k = Math.min(1, (t - ini) / dur), e = 1 - Math.pow(1 - k, 3);
      mostrar(alvo * e);
      if (k < 1) requestAnimationFrame(passo); else mostrar(alvo);
    };
    requestAnimationFrame(passo);
  });
}
export function revelar(raiz) {
  [...raiz.querySelectorAll('.reveal')].forEach((el, i) => el.style.setProperty('--i', Math.min(i, 14)));
}

// ---------- estados: carregando, vazio, erro ----------
export const sk = (h = 16, w = '100%', r = 16) => `<div class="sk" style="height:${h}px;width:${w};border-radius:${r}px"></div>`;
export const skeletonPagina = () => `<div class="empilha" aria-busy="true" aria-label="Carregando">
  <div class="grade g2">${sk(220, '100%', 22)}${sk(220, '100%', 22)}</div>
  <div class="grade-kpi">${[1, 2, 3, 4].map(() => sk(104, '100%', 22)).join('')}</div>
  ${sk(300, '100%', 22)}</div>`;

const ILU = `<svg class="ilu" viewBox="0 0 120 120" aria-hidden="true"><defs><linearGradient id="iluG" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#FF9F1C"/><stop offset="1" stop-color="#8B5CF6"/></linearGradient></defs>
  <circle cx="60" cy="60" r="46" fill="none" stroke="url(#iluG)" stroke-width="1.5" opacity=".5" stroke-dasharray="3 6"/><rect x="34" y="22" width="52" height="76" rx="12" fill="rgba(255,255,255,.06)" stroke="url(#iluG)" stroke-width="1.5"/>
  <circle cx="60" cy="56" r="14" fill="none" stroke="#FF9F1C" stroke-width="3" stroke-dasharray="60 100" stroke-linecap="round"/><path d="M46 80h28M50 87h20" stroke="#22D3EE" stroke-width="2" stroke-linecap="round" opacity=".8"/></svg>`;
export function estadoVazio({ titulo, texto = '', acaoRotulo = '', acaoId = 'estado-acao' }) {
  return `<div class="estado glass reveal">${ILU}<h3>${esc(titulo)}</h3>${texto ? `<p>${esc(texto)}</p>` : ''}${acaoRotulo ? `<button class="btn btn-primary" id="${acaoId}" type="button">${icon('plus')}${esc(acaoRotulo)}</button>` : ''}</div>`;
}
export function estadoErro({ titulo = 'Não foi possível carregar esta tela', texto = '', refazer = true }) {
  return `<div class="estado glass erro-estado reveal"><div class="orb amber">${icon('alert')}</div><h3>${esc(titulo)}</h3>
    <p>${esc(texto || 'Verifique sua conexão e tente de novo. Se continuar, entre novamente com a senha.')}</p>${refazer ? `<button class="btn btn-primary" id="estado-refazer" type="button">${icon('repetir')}Tentar de novo</button>` : ''}</div>`;
}
