import { $, $$, api, dataBR, dinheiroInput, esc, hojeISO, modal, opcoes, parseDinheiro, toast } from './util.js';
import { icon } from './ui.js';

export const GRUPOS = {
  receita_bruta: 'Receita operacional bruta', deducoes: 'Deduções da receita bruta', custos_operacionais: 'Custos operacionais',
  despesas_operacionais: 'Despesas operacionais', financeiras: 'Receitas e despesas financeiras', nao_operacionais: 'Outras receitas e despesas não operacionais',
};

let cache = null;
export async function cadastros(forcar = false) {
  if (!cache || forcar) {
    const [contas, categorias, centros, pessoas, contratos, servicos] = await Promise.all(
      ['contas', 'categorias', 'centros', 'pessoas', 'contratos', 'servicos'].map((c) => api(`cadastros/${c}`)));
    cache = { contas: contas.filter((c) => c.ativa), todasContas: contas, categorias, centros, pessoas, contratos, servicos };
  }
  return cache;
}
export const limparCache = () => { cache = null; };

const guardar = (k, v) => { try { localStorage.setItem(k, v); } catch { /* sem storage */ } };
const ler = (k) => { try { return localStorage.getItem(k); } catch { return null; } };
const tirar = (k) => { try { localStorage.removeItem(k); } catch { /* sem storage */ } };
const interruptor = (nome, texto, extra = '') => `<label class="switch"><input type="checkbox" name="${nome}"${extra}><span class="trilho" aria-hidden="true"></span><span>${texto}</span></label>`;

// ---------- novo lançamento (receita ou despesa), em 3 etapas ----------
export async function formLancamento(tipo, aoSalvar, prefill = {}) {
  const c = await cadastros();
  const ehReceita = tipo === 'receita';
  const pessoas = c.pessoas.filter((p) => p.tipo === 'ambos' || p.tipo === (ehReceita ? 'cliente' : 'fornecedor'));
  const cats = c.categorias.filter((x) => x.tipo === tipo);
  const hoje = hojeISO();
  const chaveRascunho = `rascunho_${tipo}`;
  const contaPadrao = ler('ultimaConta') && c.contas.some((x) => x.id === ler('ultimaConta')) ? ler('ultimaConta') : c.contas[0]?.id;
  const catPadrao = ler(`ultimaCategoria_${tipo}`) && cats.some((x) => x.id === ler(`ultimaCategoria_${tipo}`)) ? ler(`ultimaCategoria_${tipo}`) : '';
  const NOMES = ['Básico', 'Classificação', 'Detalhes'];

  const corpo = `
    <div class="passos" aria-hidden="true" style="padding:0">${NOMES.map((n, i) => `<div class="passo" data-ind="${i + 1}"><i></i>${i + 1}. ${n}</div>`).join('')}</div>
    <div class="aviso-rascunho" id="aviso-rascunho" hidden>${icon('info')}<span style="flex:1">Rascunho recuperado automaticamente.</span><button type="button" class="btn btn-ghost btn-sm" id="descartar-rascunho">Descartar</button></div>
    <section data-passo="1" class="empilha" aria-label="Etapa 1: Básico">
      <label class="f">Nome *<input class="campo" name="nome" required autocomplete="off" placeholder="${ehReceita ? 'Ex.: Projeto arquitetônico — Residência Silva' : 'Ex.: Pagamento engenheiro estrutural'}"><span class="msg" data-msg="nome"></span></label>
      <div class="linha2">
        <label class="f">Valor (R$) *<input class="campo" name="valor" inputmode="decimal" required placeholder="0,00" autocomplete="off"><span class="dica" data-dica="valor">Total do lançamento — ou o valor de cada mês, se repetir.</span><span class="msg" data-msg="valor"></span></label>
        <label class="f">Primeiro vencimento *<input class="campo" type="date" name="primeiro_vencimento" value="${hoje}" required><span class="msg" data-msg="primeiro_vencimento"></span></label>
      </div>
      <div class="linha2" id="bloco-parcelas">
        <label class="f">Número de parcelas<input class="campo" type="number" name="parcelas" min="1" max="360" value="1" inputmode="numeric"><span class="dica" data-dica="parcelas"></span><span class="msg" data-msg="parcelas"></span></label>
        <div></div>
      </div>
      ${interruptor('recorrente', `Repete todo mês <span class="suave">(aluguel, salário, assinatura…)</span>`)}
      <label class="f" id="bloco-repeticoes" hidden>Quantos meses<input class="campo" type="number" name="repeticoes" min="1" max="360" value="12" inputmode="numeric"><span class="msg" data-msg="repeticoes"></span></label>
    </section>
    <section data-passo="2" class="empilha" aria-label="Etapa 2: Classificação" hidden>
      <div class="linha2">
        <label class="f">${ehReceita ? 'Cliente' : 'Fornecedor'}<select class="campo" name="pessoa_id">${opcoes(pessoas, '', '—')}</select></label>
        <label class="f">Categoria *<select class="campo" name="categoria_id" required>${opcoes(cats, catPadrao, 'Escolha…')}</select><span class="msg" data-msg="categoria_id"></span></label>
      </div>
      <div class="linha2">
        <label class="f">Centro de custo<select class="campo" name="centro_custo_id">${opcoes(c.centros, '', '—')}</select></label>
        <label class="f">Projeto / contrato<select class="campo" name="contrato_id">${opcoes(c.contratos, '', 'Nenhum (é do escritório)', (x) => `${x.codigo} - ${x.nome}`)}</select></label>
      </div>
      <label class="f">Conta bancária<select class="campo" name="conta_id">${opcoes(c.contas, contaPadrao, '—')}</select></label>
      ${pessoas.length === 0 ? `<p class="suave">Nenhum ${ehReceita ? 'cliente' : 'fornecedor'} cadastrado ainda. Cadastre em Cadastros → Clientes e fornecedores.</p>` : ''}
    </section>
    <section data-passo="3" class="empilha" aria-label="Etapa 3: Detalhes" hidden>
      <div class="linha2">
        <label class="f">Data de competência<input class="campo" type="date" name="competencia" value="${hoje}"><span class="dica">Mês a que o valor pertence na DRE.</span></label>
        <label class="f">Nota fiscal<input class="campo" name="nota_fiscal" autocomplete="off"></label>
      </div>
      <label class="f">Etiquetas<input class="campo" name="etiquetas" placeholder="separe por vírgula" autocomplete="off"></label>
      ${interruptor('primeira_paga', ehReceita ? 'A primeira parcela já foi recebida' : 'A primeira parcela já foi paga')}
      <label class="f" id="bloco-pagamento" hidden>Data do ${ehReceita ? 'recebimento' : 'pagamento'}<input class="campo" type="date" name="data_pagamento" value="${hoje}"></label>
      <div class="glass painel" id="resumo-lanc" style="padding:var(--s4)"></div>
    </section>`;

  const rodape = `<button type="button" class="btn btn-ghost" id="passo-voltar" data-fechar>Cancelar</button>
    <button type="button" class="btn btn-primary" id="passo-seguir">Continuar</button>
    <button type="submit" class="btn btn-primary" id="passo-criar" hidden>${icon('check')}Criar lançamento</button>`;

  const m = modal(ehReceita ? 'Nova receita' : 'Nova despesa', corpo, {
    rotulo: 'Criar lançamento', rodape, classe: 'folha-passos',
    onSubmit: async (d) => {
      const valor = parseDinheiro(d.valor);
      if (!valor || valor <= 0) throw new Error('Informe um valor válido, como 1.250,00.');
      const n = (v) => v || null;
      await api('lancamentos', { method: 'POST', body: {
        tipo, nome: d.nome, valor_total_cents: valor, primeiro_vencimento: d.primeiro_vencimento,
        competencia: d.competencia || d.primeiro_vencimento, parcelas: Number(d.parcelas || 1), recorrente: d.recorrente,
        repeticoes: Number(d.repeticoes || 12), pessoa_id: n(d.pessoa_id), categoria_id: n(d.categoria_id), centro_custo_id: n(d.centro_custo_id),
        contrato_id: n(d.contrato_id), conta_id: n(d.conta_id), nota_fiscal: d.nota_fiscal || null, etiquetas: d.etiquetas || null,
        primeira_paga: d.primeira_paga, data_pagamento: d.data_pagamento,
      } });
      if (d.conta_id) guardar('ultimaConta', d.conta_id);
      if (d.categoria_id) guardar(`ultimaCategoria_${tipo}`, d.categoria_id);
      tirar(chaveRascunho);
      toast(ehReceita ? 'Receita criada.' : 'Despesa criada.');
      aoSalvar?.();
    },
  });
  const f = $('form', m.dlg);
  let passo = 1;

  // validação em tempo real
  const regras = {
    nome: (v) => (v.trim().length >= 2 ? '' : 'Informe o nome (mínimo 2 letras).'),
    valor: (v) => { const x = parseDinheiro(v); return x && x > 0 ? '' : 'Digite um valor maior que zero, como 1.250,00.'; },
    primeiro_vencimento: (v) => (v ? '' : 'Escolha a data do primeiro vencimento.'),
    parcelas: (v) => { const x = Number(v); return Number.isInteger(x) && x >= 1 && x <= 360 ? '' : 'Use um número de 1 a 360.'; },
    repeticoes: (v) => { const x = Number(v); return Number.isInteger(x) && x >= 1 && x <= 360 ? '' : 'Use um número de 1 a 360.'; },
    categoria_id: (v) => (v ? '' : 'Escolha uma categoria.'),
  };
  const validar = (nome) => {
    const campo = f.elements[nome];
    if (!campo || campo.closest('[hidden]') && nome !== 'categoria_id') return true;
    if ((nome === 'parcelas' && f.recorrente.checked) || (nome === 'repeticoes' && !f.recorrente.checked)) return true;
    const erro = regras[nome](campo.value);
    const msg = $(`[data-msg="${nome}"]`, f);
    if (msg) msg.textContent = erro;
    campo.setAttribute('aria-invalid', erro ? 'true' : 'false');
    return !erro;
  };
  const camposDoPasso = { 1: ['nome', 'valor', 'primeiro_vencimento', 'parcelas', 'repeticoes'], 2: ['categoria_id'], 3: [] };
  Object.keys(regras).forEach((nome) => {
    const campo = f.elements[nome];
    campo.addEventListener('input', () => { if (campo.getAttribute('aria-invalid') === 'true') validar(nome); atualizarDicas(); });
    campo.addEventListener('blur', () => { if (campo.value !== '' || campo.required) validar(nome); });
  });

  // dicas inteligentes (valor de cada parcela) e resumo da etapa final
  function atualizarDicas() {
    const v = parseDinheiro(f.valor.value), n = Number(f.parcelas.value);
    const dica = $('[data-dica="parcelas"]', f);
    if (v && !f.recorrente.checked && n > 1 && n <= 360) dica.textContent = `${n} parcelas de cerca de ${dinheiroInput(Math.floor(v / n))} (a soma fecha exatamente o total).`;
    else dica.textContent = '';
    $('[data-dica="valor"]', f).textContent = f.recorrente.checked ? 'Valor de CADA mês.' : 'Valor total do lançamento.';
    const pessoa = f.pessoa_id.selectedOptions[0]?.text, cat = f.categoria_id.selectedOptions[0]?.text;
    $('#resumo-lanc', f).innerHTML = `<h3>Resumo</h3><ul class="lista"><li><span>${esc(f.nome.value || '—')}</span><b class="num">${v ? dinheiroInput(v) : '—'}</b></li>
      <li><span class="suave">${f.recorrente.checked ? `Todo mês, ${f.repeticoes.value} vezes` : `${n || 1} parcela(s)`} · 1º vencimento ${dataBR(f.primeiro_vencimento.value)}</span></li>
      <li><span class="suave">${esc(cat && cat !== 'Escolha…' ? cat : 'Sem categoria')}${pessoa && pessoa !== '—' ? ' · ' + esc(pessoa) : ''}</span></li></ul>`;
  }

  f.recorrente.onchange = () => {
    $('#bloco-repeticoes', f).hidden = !f.recorrente.checked;
    $('#bloco-parcelas', f).hidden = f.recorrente.checked;
    atualizarDicas();
  };
  f.primeira_paga.onchange = () => { $('#bloco-pagamento', f).hidden = !f.primeira_paga.checked; };
  f.primeiro_vencimento.addEventListener('change', () => { if (!f.competencia.dataset.manual) f.competencia.value = f.primeiro_vencimento.value; });
  f.competencia.addEventListener('change', () => { f.competencia.dataset.manual = '1'; });
  f.pessoa_id.addEventListener('change', atualizarDicas);
  f.categoria_id.addEventListener('change', atualizarDicas);

  // etapas
  const ir = (n) => {
    passo = n;
    $$('[data-passo]', f).forEach((s) => { s.hidden = Number(s.dataset.passo) !== n; });
    $$('.passo', f).forEach((p) => { const i = Number(p.dataset.ind); p.classList.toggle('atual', i === n); p.classList.toggle('feito', i < n); });
    $('#passo-seguir', f).hidden = n === 3;
    $('#passo-criar', f).hidden = n !== 3;
    const voltar = $('#passo-voltar', f);
    voltar.textContent = n === 1 ? 'Cancelar' : 'Voltar';
    if (n === 1) voltar.setAttribute('data-fechar', ''); else voltar.removeAttribute('data-fechar');
    $('.erro', m.dlg).textContent = '';
    atualizarDicas();
    const primeiro = $(`[data-passo="${n}"] .campo`, f);
    if (primeiro) setTimeout(() => primeiro.focus({ preventScroll: true }), 30);
    salvarRascunho();
  };
  const passoValido = (n) => {
    const ruins = camposDoPasso[n].filter((nome) => !validar(nome));
    if (ruins.length) f.elements[ruins[0]].focus();
    return !ruins.length;
  };
  $('#passo-seguir', f).onclick = () => { if (passoValido(passo)) ir(passo + 1); };
  $('#passo-voltar', f).onclick = () => { if (passo > 1) ir(passo - 1); else m.fechar(); };
  // Enter nas primeiras etapas avança em vez de enviar
  f.addEventListener('submit', (e) => {
    if (passo < 3) { e.preventDefault(); e.stopImmediatePropagation(); if (passoValido(passo)) ir(passo + 1); }
  }, true);

  // rascunho salvo automaticamente
  const CAMPOS_RASCUNHO = ['nome', 'valor', 'primeiro_vencimento', 'parcelas', 'repeticoes', 'pessoa_id', 'categoria_id', 'centro_custo_id', 'contrato_id', 'conta_id', 'competencia', 'nota_fiscal', 'etiquetas', 'data_pagamento'];
  let tmr;
  function salvarRascunho() {
    clearTimeout(tmr);
    tmr = setTimeout(() => {
      if (!f.nome.value && !f.valor.value) return;
      const dados = { passo };
      CAMPOS_RASCUNHO.forEach((k) => { dados[k] = f.elements[k].value; });
      dados.recorrente = f.recorrente.checked; dados.primeira_paga = f.primeira_paga.checked;
      guardar(chaveRascunho, JSON.stringify(dados));
    }, 350);
  }
  f.addEventListener('input', salvarRascunho);
  f.addEventListener('change', salvarRascunho);
  const bruto = ler(chaveRascunho);
  if (bruto) {
    try {
      const r = JSON.parse(bruto);
      CAMPOS_RASCUNHO.forEach((k) => { if (r[k] !== undefined && f.elements[k]) f.elements[k].value = r[k]; });
      f.recorrente.checked = !!r.recorrente; f.primeira_paga.checked = !!r.primeira_paga;
      f.recorrente.onchange(); f.primeira_paga.onchange();
      $('#aviso-rascunho', f).hidden = false;
      if (r.competencia && r.competencia !== r.primeiro_vencimento) f.competencia.dataset.manual = '1';
    } catch { tirar(chaveRascunho); }
  }
  if (prefill.pessoa_id) f.pessoa_id.value = prefill.pessoa_id;
  if (prefill.contrato_id) f.contrato_id.value = prefill.contrato_id;
  $('#descartar-rascunho', f).onclick = () => { tirar(chaveRascunho); m.fechar(); formLancamento(tipo, aoSalvar, prefill); };
  ir(1);
  setTimeout(() => f.nome.focus({ preventScroll: true }), 60);
}

export async function formBaixa(p, aoSalvar) {
  const c = await cadastros();
  const rec = p.tipo === 'receita';
  modal(rec ? 'Registrar recebimento' : 'Registrar pagamento', `
    <div class="glass painel" style="padding:var(--s4)"><strong>${esc(p.nome)}</strong><p class="suave">Vencimento ${dataBR(p.vencimento)} · falta ${(p.aberto_cents / 100).toLocaleString('pt-BR', { style: 'currency', currency: 'BRL' })}</p></div>
    <div class="linha2"><label class="f">Data *<input class="campo" type="date" name="data" value="${hojeISO()}" required></label>
    <label class="f">Valor (R$) *<input class="campo" name="valor" value="${dinheiroInput(p.aberto_cents)}" inputmode="decimal" required><span class="dica">Pode ser parcial: o restante continua em aberto.</span></label></div>
    <label class="f">Conta bancária *<select class="campo" name="conta_id" required>${opcoes(c.contas, p.conta_id || ler('ultimaConta') || c.contas[0]?.id, 'Escolha…')}</select></label>`, {
    rotulo: 'Confirmar',
    onSubmit: async (d) => {
      const valor = parseDinheiro(d.valor);
      if (!valor) throw new Error('Valor inválido.');
      await api(`parcelas/${p.id}/baixa`, { method: 'POST', body: { data: d.data, valor_cents: valor, conta_id: d.conta_id } });
      guardar('ultimaConta', d.conta_id);
      toast(rec ? 'Recebimento registrado.' : 'Pagamento registrado.');
      aoSalvar?.();
    },
  });
}

export function formEditarParcela(p, aoSalvar) {
  modal('Editar parcela', `<div class="glass painel" style="padding:var(--s4)"><strong>${esc(p.nome)}</strong></div>
    <div class="linha2"><label class="f">Vencimento *<input class="campo" type="date" name="vencimento" value="${p.vencimento}" required></label>
    <label class="f">Valor (R$) *<input class="campo" name="valor" value="${dinheiroInput(p.valor_cents)}" inputmode="decimal" required></label></div>`, {
    onSubmit: async (d) => {
      const valor = parseDinheiro(d.valor);
      if (!valor) throw new Error('Valor inválido.');
      await api(`parcelas/${p.id}`, { method: 'PUT', body: { vencimento: d.vencimento, valor_cents: valor } });
      toast('Parcela atualizada.');
      aoSalvar?.();
    },
  });
}

export async function formTransferencia(aoSalvar) {
  const c = await cadastros();
  modal('Nova transferência', `
    <label class="f">Descrição *<input class="campo" name="descricao" required autocomplete="off" placeholder="Ex.: Reserva de caixa"></label>
    <div class="linha2"><label class="f">Valor (R$) *<input class="campo" name="valor" inputmode="decimal" required placeholder="0,00" autocomplete="off"></label>
    <label class="f">Data *<input class="campo" type="date" name="data" value="${hojeISO()}" required></label></div>
    <div class="linha2"><label class="f">De *<select class="campo" name="origem" required>${opcoes(c.contas, '', 'Escolha…')}</select></label>
    <label class="f">Para *<select class="campo" name="destino" required>${opcoes(c.contas, '', 'Escolha…')}</select></label></div>
    <p class="suave">Transferências entre contas próprias não entram no resultado nem na DRE.</p>`, {
    onSubmit: async (d) => {
      const valor = parseDinheiro(d.valor);
      if (!valor) throw new Error('Valor inválido.');
      await api('transferencias', { method: 'POST', body: { descricao: d.descricao, valor_cents: valor, data: d.data,
        conta_origem_id: d.origem, conta_destino_id: d.destino } });
      toast('Transferência registrada.');
      aoSalvar?.();
    },
  });
}

// Configuração dos cadastros: [campo, rótulo, tipo, opções]
export const CADASTROS = {
  contas: { titulo: 'Contas bancárias', singular: 'conta', campos: [['nome', 'Nome', 'text'], ['banco', 'Banco', 'text'], ['saldo_inicial_cents', 'Saldo inicial', 'money']],
    colunas: [['nome', 'Nome'], ['banco', 'Banco'], ['saldo_inicial_cents', 'Saldo inicial', 'money']] },
  categorias: { titulo: 'Categorias', singular: 'categoria', campos: [['nome', 'Nome', 'text'], ['tipo', 'Tipo', 'select', { receita: 'Receita', despesa: 'Despesa' }],
    ['grupo_dre', 'Grupo da DRE', 'select', GRUPOS]], colunas: [['nome', 'Nome'], ['tipo', 'Tipo'], ['grupo_dre', 'Grupo da DRE', 'grupo']] },
  centros: { titulo: 'Centros de custo', singular: 'centro de custo', campos: [['nome', 'Nome', 'text']], colunas: [['nome', 'Nome']] },
  pessoas: { titulo: 'Clientes e fornecedores', singular: 'pessoa', campos: [['codigo', 'Código (automático se vazio)', 'text'], ['nome', 'Nome', 'text'],
    ['tipo', 'Tipo', 'select', { cliente: 'Cliente', fornecedor: 'Fornecedor', ambos: 'Cliente e fornecedor' }], ['documento', 'CPF/CNPJ', 'text'], ['email', 'E-mail', 'text'], ['telefone', 'Telefone', 'text'],
      ['cidade', 'Cidade', 'text'], ['endereco', 'Endereço', 'text'], ['observacoes', 'Observações', 'textarea']],
    colunas: [['codigo', 'Código'], ['nome', 'Nome'], ['tipo', 'Tipo'], ['documento', 'CPF/CNPJ'], ['email', 'E-mail'], ['telefone', 'Telefone']] },
  servicos: { titulo: 'Serviços', singular: 'serviço', campos: [['nome', 'Nome', 'text'], ['descricao', 'Descrição', 'text'], ['valor_padrao_cents', 'Valor padrão', 'money'], ['categoria_id', 'Categoria da receita', 'categoria']],
    colunas: [['nome', 'Nome'], ['descricao', 'Descrição'], ['valor_padrao_cents', 'Valor padrão', 'money']] },
  contratos: { titulo: 'Projetos / contratos', singular: 'contrato', campos: [['codigo', 'Código', 'text'], ['nome', 'Nome', 'text'], ['pessoa_id', 'Cliente', 'pessoa'],
    ['valor_total_cents', 'Valor do contrato', 'money'], ['competencia', 'Data de competência', 'date'], ['status', 'Status', 'select', { ativo: 'Ativo', concluido: 'Concluído', cancelado: 'Cancelado' }]],
    colunas: [['codigo', 'Código'], ['nome', 'Nome'], ['pessoa_nome', 'Cliente'], ['valor_total_cents', 'Valor', 'money'], ['status', 'Status']] },
};

export async function formCadastro(tipo, reg, aoSalvar) {
  if (tipo === 'pessoas') { const { formPessoa } = await import('./clientes.js'); return formPessoa({ reg, aoSalvar }); }
  if (tipo === 'contratos') { const { formContrato } = await import('./clientes.js'); return formContrato({ reg, aoSalvar }); }
  const cfg = CADASTROS[tipo];
  const c = await cadastros();
  const campo = ([k, rot, t, ops]) => {
    const v = reg?.[k] ?? '';
    const obrig = k === 'nome' || k === 'codigo';
    let ctl;
    if (t === 'select') ctl = `<select class="campo" name="${k}">${Object.entries(ops).map(([val, r]) => `<option value="${val}"${v === val ? ' selected' : ''}>${esc(r)}</option>`).join('')}</select>`;
    else if (t === 'pessoa') ctl = `<select class="campo" name="${k}">${opcoes(c.pessoas.filter((p) => p.tipo !== 'fornecedor'), v, '—')}</select>`;
    else if (t === 'categoria') ctl = `<select class="campo" name="${k}">${opcoes(c.categorias.filter((x) => x.tipo === 'receita'), v, '—')}</select>`;
    else if (t === 'textarea') ctl = `<textarea class="campo" name="${k}" rows="2">${esc(v)}</textarea>`;
    else if (t === 'money') ctl = `<input class="campo" name="${k}" inputmode="decimal" value="${v === '' ? '' : dinheiroInput(v)}" placeholder="0,00">`;
    else ctl = `<input class="campo" type="${t === 'date' ? 'date' : 'text'}" name="${k}" value="${esc(v)}"${obrig ? ' required' : ''} autocomplete="off">`;
    return `<label class="f">${esc(rot)}${obrig ? ' *' : ''}${ctl}</label>`;
  };
  modal(`${reg ? 'Editar' : 'Novo(a)'} ${cfg.singular}`, cfg.campos.map(campo).join(''), {
    onSubmit: async (d) => {
      const body = {};
      for (const [k, , t] of cfg.campos) {
        if (t === 'money') { const v = parseDinheiro(d[k]); body[k] = v ?? 0; } else if (t === 'pessoa' || t === 'categoria') body[k] = d[k] || null;
        else body[k] = d[k];
      }
      await api(`cadastros/${tipo}${reg ? '/' + reg.id : ''}`, { method: reg ? 'PUT' : 'POST', body });
      limparCache();
      toast('Salvo.');
      aoSalvar?.();
    },
  });
}
