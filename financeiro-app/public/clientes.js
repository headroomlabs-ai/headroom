// Clientes: cadastro com código, serviços contratados e projeto, mais a tela de listagem.
import { $, $$, api, brl, dataBR, dinheiroInput, esc, hojeISO, modal, opcoes, parseDinheiro, qs, toast } from './util.js';
import { animarContadores, estadoVazio, gauge, icon, kpi, revelar, skeletonPagina } from './ui.js';
import { cadastros, formCadastro, formLancamento, limparCache } from './forms.js';

const pronto = (el) => { revelar(el); animarContadores(el); };
const pct = (a, b) => (b > 0 ? (a / b) * 100 : 0);
const fmtPct = (v) => `${v.toFixed(1).replace('.', ',')}%`;
const STATUS_PROJETO = { ativo: 'Ativo', concluido: 'Concluído', cancelado: 'Cancelado' };
const soDigitos = (v) => String(v || '').replace(/\D/g, '');

// ---------- editor de serviços contratados ----------
// Lista de linhas (nome + valor) com um seletor que puxa do catálogo. Devolve leitura dos valores e do total.
function editorServicos(raiz, catalogo, iniciais, aoMudar) {
  const linhas = iniciais.map((s) => ({ ...s }));
  const ativos = catalogo.filter((s) => s.ativo);
  const total = () => linhas.reduce((s, l) => s + (l.valor_cents || 0), 0);
  const atualizarTotal = () => { $('[data-total]', raiz).textContent = brl(total()); aoMudar?.(); };
  const desenhar = () => {
    raiz.innerHTML = `<div class="servicos-lista">${linhas.map((s, i) => `<div class="servico-linha" data-i="${i}">
        <input class="campo" data-nome value="${esc(s.nome)}" placeholder="Nome do serviço" aria-label="Serviço ${i + 1}" autocomplete="off">
        <input class="campo" data-valor inputmode="decimal" value="${s.valor_cents ? dinheiroInput(s.valor_cents) : ''}" placeholder="0,00" aria-label="Valor do serviço ${i + 1} em reais" autocomplete="off">
        <button type="button" class="icon-btn" data-rm aria-label="Remover serviço ${i + 1}">${icon('x')}</button></div>`).join('')}</div>
      <select class="campo" data-add aria-label="Adicionar serviço"><option value="">+ Adicionar serviço…</option>
        ${ativos.map((s) => `<option value="${s.id}">${esc(s.nome)}</option>`).join('')}<option value="__outro">Outro (digitar o nome)</option></select>
      <div class="servicos-total"><span class="suave">Total dos serviços</span><b class="num" data-total>${brl(total())}</b></div>`;
  };
  raiz.addEventListener('input', (e) => {
    const l = linhas[Number(e.target.closest('.servico-linha')?.dataset.i)];
    if (!l) return;
    if (e.target.matches('[data-nome]')) l.nome = e.target.value;
    if (e.target.matches('[data-valor]')) l.valor_cents = parseDinheiro(e.target.value) || 0;
    atualizarTotal();
  });
  raiz.addEventListener('click', (e) => {
    const b = e.target.closest('[data-rm]');
    if (!b) return;
    linhas.splice(Number(b.closest('.servico-linha').dataset.i), 1);
    desenhar(); aoMudar?.();
  });
  raiz.addEventListener('change', (e) => {
    if (!e.target.matches('[data-add]') || !e.target.value) return;
    const s = ativos.find((x) => x.id === e.target.value);
    linhas.push(s ? { servico_id: s.id, nome: s.nome, valor_cents: s.valor_padrao_cents || 0 } : { servico_id: null, nome: '', valor_cents: 0 });
    desenhar(); aoMudar?.();
    $$('[data-nome]', raiz).at(-1)?.focus();
  });
  desenhar();
  return {
    total,
    valores: () => linhas.filter((l) => l.nome.trim()).map((l) => ({ servico_id: l.servico_id || null, nome: l.nome.trim(), valor_cents: l.valor_cents || 0 })),
    nomes: () => linhas.filter((l) => l.nome.trim()).map((l) => l.nome.trim()),
    primeiraCategoria: () => linhas.map((l) => catalogo.find((s) => s.id === l.servico_id)?.categoria_id).find(Boolean) || '',
  };
}


const ESTADOS = ['AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MT', 'MS', 'MG', 'PA', 'PB', 'PR', 'PE', 'PI', 'RJ', 'RN', 'RS', 'RO', 'RR', 'SC', 'SP', 'SE', 'TO'];
const CAMPOS_END = ['cep', 'endereco', 'numero', 'complemento', 'bairro', 'cidade', 'estado'];

// Bloco de endereço; pref '' para o cliente e 'o_' para a obra.
const blocoEndereco = (pref, v = {}) => `
  <div class="linha2"><label class="f">CEP<input class="campo" name="${pref}cep" inputmode="numeric" value="${esc(v.cep ?? '')}" placeholder="00000-000" autocomplete="off"></label><div></div></div>
  <div class="linha-end"><label class="f">Endereço<input class="campo" name="${pref}endereco" value="${esc(v.endereco ?? '')}" autocomplete="off"></label>
    <label class="f">Número<input class="campo" name="${pref}numero" value="${esc(v.numero ?? '')}" autocomplete="off"></label></div>
  <div class="linha2"><label class="f">Complemento<input class="campo" name="${pref}complemento" value="${esc(v.complemento ?? '')}" placeholder="Apto 55" autocomplete="off"></label>
    <label class="f">Bairro<input class="campo" name="${pref}bairro" value="${esc(v.bairro ?? '')}" autocomplete="off"></label></div>
  <div class="linha-end"><label class="f">Cidade<input class="campo" name="${pref}cidade" value="${esc(v.cidade ?? '')}" autocomplete="off"></label>
    <label class="f">Estado<select class="campo" name="${pref}estado"><option value="">—</option>${ESTADOS.map((e) => `<option${v.estado === e ? ' selected' : ''}>${e}</option>`).join('')}</select></label></div>`;
// Ao digitar um CEP completo, preenche rua, bairro, cidade e estado (consulta pública ViaCEP). Só completa o que estiver vazio.
function ligarCep(f) {
  f.addEventListener('input', async (e) => {
    if (!e.target.name?.endsWith('cep')) return;
    const pref = e.target.name.slice(0, -3), dig = soDigitos(e.target.value);
    if (dig.length !== 8 || e.target.dataset.cep === dig) return;
    e.target.dataset.cep = dig;
    try {
      const r = await fetch(`https://viacep.com.br/ws/${dig}/json/`, { signal: AbortSignal.timeout(6000) });
      const j = await r.json();
      if (j.erro) return toast('CEP não encontrado. Preencha o endereço manualmente.', { tipo: 'info' });
      for (const [k, v] of [['endereco', j.logradouro], ['bairro', j.bairro], ['cidade', j.localidade], ['estado', j.uf]]) {
        const campo = f.elements[pref + k];
        if (campo && v && !campo.value) campo.value = v;
      }
    } catch { /* sem internet ou serviço fora: o endereço continua manual */ }
  });
}
const lerEndereco = (d, pref) => Object.fromEntries(CAMPOS_END.map((k) => [k, (d[pref + k] || '').trim() || null]));

// Dados pessoais: física/jurídica, documento, contato e NFS-e.
const blocoPessoa = (p = {}, sug = '') => {
  const nat = p.natureza || 'fisica';
  const cf = p.consumidor_final_nfse;
  return `
  <div class="linha2">
    <label class="f">Código do cliente *<input class="campo" name="codigo" value="${esc(p.codigo ?? sug)}" autocomplete="off"${p.tipo === 'fornecedor' ? '' : ' required'}><span class="dica">Sugerido automaticamente; pode editar.</span><span class="msg" data-msg="codigo"></span></label>
    <label class="f">Relação<select class="campo" name="tipo">${[['cliente', 'Cliente'], ['ambos', 'Cliente e fornecedor'], ['fornecedor', 'Fornecedor']].map(([k, r]) => `<option value="${k}"${(p.tipo || 'cliente') === k ? ' selected' : ''}>${r}</option>`).join('')}</select></label>
  </div>
  <div class="f"><span>Tipo</span><div class="segmento" role="radiogroup" aria-label="Tipo de pessoa">
    <label><input type="radio" name="natureza" value="fisica"${nat === 'fisica' ? ' checked' : ''}><span>Pessoa Física</span></label>
    <label><input type="radio" name="natureza" value="juridica"${nat === 'juridica' ? ' checked' : ''}><span>Pessoa Jurídica</span></label></div></div>
  <div class="linha2">
    <label class="f"><span><span data-rot-nome>Nome completo</span> *</span><input class="campo" name="nome" value="${esc(p.nome ?? '')}" autocomplete="off" required><span class="msg" data-msg="nome"></span></label>
    <label class="f" data-so-fisica>Data de nascimento<input class="campo" type="date" name="data_nascimento" value="${esc(p.data_nascimento ?? '')}"></label>
  </div>
  <div class="linha2">
    <label class="f"><span data-rot-doc>CPF</span><input class="campo" name="documento" inputmode="numeric" value="${esc(p.documento ?? '')}" autocomplete="off"><span class="msg" data-msg="documento"></span></label>
    <label class="f" data-so-fisica>RG<input class="campo" name="rg" value="${esc(p.rg ?? '')}" autocomplete="off"></label>
  </div>
  <div class="linha2">
    <label class="f">E-mail<input class="campo" type="email" name="email" value="${esc(p.email ?? '')}" autocomplete="off"></label>
    <div class="f"><span>Consumidor final para emissão de NFS-e</span><div class="radios">
      <label><input type="radio" name="consumidor_final" value="sim"${cf === true ? ' checked' : ''}><span>Sim</span></label>
      <label><input type="radio" name="consumidor_final" value="nao"${cf === false ? ' checked' : ''}><span>Não</span></label></div></div>
  </div>
  <div class="linha2"><label class="f">Telefone / WhatsApp<input class="campo" name="telefone" inputmode="tel" value="${esc(p.telefone ?? '')}" placeholder="(00) 00000-0000" autocomplete="off"></label><div></div></div>`;
};
// Troca rótulos e esconde campos que só valem para pessoa física.
function ligarNatureza(f) {
  const aplicar = () => {
    const pf = f.elements.natureza.value !== 'juridica';
    $$('[data-so-fisica]', f).forEach((e) => { e.hidden = !pf; });
    $('[data-rot-nome]', f).textContent = pf ? 'Nome completo' : 'Razão social';
    $('[data-rot-doc]', f).textContent = pf ? 'CPF' : 'CNPJ';
  };
  $$('[name=natureza]', f).forEach((r) => r.addEventListener('change', aplicar));
  aplicar();
}
const validarPessoa = (f) => {
  const erros = {};
  if (f.nome.value.trim().length < 2) erros.nome = 'Informe o nome (mínimo 2 letras).';
  if (f.codigo.required && !f.codigo.value.trim()) erros.codigo = 'Informe o código do cliente.';
  const dig = soDigitos(f.documento.value), pf = f.elements.natureza.value !== 'juridica';
  if (dig && dig.length !== (pf ? 11 : 14)) erros.documento = pf ? 'CPF tem 11 dígitos.' : 'CNPJ tem 14 dígitos.';
  for (const k of ['nome', 'codigo', 'documento']) { const m = $(`[data-msg="${k}"]`, f); if (m) m.textContent = erros[k] || ''; f[k].setAttribute('aria-invalid', erros[k] ? 'true' : 'false'); }
  const primeiro = Object.keys(erros)[0];
  if (primeiro) f[primeiro].focus();
  return !primeiro;
};
const lerPessoa = (d) => ({ codigo: d.codigo, nome: d.nome, tipo: d.tipo, natureza: d.natureza, data_nascimento: d.natureza === 'juridica' ? '' : d.data_nascimento,
  documento: d.documento, rg: d.natureza === 'juridica' ? '' : d.rg, email: d.email, telefone: d.telefone,
  consumidor_final_nfse: d.consumidor_final === 'sim' ? true : d.consumidor_final === 'nao' ? false : undefined, ...lerEndereco(d, ''), observacoes: d.observacoes });

// Endereço da obra: igual ao do cliente ou próprio.
const blocoObra = (obra, temCliente = true) => `
  <label class="switch"><input type="checkbox" name="obra_igual"${obra ? '' : ' checked'}><span class="trilho" aria-hidden="true"></span><span>${temCliente ? 'A obra fica no mesmo endereço do cliente' : 'Sem endereço de obra'}</span></label>
  <div class="empilha" data-obra-campos${obra ? '' : ' hidden'}><h3>Endereço da obra</h3>${blocoEndereco('o_', obra || {})}</div>`;
const ligarObra = (f) => { f.obra_igual.addEventListener('change', () => { $('[data-obra-campos]', f).hidden = f.obra_igual.checked; }); };
const lerObra = (d, doCliente) => (d.obra_igual ? doCliente : lerEndereco(d, 'o_'));

// ---------- cliente: edição completa (e cadastro avulso em Cadastros) ----------
export async function formPessoa({ reg = null, aoSalvar } = {}) {
  const sug = reg ? '' : (await api('sugestoes-codigo')).cliente;
  const m = modal(reg ? 'Editar cliente' : 'Novo cliente', `${blocoPessoa(reg || {}, sug)}
    <h3>Endereço do cliente</h3>${blocoEndereco('', reg || {})}
    <label class="f">Observações<textarea class="campo" name="observacoes" rows="3" placeholder="Observações">${esc(reg?.observacoes ?? '')}</textarea></label>`, {
    rotulo: 'Confirmar', classe: 'folha-larga',
    onSubmit: async (d) => {
      if (!validarPessoa(f)) throw new Error('Corrija os campos destacados.');
      await api(`cadastros/pessoas${reg ? '/' + reg.id : ''}`, { method: reg ? 'PUT' : 'POST', body: lerPessoa(d) });
      limparCache(); toast('Cliente salvo.'); aoSalvar?.();
    },
  });
  const f = $('form', m.dlg);
  ligarNatureza(f); ligarCep(f);
}

const camposProjeto = (c, p, sugestao) => `
  <div class="linha2">
    <label class="f">Código do projeto *<input class="campo" name="p_codigo" value="${esc(p?.codigo ?? sugestao ?? '')}" autocomplete="off" placeholder="CA261001"><span class="dica">Padrão: CA + ano + mês + número.</span></label>
    <label class="f">Nome do projeto *<input class="campo" name="p_nome" value="${esc(p?.nome ?? '')}" autocomplete="off" placeholder="Ex.: Residência Silva"></label>
  </div>
  <div class="linha2">
    <label class="f">Área (m²)<input class="campo" name="p_area" inputmode="decimal" value="${p?.area_m2 ?? ''}" placeholder="Ex.: 180" autocomplete="off"></label>
    <label class="f">Data de competência<input class="campo" type="date" name="p_competencia" value="${p?.competencia || hojeISO()}"></label>
  </div>
  ${blocoObra(p?.obra, true)}
  <div class="f"><span>Serviços contratados</span><div data-servicos class="empilha" style="gap:var(--s2)"></div></div>`;

function corpoProjeto(d, c, noContrato, endCliente = null) {
  const area = String(d.p_area || '').trim().replace(',', '.');
  return { codigo: d.p_codigo.trim(), nome: d.p_nome.trim(), area_m2: area === '' ? null : Number(area), competencia: d.p_competencia || hojeISO(), obra: lerObra(d, endCliente), ...noContrato };
}

// ---------- projeto / contrato (novo ou edição) ----------
export async function formContrato({ reg = null, pessoa_id = null, aoSalvar } = {}) {
  const c = await cadastros();
  const sug = reg ? null : (await api('sugestoes-codigo')).projeto;
  const clientes = c.pessoas.filter((p) => p.tipo !== 'fornecedor');
  const m = modal(reg ? 'Editar projeto' : 'Novo projeto', `
    <label class="f">Cliente<select class="campo" name="pessoa_id">${opcoes(clientes, reg?.pessoa_id ?? pessoa_id, '—')}</select></label>
    ${camposProjeto(c, reg, sug)}
    <div class="linha2">
      <label class="f" id="bloco-valor-manual">Valor do contrato (R$)<input class="campo" name="valor" inputmode="decimal" value="${reg?.valor_total_cents && !(reg.servicos || []).length ? dinheiroInput(reg.valor_total_cents) : ''}" placeholder="0,00"><span class="dica">Usado só quando não há serviços com valor.</span></label>
      <label class="f">Status<select class="campo" name="status">${Object.entries(STATUS_PROJETO).map(([k, r]) => `<option value="${k}"${(reg?.status || 'ativo') === k ? ' selected' : ''}>${r}</option>`).join('')}</select></label>
    </div>`, {
    onSubmit: async (d) => {
      if (!d.p_codigo.trim() || !d.p_nome.trim()) throw new Error('Informe o código e o nome do projeto.');
      const servicos = ed.valores();
      const cli = c.pessoas.find((x) => x.id === d.pessoa_id);
      const body = corpoProjeto(d, c, { pessoa_id: d.pessoa_id || null, status: d.status, servicos,
        valor_total_cents: servicos.length ? ed.total() : (parseDinheiro(d.valor) ?? 0) }, cli ? Object.fromEntries(CAMPOS_END.map((k) => [k, cli[k] || null])) : null);
      await api(`cadastros/contratos${reg ? '/' + reg.id : ''}`, { method: reg ? 'PUT' : 'POST', body });
      limparCache();
      toast('Projeto salvo.');
      aoSalvar?.();
    },
  });
  ligarObra($('form', m.dlg)); ligarCep($('form', m.dlg));
  const ed = editorServicos($('[data-servicos]', m.dlg), c.servicos, reg?.servicos || [], () => { $('#bloco-valor-manual', m.dlg).hidden = ed.valores().length > 0; });
  $('#bloco-valor-manual', m.dlg).hidden = ed.valores().length > 0;
}

// ---------- novo cliente em 3 etapas: dados → projeto e serviços → cobrança ----------
export async function formCliente(aoSalvar) {
  const c = await cadastros();
  const sug = await api('sugestoes-codigo');
  const cats = c.categorias.filter((x) => x.tipo === 'receita');
  const NOMES = ['Dados do cliente', 'Endereço', 'Projeto e serviços', 'Cobrança'];
  let pessoaId = null, contratoId = null, lancId = null; // permitem repetir o envio sem duplicar o que já foi criado

  const corpo = `
    <div class="passos" aria-hidden="true" style="padding:0">${NOMES.map((n, i) => `<div class="passo" data-ind="${i + 1}"><i></i>${i + 1}. ${n}</div>`).join('')}</div>
    <section data-passo="1" class="empilha" aria-label="Etapa 1: Dados do cliente">${blocoPessoa({}, sug.cliente)}</section>
    <section data-passo="2" class="empilha" aria-label="Etapa 2: Endereço do cliente" hidden>${blocoEndereco('', {})}
      <label class="f">Observações<textarea class="campo" name="observacoes" rows="3" placeholder="Observações"></textarea></label></section>
    <section data-passo="3" class="empilha" aria-label="Etapa 3: Projeto e serviços" hidden>
      <p class="suave">Opcional: deixe em branco se ainda não fechou projeto com este cliente.</p>
      ${camposProjeto(c, null, sug.projeto)}
    </section>
    <section data-passo="4" class="empilha" aria-label="Etapa 4: Cobrança" hidden>
      <div id="sem-cobranca" class="suave"></div>
      <div id="bloco-cobranca" class="empilha">
        <label class="switch"><input type="checkbox" name="gerar" checked><span class="trilho" aria-hidden="true"></span><span>Já gerar a receita (parcelas a receber) deste projeto</span></label>
        <div id="campos-cobranca" class="empilha">
          <div class="linha2">
            <label class="f">Parcelas<input class="campo" type="number" name="parcelas" min="1" max="360" value="1" inputmode="numeric"></label>
            <label class="f">Primeiro vencimento<input class="campo" type="date" name="primeiro_vencimento" value="${hojeISO()}"></label>
          </div>
          <div class="linha2">
            <label class="f">Categoria<select class="campo" name="categoria_id">${opcoes(cats, '', 'Escolha…')}</select></label>
            <label class="f">Conta de recebimento<select class="campo" name="conta_id">${opcoes(c.contas, c.contas[0]?.id, '—')}</select></label>
          </div>
        </div>
      </div>
      <div class="glass painel" id="resumo-cli" style="padding:var(--s4)"></div>
    </section>`;
  const rodape = `<button type="button" class="btn btn-ghost" id="passo-voltar" data-fechar>Cancelar</button>
    <button type="button" class="btn btn-primary" id="passo-seguir">Continuar</button>
    <button type="submit" class="btn btn-primary" id="passo-criar" hidden>${icon('check')}Cadastrar cliente</button>`;

  let ed;
  const m = modal('Novo cliente', corpo, {
    rotulo: 'Cadastrar cliente', rodape, classe: 'folha-passos',
    onSubmit: async (d) => {
      const temProjeto = !!(d.p_nome.trim() || ed.valores().length);
      if (!pessoaId) {
        const r = await api('cadastros/pessoas', { method: 'POST', body: lerPessoa(d) });
        pessoaId = r.id; limparCache();
      }
      if (temProjeto && !contratoId) {
        const servicos = ed.valores();
        const r = await api('cadastros/contratos', { method: 'POST', body: corpoProjeto(d, c, { pessoa_id: pessoaId, status: 'ativo', servicos, valor_total_cents: ed.total() }, lerEndereco(d, '')) });
        contratoId = r.id; limparCache();
      }
      const total = ed.total();
      if (temProjeto && d.gerar && total > 0 && !lancId) {
        const parcelas = Number(d.parcelas || 1);
        const r = await api('lancamentos', { method: 'POST', body: { tipo: 'receita', nome: `${ed.nomes().join(' + ') || 'Projeto'} — ${d.p_nome.trim()}`, valor_total_cents: total,
          primeiro_vencimento: d.primeiro_vencimento, competencia: d.p_competencia || d.primeiro_vencimento, parcelas, pessoa_id: pessoaId, contrato_id: contratoId,
          categoria_id: d.categoria_id || null, conta_id: d.conta_id || null } });
        lancId = r.id;
      }
      toast(`Cliente ${d.nome} cadastrado.`);
      aoSalvar?.();
    },
  });
  const f = $('form', m.dlg);
  ed = editorServicos($('[data-servicos]', m.dlg), c.servicos, [], () => { if (passo === 4) atualizarResumo(); });
  let passo = 1;

  function atualizarResumo() {
    const total = ed.total(), temProjeto = !!(f.p_nome.value.trim() || ed.valores().length);
    const cobrar = temProjeto && total > 0;
    $('#bloco-cobranca', f).hidden = !cobrar;
    $('#campos-cobranca', f).hidden = !cobrar || !f.gerar.checked;
    $('#sem-cobranca', f).textContent = cobrar ? '' : (temProjeto ? 'Nenhum serviço com valor: não há o que cobrar agora. Você poderá lançar a receita depois.' : 'Sem projeto neste cadastro: nada a cobrar agora.');
    if (cobrar && !f.categoria_id.value) f.categoria_id.value = ed.primeiraCategoria();
    $('#resumo-cli', f).innerHTML = `<h3>Resumo</h3><ul class="lista">
      <li><span><b>${esc(f.codigo.value)}</b> · ${esc(f.nome.value || '—')}</span></li>
      ${temProjeto ? `<li><span>${esc(f.p_codigo.value)} · ${esc(f.p_nome.value || '—')}${f.p_area.value ? ` · ${esc(f.p_area.value)} m²` : ''}</span><b class="num">${brl(total)}</b></li>
      <li><span class="suave">${esc(ed.nomes().join(', ') || 'Sem serviços informados')}</span></li>` : '<li><span class="suave">Somente o cadastro do cliente.</span></li>'}
      ${cobrar && f.gerar.checked ? `<li><span class="suave">${f.parcelas.value || 1} parcela(s) a partir de ${dataBR(f.primeiro_vencimento.value)}</span></li>` : ''}</ul>`;
  }
  const validar = {
    1: () => validarPessoa(f),
    2: () => true,
    3: () => {
      const tem = f.p_nome.value.trim() || ed.valores().length;
      if (tem && !f.p_nome.value.trim()) { toast('Informe o nome do projeto.', 'erro'); f.p_nome.focus(); return false; }
      if (tem && !f.p_codigo.value.trim()) { toast('Informe o código do projeto.', 'erro'); f.p_codigo.focus(); return false; }
      return true;
    },
    4: () => true,
  };
  const ir = (n) => {
    passo = n;
    $$('[data-passo]', f).forEach((s) => { s.hidden = Number(s.dataset.passo) !== n; });
    $$('.passo', f).forEach((p) => { const i = Number(p.dataset.ind); p.classList.toggle('atual', i === n); p.classList.toggle('feito', i < n); });
    $('#passo-seguir', f).hidden = n === 4;
    $('#passo-criar', f).hidden = n !== 4;
    const voltar = $('#passo-voltar', f);
    voltar.textContent = n === 1 ? 'Cancelar' : 'Voltar';
    if (n === 1) voltar.setAttribute('data-fechar', ''); else voltar.removeAttribute('data-fechar');
    $('.erro', m.dlg).textContent = '';
    if (n === 4) atualizarResumo();
    const primeiro = $(`[data-passo="${n}"] .campo`, f);
    if (primeiro) setTimeout(() => primeiro.focus({ preventScroll: true }), 30);
  };
  $('#passo-seguir', f).onclick = () => { if (validar[passo]()) ir(passo + 1); };
  $('#passo-voltar', f).onclick = () => { if (passo > 1) ir(passo - 1); else m.fechar(); };
  f.addEventListener('submit', (e) => { if (passo < 4) { e.preventDefault(); e.stopImmediatePropagation(); if (validar[passo]()) ir(passo + 1); } }, true);
  ligarNatureza(f); ligarObra(f); ligarCep(f);
  f.gerar.onchange = atualizarResumo;
  f.parcelas.oninput = f.primeiro_vencimento.onchange = atualizarResumo;
  ir(1);
}

// ---------- ficha do cliente ----------
const fmtEnd = (e) => (e ? [[e.endereco, e.numero].filter(Boolean).join(', '), e.complemento, e.bairro, [e.cidade, e.estado].filter(Boolean).join('/'), e.cep && `CEP ${e.cep}`].filter(Boolean).map(esc).join(' · ') : '');
function fichaCliente(cli, depois) {
  const linhasProj = cli.projetos.map((p) => `<div class="glass painel" style="padding:var(--s4)">
    <div style="display:flex;justify-content:space-between;gap:var(--s3);flex-wrap:wrap"><strong>${esc(p.codigo)} · ${esc(p.nome)}</strong><span class="chip">${esc(STATUS_PROJETO[p.status] || p.status)}</span></div>
    <p class="suave" style="margin:var(--s1) 0 var(--s2)">${p.area_m2 ? `${String(p.area_m2).replace('.', ',')} m² · ` : ''}${p.competencia ? 'competência ' + dataBR(p.competencia) + ' · ' : ''}<b class="num">${brl(p.valor_total_cents)}</b></p>
    ${p.obra ? `<p class="suave" style="margin-bottom:var(--s2)">Obra: ${fmtEnd(p.obra)}</p>` : ''}
    ${p.servicos.length ? `<ul class="lista">${p.servicos.map((s) => `<li><span>${esc(s.nome)}</span><b class="num">${brl(s.valor_cents)}</b></li>`).join('')}</ul>` : '<p class="suave">Sem serviços informados.</p>'}
    <div style="margin-top:var(--s3);display:flex;gap:var(--s2);flex-wrap:wrap"><button type="button" class="btn btn-sm" data-edit-proj="${p.id}">${icon('editar')}Editar</button>
      <button type="button" class="btn btn-sm" data-receita="${p.id}">${icon('plus')}Nova receita</button></div></div>`).join('');
  const contato = [cli.telefone, cli.email, cli.documento && `${cli.natureza === 'juridica' ? 'CNPJ' : 'CPF'} ${cli.documento}`, cli.rg && `RG ${cli.rg}`].filter(Boolean).map(esc).join(' · ');
  const m = modal(`${cli.codigo ? cli.codigo + ' · ' : ''}${cli.nome}`, `
    <p class="suave">${contato || 'Sem dados de contato.'}${fmtEnd(cli) ? `<br>${icon('alvo')}${fmtEnd(cli)}` : ''}</p>${cli.observacoes ? `<p>${esc(cli.observacoes)}</p>` : ''}
    <div class="grade-kpi" style="grid-template-columns:repeat(2,1fr)">
      <div class="glass painel" style="padding:var(--s3)"><span class="suave">Contratado</span><br><b class="num">${brl(cli.contratado_cents)}</b></div>
      <div class="glass painel" style="padding:var(--s3)"><span class="suave">Recebido</span><br><b class="num verde">${brl(cli.pago_cents)}</b></div>
      <div class="glass painel" style="padding:var(--s3)"><span class="suave">A vencer</span><br><b class="num laranja">${brl(cli.aberto_cents)}</b></div>
      <div class="glass painel" style="padding:var(--s3)"><span class="suave">Vencido</span><br><b class="num ${cli.vencido_cents ? 'vermelho' : ''}">${brl(cli.vencido_cents)}</b></div></div>
    <h3 style="margin-top:var(--s2)">Projetos e serviços</h3>${linhasProj || '<p class="suave">Nenhum projeto cadastrado para este cliente.</p>'}`, {
    rodape: `<button type="button" class="btn btn-ghost" data-fechar>Fechar</button><button type="button" class="btn" id="cli-editar">${icon('editar')}Editar cliente</button><button type="button" class="btn btn-primary" id="cli-projeto">${icon('plus')}Novo projeto</button>`,
  });
  const abrir = (fn) => { m.fechar(); fn(); };
  $('#cli-editar', m.dlg).onclick = () => abrir(() => formCadastro('pessoas', cli, depois));
  $('#cli-projeto', m.dlg).onclick = () => abrir(() => formContrato({ pessoa_id: cli.id, aoSalvar: depois }));
  m.dlg.addEventListener('click', (e) => {
    const ed = e.target.closest('[data-edit-proj]'), rc = e.target.closest('[data-receita]');
    if (ed) abrir(() => formContrato({ reg: cli.projetos.find((p) => p.id === ed.dataset.editProj), aoSalvar: depois }));
    if (rc) abrir(() => formLancamento('receita', depois, { pessoa_id: cli.id, contrato_id: rc.dataset.receita }));
  });
}

// ---------- tela Clientes ----------
export async function clientesView(el) {
  const filtro = clientesView.filtro ||= { busca: '' };
  window.dispatchEvent(new CustomEvent('crumbs', { detail: ['Financeiro', 'Clientes'] }));
  if (el.dataset.tela !== 'cli') el.innerHTML = skeletonPagina();
  const r = await api(`clientes?${qs(filtro)}`);
  el.dataset.tela = 'cli';
  const t = r.total;
  const perc = pct(t.pago_cents, t.lancado_cents);
  const chips = (c) => (c.servicos.length ? c.servicos.slice(0, 3).map((s) => `<span class="chip tag">${esc(s)}</span>`).join(' ') + (c.servicos.length > 3 ? ` <span class="chip">+${c.servicos.length - 3}</span>` : '') : '<span class="suave">—</span>');
  const linhas = r.itens.map((c) => `<tr class="clicavel" data-cli="${c.id}" tabindex="0" role="button" aria-label="Abrir ficha de ${esc(c.nome)}">
    <td data-label="Código"><strong>${esc(c.codigo || '—')}</strong></td>
    <td class="nome" data-label="Cliente"><strong>${esc(c.nome)}</strong><small>${esc([c.telefone, [c.cidade, c.estado].filter(Boolean).join('/')].filter(Boolean).join(' · '))}</small></td>
    <td data-label="Serviços contratados">${chips(c)}</td><td class="num" data-label="Projetos">${c.projetos.length}</td>
    <td class="num" data-label="Contratado"><b>${brl(c.contratado_cents)}</b></td><td class="num verde" data-label="Recebido">${brl(c.pago_cents)}</td>
    <td class="num laranja" data-label="A vencer">${brl(c.aberto_cents)}</td><td class="num ${c.vencido_cents ? 'vermelho' : ''}" data-label="Vencido">${brl(c.vencido_cents)}</td></tr>`).join('');
  el.innerHTML = `
    <div class="filtros"><label class="pilula-vidro busca-campo">${icon('search')}<input class="campo" type="search" name="busca" placeholder="Pesquisar cliente, código, projeto ou serviço" value="${esc(filtro.busca)}" aria-label="Pesquisar clientes"></label></div>
    <div class="grade g-destaque">
      <section class="glass painel hero mira reveal" style="display:grid;place-items:center">${gauge({ pct: perc, rotulo: 'Recebido', valor: `<span data-count="${perc}" data-fmt="pct">${fmtPct(perc)}</span>`, sub: `${brl(t.pago_cents)} de ${brl(t.lancado_cents)} lançados`, cor: 'amber', ariaLabel: `${fmtPct(perc)} do valor lançado já recebido` })}</section>
      <div class="grade-kpi" style="align-content:center">${kpi({ rotulo: 'Clientes', icone: 'usuarios', valor: t.qtd, cor: 'azul', formato: 'int' })}${kpi({ rotulo: 'Total contratado', icone: 'contrato', valor: t.contratado_cents, cor: 'violet' })}${kpi({ rotulo: 'A receber (a vencer)', icone: 'relogio', valor: t.aberto_cents, cor: 'laranja' })}${kpi({ rotulo: 'Vencido', icone: 'alert', valor: t.vencido_cents, cor: 'vermelho' })}</div>
    </div>
    ${r.itens.length ? `<section class="glass reveal" style="margin-top:var(--s4)"><div class="tabela-wrap"><table class="tbl"><thead><tr><th>Código</th><th>Cliente</th><th>Serviços contratados</th><th class="num">Projetos</th><th class="num">Contratado</th><th class="num">Recebido</th><th class="num">A vencer</th><th class="num">Vencido</th></tr></thead><tbody>${linhas}</tbody></table></div></section>`
    : estadoVazio({ titulo: filtro.busca ? 'Nenhum cliente encontrado' : 'Nenhum cliente cadastrado', texto: filtro.busca ? 'Tente outro nome, código ou serviço.' : 'Cadastre o cliente com código, dados de contato e os serviços que ele contratou. Depois, é só lançar as receitas.', acaoRotulo: 'Novo cliente', acaoId: 'estado-novo' })}`;
  const recarregar = () => clientesView(el);
  let tmr; $('[name=busca]', el).oninput = (e) => { clearTimeout(tmr); tmr = setTimeout(() => { filtro.busca = e.target.value; clientesView(el).then(() => { const b = $('[name=busca]', el); b.focus(); b.setSelectionRange(b.value.length, b.value.length); }); }, 350); };
  const abrir = (tr) => { const cli = r.itens.find((x) => x.id === tr.dataset.cli); if (cli) fichaCliente(cli, recarregar); };
  el.onclick = (e) => {
    if (e.target.closest('#estado-novo')) return formCliente(recarregar).catch((err) => toast(err.message, 'erro'));
    const tr = e.target.closest('[data-cli]'); if (tr) abrir(tr);
  };
  el.onkeydown = (e) => { const tr = e.target.closest?.('[data-cli]'); if (tr && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); abrir(tr); } };
  pronto(el);
}
