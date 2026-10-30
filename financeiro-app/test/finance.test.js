import test from 'node:test';
import assert from 'node:assert/strict';
import {
  criarDados, criarLancamento, baixarParcela, estornarParcela, editarParcela, dividirCentavos, resumo, fluxoCaixa, dre, listarParcelas,
  criarTransferencia, saldoContas, extratoConta, ErroValidacao, resultadosGerais, resultadosPorProjeto, outrosRelatorios,
  validarCadastro, emUso, pagamentosCliente, clientes, sugestoesCodigo, aReceber,
} from '../api/_lib/finance.js';

const HOJE = '2026-06-15';

function base() {
  return criarDados({
    contas: [{ id: 'c1', nome: 'Inter', saldo_inicial_cents: 100000, ativa: 1 }, { id: 'c2', nome: 'Nubank', saldo_inicial_cents: 0, ativa: 1 }],
    categorias: [
      { id: 'k1', nome: 'Projetos', tipo: 'receita', grupo_dre: 'receita_bruta' }, { id: 'k2', nome: 'Impostos', tipo: 'despesa', grupo_dre: 'deducoes' },
      { id: 'k3', nome: 'Terceirizados', tipo: 'despesa', grupo_dre: 'custos_operacionais' }, { id: 'k4', nome: 'Aluguel', tipo: 'despesa', grupo_dre: 'despesas_operacionais' },
      { id: 'k5', nome: 'Juros', tipo: 'despesa', grupo_dre: 'financeiras' }],
    pessoas: [{ id: 'p1', nome: 'Cliente A', tipo: 'cliente' }, { id: 'p2', nome: 'Fornecedor B', tipo: 'fornecedor' }],
    contratos: [{ id: 't1', codigo: 'CA260101', nome: 'Casa A', pessoa_id: 'p1' }],
  });
}
const incluir = (d, l) => { d.lancamentos.push(l); d.mapa.lancamentos.set(l.id, l); return l; };
const trocar = (d, l) => { d.lancamentos.splice(d.lancamentos.findIndex((x) => x.id === l.id), 1, l); d.mapa.lancamentos.set(l.id, l); d._exp = null; };
const rec = (d, o) => incluir(d, criarLancamento(d, { tipo: 'receita', nome: 'R', categoria_id: 'k1', conta_id: 'c1', ...o }));
const desp = (d, o) => incluir(d, criarLancamento(d, { tipo: 'despesa', nome: 'D', categoria_id: 'k4', conta_id: 'c1', ...o }));
const pagar = (d, l, n, dados = {}) => trocar(d, baixarParcela(d, `${l.id}:${n}`, dados));

test('divide centavos sem perder nem criar dinheiro', () => {
  assert.deepEqual(dividirCentavos(10000, 3), [3334, 3333, 3333]);
  assert.equal(dividirCentavos(99999, 7).reduce((a, b) => a + b, 0), 99999);
});

test('parcelado: parcelas somam o total e a competência fica na venda', () => {
  const d = base();
  const l = rec(d, { nome: 'Projeto', valor_total_cents: 100000, parcelas: 3, primeiro_vencimento: '2026-01-31', competencia: '2026-01-10' });
  assert.deepEqual(l.parcelas.map((p) => p.vencimento), ['2026-01-31', '2026-02-28', '2026-03-31']);
  assert.equal(l.parcelas.reduce((s, p) => s + p.valor_cents, 0), 100000);
  assert.ok(l.parcelas.every((p) => p.competencia === '2026-01-10'));
  assert.equal(listarParcelas(d, { de: '2026-01-01', ate: '2026-12-31' }, HOJE).itens[1].nome, 'Projeto 2/3');
});

test('recorrente: cada ocorrência tem o valor informado e a competência do seu mês', () => {
  const d = base();
  const l = desp(d, { nome: 'Aluguel', valor_total_cents: 5000, recorrente: true, repeticoes: 3, primeiro_vencimento: '2026-05-05' });
  assert.equal(l.parcelas.length, 3);
  assert.ok(l.parcelas.every((p) => p.valor_cents === 5000));
  assert.deepEqual(l.parcelas.map((p) => p.competencia.slice(0, 7)), ['2026-05', '2026-06', '2026-07']);
});

test('"em aberto" soma a vencer + vence hoje e deixa os vencidos de fora; balanço = recebido − pago', () => {
  const d = base();
  rec(d, { nome: 'Vencida', valor_total_cents: 5000, primeiro_vencimento: '2026-06-01' });
  rec(d, { nome: 'Hoje', valor_total_cents: 1000, primeiro_vencimento: HOJE });
  rec(d, { nome: 'Futura', valor_total_cents: 20000, primeiro_vencimento: '2026-08-01' });
  rec(d, { nome: 'Paga', valor_total_cents: 30000, primeiro_vencimento: '2026-03-01', primeira_paga: true });
  desp(d, { nome: 'Conta paga', valor_total_cents: 12000, primeiro_vencimento: '2026-04-01', primeira_paga: true });
  const r = resumo(d, { de: '2026-01-01', ate: '2026-12-31' }, HOJE);
  assert.deepEqual(r.receitas.em_aberto, { valor: 21000, qtd: 2 });
  assert.deepEqual(r.receitas.vencido, { valor: 5000, qtd: 1 });
  assert.deepEqual(r.receitas.realizado, { valor: 30000, qtd: 1 });
  assert.equal(r.balanco_cents, 18000);
  const f = listarParcelas(d, { tipo: 'receita', de: '2026-01-01', ate: '2026-12-31' }, HOJE).faixas;
  assert.equal(f.vencidos.valor + f.vence_hoje.valor + f.a_vencer.valor + f.pagos.valor, f.total.valor);
  assert.equal(f.total.qtd, 4);
});

test('pagamento parcial: continua em aberto só pelo que falta, o total fecha e cada pagamento fica na sua conta', () => {
  const d = base();
  const l = rec(d, { nome: 'Parcial', valor_total_cents: 10000, primeiro_vencimento: '2026-07-01' });
  pagar(d, l, 1, { data: '2026-06-10', valor_cents: 4000 });
  const fx = listarParcelas(d, { tipo: 'receita', de: '2026-01-01', ate: '2026-12-31' }, HOJE).faixas;
  assert.equal(fx.a_vencer.valor, 6000); assert.equal(fx.pagos.valor, 4000); assert.equal(fx.total.valor, 10000);
  assert.throws(() => baixarParcela(d, `${l.id}:1`, { valor_cents: 6001 }), ErroValidacao);
  pagar(d, l, 1, { data: '2026-06-12', conta_id: 'c2' });
  const s = Object.fromEntries(saldoContas(d, null, HOJE).map((c) => [c.nome, c.saldo_cents]));
  assert.deepEqual(s, { Inter: 104000, Nubank: 6000 });
  assert.throws(() => baixarParcela(d, `${l.id}:1`), /já está paga/);
  assert.throws(() => editarParcela(d, `${l.id}:1`, { valor_cents: 1 }), /Estorne/);
  trocar(d, estornarParcela(d, `${l.id}:1`));
  trocar(d, editarParcela(d, `${l.id}:1`, { valor_cents: 9000 }));
  assert.equal(d.mapa.lancamentos.get(l.id).parcelas[0].valor_cents, 9000);
});

test('baixar não altera o objeto original até ser gravado (cópia)', () => {
  const d = base();
  const l = rec(d, { valor_total_cents: 1000, primeiro_vencimento: '2026-07-01' });
  const novo = baixarParcela(d, `${l.id}:1`, { data: '2026-07-01' });
  assert.equal(l.parcelas[0].pagamentos.length, 0);
  assert.equal(novo.parcelas[0].pagamentos.length, 1);
});

test('fluxo de caixa: saldo final de um mês é o saldo inicial do seguinte (previsto e realizado)', () => {
  const d = base();
  const a = rec(d, { nome: 'A', valor_total_cents: 50000, primeiro_vencimento: '2026-01-10', primeira_paga: true });
  desp(d, { nome: 'B', valor_total_cents: 10000, primeiro_vencimento: '2026-01-15', primeira_paga: true });
  rec(d, { nome: 'C', valor_total_cents: 20000, primeiro_vencimento: '2026-02-10' }); // só previsto
  void a;
  const fx = fluxoCaixa(d, 2026, HOJE);
  assert.equal(fx.saldo_inicial.realizado[0], 100000);
  assert.equal(fx.saldo_inicial.realizado[1], 140000);
  assert.equal(fx.saldo_inicial.previsto[2], 160000);
  assert.equal(fx.saldo_inicial.realizado[2], 140000);
  assert.equal(fx.receitas.total.previsto[1], 20000);
  assert.equal(fx.receitas.total.realizado[1], 0);
});

test('fluxo de caixa: o que foi pago antes do ano entra no saldo inicial do ano', () => {
  const d = base();
  rec(d, { valor_total_cents: 30000, primeiro_vencimento: '2025-12-10', primeira_paga: true });
  assert.equal(fluxoCaixa(d, 2026, HOJE).saldo_inicial.realizado[0], 130000);
});

test('DRE: estrutura e margens sobre a receita bruta (números do relatório de referência)', () => {
  const d = base();
  const c = (o) => ({ primeiro_vencimento: '2026-01-10', competencia: '2026-01-10', primeira_paga: true, ...o });
  rec(d, c({ nome: 'Receita', valor_total_cents: 1675380 }));
  desp(d, c({ nome: 'Impostos', categoria_id: 'k2', valor_total_cents: 196225 }));
  desp(d, c({ nome: 'Terceiros', categoria_id: 'k3', valor_total_cents: 312847 }));
  desp(d, c({ nome: 'Despesas', categoria_id: 'k4', valor_total_cents: 3266184 }));
  const L = Object.fromEntries(dre(d, 2026, {}, HOJE).linhas.map((l) => [l.chave, l]));
  assert.equal(L.receita_liquida.valores[0], 1479155);
  assert.equal(L.resultado_bruto.valores[0], 1166308);
  assert.equal(L.resultado_operacional.valores[0], -2099876);
  assert.equal(L.margem_bruta.valores[0].toFixed(2), '69.61');
  assert.equal(L.margem_operacional.valores[0].toFixed(2), '-125.34');
  assert.equal(L.margem_bruta.valores[1], null);
});

test('DRE: financeiras e não operacionais entram só no resultado líquido', () => {
  const d = base();
  const c = { primeiro_vencimento: '2026-03-10', primeira_paga: true };
  rec(d, { ...c, valor_total_cents: 100000 });
  desp(d, { ...c, categoria_id: 'k5', valor_total_cents: 10000 });
  const L = Object.fromEntries(dre(d, 2026, {}, HOJE).linhas.map((l) => [l.chave, l]));
  assert.equal(L.resultado_operacional.valores[2], 100000);
  assert.equal(L.resultado_liquido.valores[2], 90000);
});

test('DRE por competência × caixa usam datas diferentes', () => {
  const d = base();
  rec(d, { nome: 'Venda', valor_total_cents: 10000, primeiro_vencimento: '2026-03-10', competencia: '2026-01-05', primeira_paga: true });
  const comp = dre(d, 2026, { regime: 'competencia' }, HOJE).linhas[0].valores;
  const caixa = dre(d, 2026, { regime: 'caixa' }, HOJE).linhas[0].valores;
  assert.equal(comp[0], 10000); assert.equal(comp[2], 0);
  assert.equal(caixa[0], 0); assert.equal(caixa[2], 10000);
});

test('transferência não altera resultado, só move saldo entre contas, e aparece no extrato', () => {
  const d = base();
  d.transferencias.push(criarTransferencia(d, { descricao: 'Reserva', valor_cents: 30000, conta_origem_id: 'c1', conta_destino_id: 'c2', data: '2026-02-01' }));
  const s = Object.fromEntries(saldoContas(d, null, HOJE).map((c) => [c.nome, c.saldo_cents]));
  assert.deepEqual(s, { Inter: 70000, Nubank: 30000 });
  assert.equal(resultadosGerais(d, { de: '2026-01-01', ate: '2026-12-31' }, HOJE).resultado.valor, 0);
  assert.equal(extratoConta(d, 'c2', HOJE).saldo_cents, 30000);
  assert.throws(() => criarTransferencia(d, { descricao: 'x', valor_cents: 1, conta_origem_id: 'c1', conta_destino_id: 'c1', data: '2026-02-01' }), /diferentes/);
  assert.throws(() => criarTransferencia(d, { descricao: 'x', valor_cents: 1, conta_origem_id: 'c1', conta_destino_id: 'zzz', data: '2026-02-01' }), /não encontrada/);
});

test('validações: categoria de tipo errado, valor zero, parcelas inválidas, baixa sem conta e referências inexistentes', () => {
  const d = base();
  assert.throws(() => criarLancamento(d, { tipo: 'despesa', nome: 'x', categoria_id: 'k1', valor_total_cents: 100, primeiro_vencimento: '2026-01-01' }), /receita/);
  assert.throws(() => rec(d, { valor_total_cents: 0, primeiro_vencimento: '2026-01-01' }), /valor/i);
  assert.throws(() => rec(d, { valor_total_cents: 100, parcelas: 0, primeiro_vencimento: '2026-01-01' }), /parcelas/);
  assert.throws(() => rec(d, { valor_total_cents: 100, pessoa_id: 'nao-existe', primeiro_vencimento: '2026-01-01' }), /não encontrado/);
  const l = incluir(d, criarLancamento(d, { tipo: 'receita', nome: 'x', valor_total_cents: 100, primeiro_vencimento: '2026-01-01' }));
  assert.throws(() => baixarParcela(d, `${l.id}:1`, {}), /conta/);
  assert.throws(() => baixarParcela(d, 'nada:1', {}), /não encontrada/);
});

test('resultado por projeto e outros relatórios', () => {
  const d = base();
  rec(d, { nome: 'P1', contrato_id: 't1', pessoa_id: 'p1', valor_total_cents: 100000, primeiro_vencimento: '2026-02-01', primeira_paga: true });
  desp(d, { nome: 'Eng', contrato_id: 't1', pessoa_id: 'p2', categoria_id: 'k3', valor_total_cents: 30000, primeiro_vencimento: '2026-02-05', primeira_paga: true });
  rec(d, { nome: 'P2', contrato_id: 't1', pessoa_id: 'p1', valor_total_cents: 50000, primeiro_vencimento: '2026-09-01' });
  const periodo = { de: '2026-01-01', ate: '2026-12-31' };
  const real = resultadosPorProjeto(d, periodo, HOJE);
  assert.deepEqual([real.itens[0].receitas_cents, real.itens[0].despesas_cents, real.itens[0].resultado_cents], [100000, 30000, 70000]);
  assert.equal(resultadosPorProjeto(d, { ...periodo, previsto: true }, HOJE).itens[0].receitas_cents, 50000);
  const o = outrosRelatorios(d, { ...periodo, agrupar: 'cliente', tipo: 'receita' }, HOJE);
  assert.equal(o.itens[0].nome, 'Cliente A');
  assert.deepEqual([o.total.valor_cents, o.total.pago_cents, o.total.aberto_cents], [150000, 100000, 50000]);
  const pc = pagamentosCliente(d, periodo, HOJE);
  assert.equal(pc.length, 2); assert.equal(pc[0].parcelas.length, 1);
});

test('cadastros: categoria com grupo incoerente é recusada; item em uso não pode ser excluído', () => {
  const d = base();
  assert.throws(() => validarCadastro(d, 'categorias', { nome: 'x', tipo: 'despesa', grupo_dre: 'receita_bruta' }), /receita bruta/);
  assert.throws(() => validarCadastro(d, 'categorias', { nome: 'x', tipo: 'receita', grupo_dre: 'custos_operacionais' }), /Receita não pode/);
  assert.throws(() => validarCadastro(d, 'contas', { nome: ' ' }), /nome/i);
  assert.equal(validarCadastro(d, 'contas', { nome: 'Caixa', saldo_inicial_cents: 500 }).ativa, 1);
  rec(d, { valor_total_cents: 100, primeiro_vencimento: '2026-01-01', pessoa_id: 'p1', contrato_id: 't1' });
  assert.ok(emUso(d, 'categorias', 'k1') && emUso(d, 'pessoas', 'p1') && emUso(d, 'contas', 'c1') && emUso(d, 'contratos', 't1'));
  assert.ok(!emUso(d, 'categorias', 'k5'));
});

test('cliente: código único, documento válido, serviços somam o valor do projeto', () => {
  const d = base();
  d.pessoas[0].codigo = 'CLI-0001';
  assert.throws(() => validarCadastro(d, 'pessoas', { nome: 'X', codigo: 'cli-0001' }), /código/);
  assert.throws(() => validarCadastro(d, 'pessoas', { nome: 'X', documento: '123' }), /CPF/);
  assert.equal(validarCadastro(d, 'pessoas', { nome: 'X', documento: '123.456.789-09', codigo: 'cli-0002' }).codigo, 'CLI-0002');
  assert.throws(() => validarCadastro(d, 'contratos', { codigo: 'ca260101', nome: 'Dup' }), /código/);
  d.servicos.push({ id: 's1', nome: 'Projeto arquitetônico', categoria_id: 'k1' }); d.mapa.servicos.set('s1', d.servicos[0]);
  const c = validarCadastro(d, 'contratos', { codigo: 'CA260102', nome: 'Casa B', pessoa_id: 'p1', area_m2: '120.5', valor_total_cents: 1,
    servicos: [{ servico_id: 's1', nome: 'Projeto arquitetônico', valor_cents: 800000 }, { nome: 'Consultoria', valor_cents: 200000 }] });
  assert.equal(c.valor_total_cents, 1000000);
  assert.equal(c.area_m2, 120.5);
  assert.throws(() => validarCadastro(d, 'contratos', { codigo: 'CA260103', nome: 'x', servicos: [{ servico_id: 'nao', nome: 'y', valor_cents: 1 }] }), /Serviço não encontrado/);
  d.contratos.push({ ...c, id: 't2' }); d.mapa.contratos.set('t2', d.contratos.at(-1));
  assert.ok(emUso(d, 'servicos', 's1'));
});

test('clientes: serviços contratados, situação financeira e códigos sugeridos', () => {
  const d = base();
  d.pessoas[0].codigo = 'CLI-0007';
  Object.assign(d.contratos[0], { valor_total_cents: 150000, servicos: [{ nome: 'Projeto arquitetônico', valor_cents: 150000 }], area_m2: 90 });
  rec(d, { nome: 'P1', contrato_id: 't1', pessoa_id: 'p1', valor_total_cents: 100000, primeiro_vencimento: '2026-05-01', primeira_paga: true });
  rec(d, { nome: 'P2', contrato_id: 't1', pessoa_id: 'p1', valor_total_cents: 50000, parcelas: 1, primeiro_vencimento: '2026-05-10' });
  const r = clientes(d, {}, HOJE);
  assert.equal(r.itens.length, 1); // fornecedor fica de fora
  const c = r.itens[0];
  assert.deepEqual([c.contratado_cents, c.pago_cents, c.vencido_cents, c.aberto_cents], [150000, 100000, 50000, 0]);
  assert.deepEqual(c.servicos, ['Projeto arquitetônico']);
  assert.equal(c.projetos[0].area_m2, 90);
  assert.equal(clientes(d, { busca: 'arquitet' }, HOJE).itens.length, 1);
  assert.equal(clientes(d, { busca: 'zzz' }, HOJE).itens.length, 0);
  assert.deepEqual(sugestoesCodigo(d, HOJE), { cliente: 'CLI-0008', projeto: 'CA260601' });
  d.contratos[0].codigo = 'CA260601';
  assert.equal(sugestoesCodigo(d, HOJE).projeto, 'CA260602');
});

test('cliente: pessoa física/jurídica, documento por natureza, endereço do cliente e da obra', () => {
  const d = base();
  assert.throws(() => validarCadastro(d, 'pessoas', { nome: 'X', natureza: 'fisica', documento: '12345678000199' }), /CPF/);
  assert.throws(() => validarCadastro(d, 'pessoas', { nome: 'X', natureza: 'juridica', documento: '12345678909' }), /CNPJ/);
  assert.throws(() => validarCadastro(d, 'pessoas', { nome: 'X', estado: 'São Paulo' }), /Estado/);
  assert.throws(() => validarCadastro(d, 'pessoas', { nome: 'X', data_nascimento: '31/02/2000' }), /nascimento/);
  const p = validarCadastro(d, 'pessoas', { nome: 'Djair', natureza: 'fisica', documento: '894.409.608-25', rg: '8375443 SSP/SP', cep: '18270-000',
    endereco: 'Estrada Municipal', numero: '22', bairro: 'Congonhal', cidade: 'Tatuí', estado: 'sp', consumidor_final_nfse: 'nao', data_nascimento: '1980-05-10' });
  assert.equal(p.estado, 'SP'); assert.equal(p.consumidor_final_nfse, false);
  assert.equal(validarCadastro(d, 'pessoas', { nome: 'Empresa', documento: '12.345.678/0001-99' }).natureza, 'juridica');
  const c = validarCadastro(d, 'contratos', { codigo: 'CA260109', nome: 'Obra', obra: { endereco: ' Rua A ', cidade: 'Tatuí', estado: 'sp', numero: '' } });
  assert.deepEqual([c.obra.endereco, c.obra.estado, c.obra.numero], ['Rua A', 'SP', null]);
  assert.equal(validarCadastro(d, 'contratos', { codigo: 'CA260110', nome: 'Sem obra', obra: { endereco: '' } }).obra, null);
});

test('a receber: atrasos por cliente, faixas de atraso, agenda e projeção de caixa', () => {
  const d = base(); // hoje = 2026-06-15; Inter começa com 1.000,00
  rec(d, { nome: 'Atrasada', pessoa_id: 'p1', contrato_id: 't1', valor_total_cents: 30000, primeiro_vencimento: '2026-05-16' }); // 30 dias
  rec(d, { nome: 'Muito atrasada', pessoa_id: 'p1', valor_total_cents: 20000, primeiro_vencimento: '2026-02-01' }); // 134 dias
  rec(d, { nome: 'Semana que vem', pessoa_id: 'p1', valor_total_cents: 10000, primeiro_vencimento: '2026-06-20' });
  rec(d, { nome: 'Paga', pessoa_id: 'p1', valor_total_cents: 99900, primeiro_vencimento: '2026-06-01', primeira_paga: true });
  rec(d, { nome: 'Sem cliente', valor_total_cents: 5000, primeiro_vencimento: '2026-06-15' });
  desp(d, { nome: 'Aluguel', valor_total_cents: 40000, primeiro_vencimento: '2026-06-25' });
  const r = aReceber(d, {}, HOJE);
  assert.deepEqual([r.totais.vencido.qtd, r.totais.vencido.valor_cents], [2, 50000]);
  assert.deepEqual([r.totais.vence_hoje.qtd, r.totais.proximos_7.valor_cents, r.totais.clientes_em_atraso], [1, 15000, 1]);
  assert.equal(r.aging[0].valor_cents, 30000); assert.equal(r.aging[3].valor_cents, 20000);
  assert.equal(r.clientes[0].nome, 'Cliente A'); assert.equal(r.clientes[0].max_atraso_dias, 134);
  assert.equal(r.clientes[0].parcelas.length, 3); assert.equal(r.clientes[1].nome, 'Sem cliente informado');
  assert.equal(r.agenda[0].nome, 'Sem cliente');
  const caixa = 100000 + 99900; // saldo inicial + recebido
  assert.equal(r.caixa.saldo_cents, caixa);
  assert.equal(r.caixa.projecao[0].saldo_projetado_cents, caixa + 15000); // 7 dias: entra 150,00
  assert.equal(r.caixa.projecao[2].saldo_projetado_cents, caixa + 15000 - 40000);
  assert.equal(aReceber(d, { busca: 'zzz' }, HOJE).clientes.length, 0);
});
