// Dados FICTÍCIOS só para demonstração local (npm run demo). Nunca vão para produção.
import * as F from '../api/_lib/finance.js';
import { gravar } from '../api/_lib/dados.js';
import { addMeses } from '../api/_lib/dates.js';
import { registrosDeServicos, registrosDoPlano } from '../api/_lib/plano.js';

export async function semearDemo(store, hoje = '2026-10-05') {
  const plano = registrosDoPlano();
  const servicos = registrosDeServicos();
  const d = F.criarDados({ ...plano, servicos });
  const add = (col, obj) => { const o = { id: F.novoId(), ...obj }; d[col].push(o); d.mapa[col].set(o.id, o); return o; };
  const trocar = (l) => { d.lancamentos.splice(d.lancamentos.findIndex((x) => x.id === l.id), 1, l); d.mapa.lancamentos.set(l.id, l); };
  const cat = (nome) => plano.categorias.find((c) => c.nome === nome).id;
  const cc = (nome) => plano.centros.find((c) => c.nome === nome).id;
  const inter = add('contas', { nome: 'Banco Inter', banco: 'Inter', saldo_inicial_cents: 2500000, ativa: 1 });
  const nubank = add('contas', { nome: 'Nubank PJ', banco: 'Nubank', saldo_inicial_cents: 300000, ativa: 1 });
  const reserva = add('contas', { nome: 'Reserva', banco: 'Inter', saldo_inicial_cents: 1000000, ativa: 1 });
  const nomes = ['Ana Beatriz Souza', 'Carlos Menezes', 'Daniela Ribeiro', 'Eduardo Prado', 'Fernanda Lima', 'Gustavo Alves',
    'Helena Costa', 'Igor Martins', 'Julia Ferraz', 'Marcos Tavares', 'Natália Pires', 'Otávio Neves'];
  const cidades = ['Goiânia', 'Brasília', 'Anápolis', 'Aparecida de Goiânia', 'Trindade'];
  const clientes = nomes.map((nome, i) => add('pessoas', { nome, tipo: 'cliente', codigo: `CLI-${String(i + 1).padStart(4, '0')}`, cidade: cidades[i % cidades.length],
    telefone: `(62) 99${String(100 + i * 7).padStart(3, '0')}-${String(1000 + i * 131).slice(0, 4)}`, email: `${nome.split(' ')[0].toLowerCase().normalize('NFD').replace(/\p{M}/gu, '')}@exemplo.com.br`, criado_em: '2026-01-01T00:00:00.000Z' }));
  const forn = (nome) => add('pessoas', { nome, tipo: 'fornecedor' });
  const [eng, hid, imob, contab, soft, socio, colab, gov] = ['Engenharia Estrutural Silva', 'Projetos Hidráulicos Rocha', 'Imobiliária Central', 'Contabilidade Exata',
    'Assinaturas de software', 'Sócio-administrador', 'Auxiliar administrativo', 'Receita Federal (DAS)'].map(forn);

  const corte = `${hoje.slice(0, 8)}01`; // o que venceu antes do mês atual já foi pago
  const lanc = (i) => { const l = F.criarLancamento(d, i, `2026-01-01T00:00:00.${String(d.lancamentos.length).padStart(3, '0')}Z`); d.lancamentos.push(l); d.mapa.lancamentos.set(l.id, l); return l; };
  const pagarPassadas = (l, conta_id) => {
    for (const p of l.parcelas) {
      if (p.vencimento < corte) { const novo = F.baixarParcela(d, `${l.id}:${p.numero}`, { data: p.vencimento, conta_id }); trocar(novo); l = novo; }
    }
  };
  const tipos = [['Projeto arquitetônico', 28000], ['Projeto de interiores', 18000], ['Gestão de obras', 36000], ['Regularização e documentação', 7500]];
  clientes.forEach((cl, i) => {
    const [tipo, base] = tipos[i % tipos.length];
    const mes = (i % 9) + 1;
    const inicio = `2026-${String(mes).padStart(2, '0')}-10`;
    const ct = add('contratos', { codigo: `CA26${String(mes).padStart(2, '0')}${String(i + 1).padStart(2, '0')}`, nome: nomes[i].split(' ')[0],
      pessoa_id: cl.id, valor_total_cents: (base + i * 1500) * 100, competencia: inicio, status: 'ativo', area_m2: 80 + i * 17,
      servicos: [{ servico_id: servicos.find((x) => x.nome === tipo).id, nome: tipo, valor_cents: (base + i * 1500) * 100 }] });
    const conta = i % 3 === 0 ? nubank.id : inter.id;
    pagarPassadas(lanc({ tipo: 'receita', nome: `${tipo} ${nomes[i].split(' ')[0]}`, pessoa_id: cl.id, categoria_id: cat(tipo), centro_custo_id: cc('Projetos'),
      contrato_id: ct.id, conta_id: conta, competencia: inicio, valor_total_cents: (base + i * 1500) * 100, parcelas: 3 + (i % 6), primeiro_vencimento: inicio }), conta);
    if (i % 2 === 0) {
      pagarPassadas(lanc({ tipo: 'despesa', nome: `Projeto complementar ${nomes[i].split(' ')[0]}`, pessoa_id: i % 4 === 0 ? eng.id : hid.id,
        categoria_id: cat('Projetos complementares (terceirizados)'), centro_custo_id: cc('Projetos'), contrato_id: ct.id, conta_id: inter.id, competencia: inicio,
        valor_total_cents: Math.round(base * 0.18) * 100, parcelas: 2, primeiro_vencimento: addMeses(inicio, 1) }), inter.id);
    }
  });
  const fixa = (nome, p, categoria, valor, centro, primeiro = '2026-01-05') => pagarPassadas(lanc({ tipo: 'despesa', nome, pessoa_id: p.id,
    categoria_id: cat(categoria), centro_custo_id: cc(centro), conta_id: inter.id, valor_total_cents: valor * 100, recorrente: true, repeticoes: 12, primeiro_vencimento: primeiro }), inter.id);
  fixa('Aluguel do escritório', imob, 'Aluguel e condomínio', 2800, 'Administrativo');
  fixa('Pró-labore', socio, 'Pró-labore e salários', 8000, 'Administrativo', '2026-01-07');
  fixa('Salário auxiliar administrativo', colab, 'Pró-labore e salários', 2200, 'Administrativo', '2026-01-07');
  fixa('Contabilidade', contab, 'Contabilidade', 900, 'Administrativo', '2026-01-10');
  fixa('Assinaturas (CAD, render, nuvem)', soft, 'Softwares e assinaturas', 650, 'Administrativo', '2026-01-12');
  fixa('DAS (Simples Nacional)', gov, 'Impostos sobre faturamento', 1800, 'Administrativo', '2026-01-20');
  lanc({ tipo: 'receita', nome: 'Visita técnica avulsa', pessoa_id: clientes[1].id, categoria_id: cat('Visitas técnicas e consultoria'), conta_id: inter.id,
    valor_total_cents: 120000, primeiro_vencimento: '2026-09-18' });
  add('transferencias', F.criarTransferencia(d, { descricao: 'Reserva de caixa', valor_cents: 500000, conta_origem_id: inter.id, conta_destino_id: reserva.id, data: '2026-06-02' }));

  for (const col of ['contas', 'servicos', 'pessoas', 'contratos', 'lancamentos', 'transferencias']) for (const o of d[col]) await gravar(store, col, o);
  // plano de contas e marcador são criados por carregarDados quando o marcador não existe
}
