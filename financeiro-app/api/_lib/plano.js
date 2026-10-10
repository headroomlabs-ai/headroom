// Plano de contas inicial para escritório de arquitetura e gestão de obras (ajuste na tela Cadastros).
// Os ids são fixos para que a criação inicial seja idempotente.
const CATEGORIAS = [
  ['Projeto arquitetônico', 'receita', 'receita_bruta'], ['Projeto de interiores', 'receita', 'receita_bruta'],
  ['Gestão de obras', 'receita', 'receita_bruta'], ['Regularização e documentação', 'receita', 'receita_bruta'],
  ['Visitas técnicas e consultoria', 'receita', 'receita_bruta'], ['Outras receitas operacionais', 'receita', 'receita_bruta'],
  ['Rendimentos financeiros', 'receita', 'financeiras'], ['Outras receitas não operacionais', 'receita', 'nao_operacionais'],
  ['Impostos sobre faturamento', 'despesa', 'deducoes'], ['Taxas de cobrança e boletos', 'despesa', 'deducoes'],
  ['Projetos complementares (terceirizados)', 'despesa', 'custos_operacionais'], ['Engenheiros e consultores', 'despesa', 'custos_operacionais'],
  ['Maquetes, plotagens e impressões', 'despesa', 'custos_operacionais'],
  ['Pró-labore e salários', 'despesa', 'despesas_operacionais'], ['Encargos e benefícios', 'despesa', 'despesas_operacionais'],
  ['Aluguel e condomínio', 'despesa', 'despesas_operacionais'], ['Energia, água e internet', 'despesa', 'despesas_operacionais'],
  ['Softwares e assinaturas', 'despesa', 'despesas_operacionais'], ['Marketing', 'despesa', 'despesas_operacionais'],
  ['Contabilidade', 'despesa', 'despesas_operacionais'], ['Material de escritório', 'despesa', 'despesas_operacionais'],
  ['Alimentação e deslocamento', 'despesa', 'despesas_operacionais'],
  ['Juros e tarifas bancárias', 'despesa', 'financeiras'], ['Outras despesas não operacionais', 'despesa', 'nao_operacionais'],
];
const CENTROS = ['Administrativo', 'Comercial e marketing', 'Projetos', 'Obras'];

export const MARCADOR = 'dados/meta/plano.json';
export const MARCADOR_SERVICOS = 'dados/meta/servicos.json';

// Catálogo inicial de serviços (cada um aponta para a categoria de receita correspondente).
const SERVICOS = [
  ['Projeto arquitetônico', 'cat01'], ['Projeto de interiores', 'cat02'], ['Gestão de obras', 'cat03'],
  ['Regularização e documentação', 'cat04'], ['Visita técnica e consultoria', 'cat05'], ['Compatibilização de projetos', 'cat06'],
];
export const registrosDeServicos = () => SERVICOS.map(([nome, categoria_id], i) => ({
  id: `srv${String(i + 1).padStart(2, '0')}`, nome, descricao: null, valor_padrao_cents: 0, categoria_id, ativo: 1 }));

export function registrosDoPlano() {
  return {
    categorias: CATEGORIAS.map(([nome, tipo, grupo_dre], i) => ({ id: `cat${String(i + 1).padStart(2, '0')}`, nome, tipo, grupo_dre })),
    centros: CENTROS.map((nome, i) => ({ id: `cc${i + 1}`, nome })),
  };
}
