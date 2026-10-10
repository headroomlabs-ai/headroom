# Financeiro Interno – Cariati Arquitetura e Gestão
Especificação baseada nas telas do módulo Financeiro do Vobi. Escopo: só a parte interna do escritório (sem obra e sem projeto do cliente).

## 1. Menu do app
Resumo · Pagamentos do cliente · Receitas · Despesas · Transferências · Central de lançamentos · Contas e Extratos · Relatórios (DRE, fluxo de caixa) · Notas fiscais · Cobrança · Cadastros

## 2. Telas

### 2.1 Resumo (dashboard)
- Filtro de período (padrão: ano corrente).
- Cartão **Contas**: saldo atual total e saldo por conta (Banco Inter, Banco Rico, Nubank…).
- Cartões **Receitas**: em aberto, vencido, recebido (valor + quantidade de parcelas).
- Cartões **Despesas**: em aberto, vencido, pagas (valor + quantidade).
- Cartão **Resultado**: balanço (receitas − despesas) e barra comparativa.
- Listas **Próximas receitas** e **Próximas despesas**: nome, favorecido, valor e vencimento (vencidas em vermelho).

### 2.2 Receitas
- Visualizar por: parcelas (ou lançamentos). Filtro por data de vencimento da parcela.
- Faixas: Vencidos · Vence hoje · A vencer · Recebidos · Total do período (valor + quantidade). Clicar na faixa filtra a lista.
- Gráfico mensal de receitas (recebido, atrasado e previsto em cores diferentes).
- Tabela: nome (ex.: "Projeto X 2/3"), valor, vencimento, data de pagamento, cliente, contrato/projeto, status (Pago, Em aberto, Vencido), etiquetas, ações. Paginação.
- Ação **Nova receita**, busca por nome ou nota fiscal, filtros, ações em lote.

### 2.3 Despesas
- Igual a Receitas, mais **Importar XML** (NF-e de compra) e **Nova despesa**.
- Despesa recorrente (aluguel, softwares, folha) e parcelada.

### 2.4 Pagamentos do cliente
- Visão por contrato: nome, fornecedor (o escritório), status, data de competência, projeto, valor total, valor pago, em aberto.
- Linha expansível com as parcelas. Filtro por data de competência.
- Ação **Novo pagamento** (gera as parcelas).

### 2.5 Transferências
Entre contas próprias (Inter → Nubank etc.), sem impacto no DRE.

### 2.6 Contas e Extratos
Contas bancárias, saldo inicial, extrato por conta, conciliação (marcar lançamento como conciliado).

### 2.7 Relatórios
- **DRE** por mês, com competência ou caixa, por categoria.
- **Fluxo de caixa** previsto × realizado, por conta.
- Inadimplência e aging (a receber vencido por faixa de dias).
- Receita por cliente e por tipo de serviço.

### 2.8 Notas fiscais
Registro de NFS-e emitidas e NF-e recebidas, vinculadas ao lançamento (campo de busca "nome ou nota fiscal").

### 2.9 Complementos vistos nas últimas telas
- **Despesas:** o campo Projeto fica vazio nas despesas do escritório (empréstimo de sócio, almoço, marketing, material, folha). Só vira projeto quando é custo de um contrato (ex.: engenheira terceirizada). Ícone de **recorrência** nas despesas fixas (empréstimo, salário).
- **Transferências:** colunas Descrição, Valor, De, Para, Data. Botão "Nova transferência".
- **Contas e Extratos:** submenu **Contas bancárias** e **Extratos**.
- **Relatórios (submenu do Vobi):** Fluxo de caixa (com subitens), DRE Gerencial, Resultados gerais, Resultados consolidados por projeto, Resultados previstos por projeto, Outros relatórios. As versões "por projeto" mostram a rentabilidade de cada contrato (fase 2).
- **Central de lançamentos** (premium no Vobi, "Agente de IA Financeiro"): recebe lançamentos por e-mail, WhatsApp ou app, preenche os campos sozinho, anexa comprovantes, concilia com o banco via Open Finance e busca notas de compra. Fica como fase 2 (leitura de comprovante por IA e Open Finance).

### 2.10 Relatórios em detalhe (telas enviadas)
**Fluxo de caixa mensal** (Previsto e realizado, ano)
- Colunas: cada mês com subcolunas Previsto e Realizado.
- Linhas: Saldo inicial; Total de Receitas (expansível por categoria: Gestão de Obras, Projetos, Receita de Vendas, Serviços documentações); Total de Despesas (expansível por categoria); saldo final.
- Saldo inicial do mês seguinte = saldo final do mês anterior.
- Verde para entradas e vermelho para saídas.

**DRE Gerencial** (ano, colunas por mês). Estrutura, na ordem:
1. Receita operacional bruta
2. (−) Deduções da receita bruta (impostos)
3. (=) Receita líquida de vendas
4. (−) Custos operacionais
5. (=) Resultado bruto, e **Margem bruta %** (resultado bruto ÷ receita operacional bruta; verificado com os números do print)
6. (−) Despesas operacionais
7. (=) Resultado operacional, e **Margem operacional %**
8. Receitas e despesas financeiras
9. Outras receitas e despesas não operacionais
(e abaixo delas, o resultado líquido)
- Cada grupo expande para as categorias. **Cada categoria tem um "grupo da DRE"**, e é isso que alimenta o relatório.

**Resultados gerais**
- Filtro por data de vencimento ou de pagamento das parcelas.
- Faixas: Receitas, Despesas, Resultado do período (valor + quantidade). Gráfico mensal do resultado. Botão "Novo registro" e busca por nome ou nota fiscal.

**Resultados consolidados por projeto**
- Mesmas faixas, mais a tabela Projeto, Cliente, Receitas, Despesas e Resultado (por contrato). Fase 2.
- "Previstos por projeto" é a mesma tabela usando as parcelas em aberto.

### 2.11 Últimas telas (previstos, outros relatórios, notas, Vobi Pay)
- **Resultados previstos por projeto:** Projeto, Cliente, Previsão de receitas, Previsão de despesas, Resultado previsto (usa as parcelas em aberto).
- **Outros relatórios:** abas **Clientes, Fornecedores, Categorias, Centros de custo**. Filtros: Data (competência, vencimento ou pagamento), data início e fim, Tipo (receitas ou despesas), Status das parcelas, Detalhamento (projeto etc.). Botão Buscar e **Exportar** (planilha). Isso é um relatório "quanto por cliente, fornecedor, categoria ou centro de custo".
- **Notas fiscais** (bloqueado no Vobi, plano premium): NFS-e emitidas, NF-e recebidas e Configurações. Fica para a fase 2.
- **Vobi Pay:** Resumo, Cobranças, Extrato, Simulador de cobranças. É boleto e Pix integrados. Fica para a fase 2 (precisa de um banco ou gateway, como Asaas, Inter ou Efí).

## 3. Modelo de dados (resumo)
- `contas` (nome, banco, saldo_inicial)
- `categorias` (nome, tipo receita|despesa, grupo_dre: receita_bruta | deducoes | custos_operacionais | despesas_operacionais | financeiras | nao_operacionais)
- `centros_custo`
- `pessoas` (cliente|fornecedor, documento, contato)
- `contratos` (cliente, nome, valor_total, data_competencia, origem: escritorio|cliente)
- `lancamentos` (tipo receita|despesa, nome, pessoa, categoria, centro_custo, contrato?, conta, competencia, etiquetas, nota_fiscal)
- `parcelas` (lancamento, numero/total, valor, vencimento, data_pagamento, valor_pago, status, conta)
- `transferencias` (conta_origem, conta_destino, valor, data)
- `cobrancas` (parcela, canal, enviada_em, status)
- `usuarios` e `perfis`
Regras:
- Status da parcela: Em aberto, Pago (parcial vira "em aberto" com valor pago), Vencido (aberta e vencimento < hoje).
- Saldo da conta = saldo inicial + recebimentos − pagamentos ± transferências.
- DRE usa a data de competência; fluxo de caixa usa a data de pagamento.

## 4. Fora do escopo (por enquanto)
Projetos e tarefas, gestão de horas, obras, estoque, compras e contratações, Vobi Pay (cobrança integrada). A cobrança nesta fase é manual: lembretes e status.

## 5. Dúvidas em aberto
1. Boleto/Pix integrado (Vobi Pay) é necessário ou basta registrar o recebimento?
2. Existem etiquetas e categorias padrão de vocês (plano de contas)? Precisa de um DRE no formato do contador.
3. As despesas têm "projeto" (rateio por obra) ou só o escritório?
4. Quantas pessoas vão usar e com quais permissões?
5. Dados atuais: importar do Vobi por planilha (Ações → exportar)?

## 6. Sugestão técnica
Next.js + TypeScript + Postgres (Supabase), login por e-mail, versão responsiva para celular.
