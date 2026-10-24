# Análise das telas do Vobi e plano de montagem do app financeiro interno

## A. Análise tela a tela

### Home e Projetos (fotos 1 a 5)
- **Home:** quatro cartões (Oportunidades 102, Projetos 70, Tarefas 14 pendentes e 950 atrasadas, Pagamentos 4 em aberto). Para nós, só o cartão **Pagamentos** importa. Ele vira o alerta do dashboard.
- **Projetos (lista e quadro):** os códigos seguem o padrão `CA` + AAMM + sequência + nome (ex.: CA261001). Etapas: Levantamento, Aguardando liberação, Briefing, Estudo Preliminar, Aprovações. Projeto e cliente são a "chave" que liga o financeiro. No nosso app, **contrato** guarda o código, o cliente e o status; o resto da gestão de projetos fica fora.
- **Administrativo:** é um projeto especial que recebe tudo o que é do escritório. Nós trocamos isso por um campo simples: o lançamento pode ou não ter contrato. Sem contrato, é do escritório.

### Tarefas e Gestão de Horas (fotos 6 e 7)
Fora do escopo. Só reaproveitamos a ideia de horas por projeto numa fase futura, para medir a rentabilidade.

### Resumo do Financeiro (foto 8)
- **Período:** 01-01-26 a 31-12-26.
- **Contas:** saldo atual total (R$ 844.310,89) e saldo por conta (Inter, Rico, Nubank). Contas com saldo negativo aparecem em vermelho.
- **Receitas:** Em aberto (59), Vencido (6), Recebido (176). **Despesas:** Em aberto (112), Vencido (0), Pagos (644).
- **Resultado:** Balanço R$ 333.517,60 e barra verde × vermelha.
- **Próximas receitas e despesas:** nome com parcela (ex.: "Comissão ... 14/18"), favorecido, valor e vencimento.
- **Regras confirmadas pelas contas:**
  - Em aberto = a vencer + vence hoje, sem os vencidos. Receitas: 1.125,00 + 152.092,93 = 153.217,93, com 59 = 58 + 1 parcelas.
  - Balanço = recebido − pagos = 679.690,42 − 346.172,82 = 333.517,60. É resultado de caixa (realizado).
- Os links "Pagamentos do cliente", "Receitas" e "Despesas" no menu sugerem que os cartões também funcionam como atalho para a lista filtrada.

### Pagamentos do cliente (foto 9)
- **Visualizar por: Pagamentos** (um registro por contrato) e filtro por **data de competência**.
- Colunas: nome do pagamento, fornecedor (o próprio escritório ou terceiro), status, data, projeto, valor total, valor pago e em aberto. Linha expansível com as parcelas.
- Exemplo: total 29.700 · pago 26.730 · em aberto 2.970. O status "Em aberto" aparece quando falta parcela.
- Há linhas "Pagamento OC00153" com fornecedor terceiro (engenheira, fornecedor) e valor de R$ 1.000: são repasses ligados à compra, cobrados do cliente. Pendência: confirmar com vocês se existem no processo de vocês.

### Receitas (fotos 10 e 11)
- **Visualizar por: Parcelas.** Filtro de data por vencimento da parcela.
- Faixas clicáveis: Vencidos (6, R$ 5.325,00) · Vence hoje (1) · A vencer (58) · Recebidos (176) · Total do período (241, R$ 838.233,35). A soma bate: 5.325 + 1.125 + 152.092,93 + 679.690,42 = 838.233,35.
- Gráfico mensal com cores por situação (verde recebido, vermelho atrasado, laranja previsto).
- Tabela: nome (ex.: "Projeto ... 10/17"), valor, vencimento, data de pagamento, cliente, projeto, status, ações. Linhas com **chips numéricos** (59, 61, 93...). Pendência: ainda não sei o que são (parece a numeração das parcelas ou de lançamentos agrupados). Vamos tratar como etiquetas até vocês confirmarem.
- Botão Nova receita, busca por "nome ou nota fiscal", paginação por 5 páginas ou mais.

### Despesas (fotos 12 e 13)
- Igual a Receitas, mais **Importar XML** e **Nova despesa**.
- Valores em vermelho. Ícone de recorrência (↻) nas fixas (empréstimo de sócio, salário). Fornecedor pessoa física ou jurídica. Projeto "-" para despesa do escritório.
- Total do período: 442.065,72 = 95.892,90 a vencer + 346.172,82 pagos.

### Transferências (foto 14)
Colunas Descrição, Valor, De, Para, Data. Estado vazio com chamada para criar. Não pode entrar em receitas nem despesas.

### Agente de IA Financeiro (foto 15) e Central de lançamentos
Recebimento de lançamentos por e-mail, WhatsApp e app, preenchimento automático, anexos, conciliação por Open Finance e busca de notas de compra. Premium no Vobi. Fica para a fase 3 (a base já prevê anexos e status "conciliado").

### Menus Contas e Extratos, Relatórios, Notas Fiscais e Vobi Pay (fotos 16, 17, 20 e 21)
- Contas e Extratos: Contas bancárias · Extratos.
- Relatórios: Fluxo de caixa (subitens) · DRE Gerencial · Resultados gerais · Resultados consolidados por projeto · Resultados previstos por projeto · Outros relatórios.
- Notas Fiscais: NFS-e emitidas · NF-e recebidas · Configurações (bloqueadas).
- Vobi Pay: Resumo · Cobranças · Extrato · Simulador de cobranças.

### Fluxo de caixa (foto 17, primeira tela de relatórios)
- Colunas: mês, com **Previsto** e **Realizado** lado a lado. Linhas: Saldo inicial, Total de Receitas (por categoria), Total de Despesas (por categoria).
- **Regra verificada:** Saldo inicial realizado de fev = saldo jan + receitas − despesas realizados = 509.133,91 + 129.447,25 − 34.358,84 = 604.222,32. O previsto segue a mesma conta (517.023,29 + 126.671,87 − 34.358,84 = 609.336,32).
- Observação: o saldo inicial de janeiro difere entre previsto (517.023,29) e realizado (509.133,91). Pendência: entender por que (provavelmente o previsto inclui parcelas vencidas). No nosso app, usar a mesma base nas duas colunas.

### DRE Gerencial (foto 18)
- Estrutura (por mês): Receita operacional bruta · (−) Deduções · (=) Receita líquida · (−) Custos operacionais · (=) Resultado bruto · Margem bruta · (−) Despesas operacionais · (=) Resultado operacional · Margem operacional · Receitas e despesas financeiras · Outras não operacionais.
- **Margens verificadas:** são calculadas sobre a **receita operacional bruta**, não sobre a líquida. Jan: 11.663,08 ÷ 16.753,80 = 69,61%. Margem operacional: −20.998,76 ÷ 16.753,80 = −125,34%.
- A DRE usa a competência (receita bruta de jan = 16.753,80), enquanto o fluxo de caixa usa o caixa (receitas de jan = 129.447,25). Os dois relatórios têm bases diferentes de propósito.
- Cada grupo expande por categoria.

### Resultados gerais (foto 19) e por projeto (fotos 22 a 24)
- Resultados gerais: Receitas 679.690,42 (176) · Despesas 346.172,82 (644) · Resultado 333.517,60. Igual ao balanço do Resumo. Gráfico mensal.
- Consolidado por projeto (paga): Receitas 683.429,80 · Despesas 65.564,04 · Resultado 617.865,76. Tabela Projeto, Cliente, Receitas, Despesas e Resultado.
- Previsto por projeto: mesma tabela com previsão de receitas, de despesas e resultado previsto.

### Outros relatórios (foto 25)
Abas Clientes, Fornecedores, Categorias e Centros de custo. Filtros: tipo de data (competência, vencimento, pagamento), período, tipo (receita ou despesa), status das parcelas e detalhamento. Botão Buscar (o relatório só é gerado ao clicar) e Exportar.

## B. Como montar no nosso app

### Princípio central
Tudo nasce de **lançamento → parcelas**. Todo número das telas (saldos, cartões, DRE, fluxo) é uma soma de parcelas. Se o modelo estiver certo, os relatórios saem de consultas, sem lançar nada duas vezes.

### Regras de cálculo (validadas com os números dos prints)
| Conceito | Regra |
|---|---|
| Status da parcela | Pago: valor pago ≥ valor. Vencido: aberta e vencimento < hoje. Vence hoje. A vencer. |
| Em aberto (cartão) | a vencer + vence hoje (sem vencidos) |
| Balanço / Resultado realizado | recebido − pago, filtrado por data de pagamento |
| Saldo da conta | saldo inicial + recebimentos − pagamentos ± transferências |
| Fluxo de caixa | previsto = vencimento; realizado = data de pagamento; saldo inicial do mês = saldo final do anterior |
| DRE | competência; margens ÷ receita operacional bruta |
| Pagamento do cliente | total, pago e em aberto = soma das parcelas do contrato |

### Telas e componentes (reaproveitáveis)
1. **Lista de parcelas** (Receitas e Despesas usam o mesmo componente): faixas clicáveis, gráfico mensal, tabela com busca, filtros, ordenação, paginação e ações em lote. Receitas e Despesas mudam só o tipo e a cor.
2. **Cartões de resumo** (Resumo, Resultados gerais, por projeto).
3. **Tabela hierárquica por mês** (Fluxo de caixa e DRE): linhas expansíveis por categoria, colunas por mês.
4. **Formulário de lançamento** com parcelamento e recorrência (gera as parcelas, ex.: "Projeto X 3/10").
5. **Seletor de tipo de data** (competência, vencimento, pagamento) e de período, igual em todas as telas.

### Ordem de construção
1. **Base:** login, cadastros (contas, categorias com grupo da DRE, centros de custo, pessoas, etiquetas).
2. **Lançamentos:** formulário de receita e de despesa com parcelas e recorrência. Baixa (marcar como pago) com data, valor e conta.
3. **Listas:** Receitas, Despesas e Pagamentos do cliente (contratos com parcelas).
4. **Contas:** transferências, extrato e saldo por conta.
5. **Resumo (dashboard)** com os cartões e as listas de próximos vencimentos.
6. **Relatórios:** Resultados gerais, Fluxo de caixa, DRE Gerencial e Outros relatórios (com exportação para planilha).
7. **Fase 2:** resultados por projeto, importar XML, notas fiscais, boleto/Pix.
8. **Fase 3:** leitura de comprovantes por IA, WhatsApp e Open Finance.

### Tecnologia sugerida
Next.js + TypeScript, Postgres (Supabase) com login e permissões, gráficos com Recharts, importação e exportação de planilha (CSV/XLSX). Valores em centavos (inteiros) para não ter erro de arredondamento.

## C. Pendências (precisam de resposta ou de print)
1. O que são os chips numéricos nas Receitas (59, 61, 93...)?
2. Os "Pagamento OC..." (repasses ligados à compra) existem no processo de vocês?
3. Campos de **Nova receita** e **Nova despesa**.
4. Lista de **categorias** (com o grupo da DRE), **centros de custo** e **contas bancárias**.
5. Por que o saldo inicial de janeiro difere entre previsto e realizado?
6. Repositório no GitHub (vazio) e quem vai usar o app.
