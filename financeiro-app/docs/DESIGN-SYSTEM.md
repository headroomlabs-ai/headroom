# Design system "Vidro & Luz"

Direção: dados flutuando em painéis de vidro sobre um ambiente escuro e luminoso (entardecer visto através de janelas). Quanto mais importante o elemento, mais ele brilha. Neon é acento, nunca cor de texto longo.

Arquivos: `public/styles.css` (tokens e componentes), `public/ui.js` (ícones, medidor, gráficos, esqueletos, estados), `public/util.js` (modal/bottom sheet, toasts, desfazer), `public/forms.js` (formulários), `public/views.js` (telas), `public/app.js` (navegação, busca global, tema).

## Tokens (CSS, `:root`)
| Grupo | Tokens |
|---|---|
| Superfícies | `--bg #0B0D14`, `--surface-1 #12151F`, `--surface-2 #1A1E2C`, `--glass`, `--glass-2`, `--linha` |
| Texto | `--text #F5F7FA`, `--text-2 #9AA3B5`, `--text-3` |
| Marca | `--amber #FF9F1C`, `--orange #FF6B35`, `--violet #8B5CF6`, `--lilac #C084FC`, `--magenta #D946EF`, `--cyan #22D3EE` |
| Estados | `--ok #34D399`, `--warn #FBBF24`, `--err #F87171` (sempre com ícone e texto, nunca só cor) |
| Gradientes | `--grad-primary` (âmbar→laranja), `--grad-violet` (violeta→magenta), `--grad-borda` (reflexo na borda do vidro) |
| Espaço | escala de 4px: `--s1` … `--s12` |
| Raios | `--r-sm 10`, `--r-md 16`, `--r-lg 22`, `--r-xl 28`, `--r-pill` |
| Efeitos | `--blur 18px`, `--sombra-1/2`, `--glow-amber/violet/cyan` |
| Movimento | `--t-fast 150ms`, `--t-med 250ms`, `--t-slow 700ms`, `--ease` |
| Tipografia | corpo Plus Jakarta Sans, títulos e números Space Grotesk; números tabulares |

Modo escuro é o padrão; modo claro (`[data-theme="light"]`) usa vidro claro e os mesmos acentos. Em aparelhos fracos (`data-fx="low"`), blur, brilho e animações pesadas são desligados e o vidro vira cor sólida.

## Componentes
- **Vidro** `.glass` (blur, borda em gradiente, reflexo diagonal), `.painel`, `.mira` (cantos de mira).
- **Medidor radial** `gauge()`: KPI principal de cada tela, com anéis finos, marcas de escala, brilho e pulsação sutil.
- **Orbes** `.orb` (violeta, âmbar, ciano) para indicadores secundários e contas.
- **KPI** `kpi()`: rótulo, ícone, valor animado, quantidade, micrográfico opcional.
- **Gráficos** `grafico()` (barras), `linha()`, `rosca()`, `spark()`, todos "se desenhando" ao entrar.
- **Botões** `.btn-primary`, `.btn-violet`, `.btn` (secundário), `.btn-ghost`, `.btn-danger`, `.btn-ok`; **um CTA por tela** (`#cta`), fixo na parte de baixo no celular.
- **Campos** `.campo` (vidro, foco âmbar/ciano), `.switch`, datas, selects, busca em pílula.
- **Chips** de status com ícone + texto, tags, avatares, barras de progresso.
- **Tabelas** `.tbl` (viram cards no celular, com ordenação), matrizes `.hier` (fluxo e DRE) com primeira coluna fixa.
- **Modais** viram **bottom sheets** no celular; formulário longo em etapas (`.passos`).
- **Toasts** com ação "Desfazer"; **busca global** (Ctrl/Cmd+K); **esqueletos**, **estados vazio e erro**.
- **Navegação:** menu lateral recolhível com favoritos e botão "+", gaveta no celular, barra inferior com 5 atalhos, migalhas de pão.

## Acessibilidade e desempenho
Contraste AA nos textos, foco visível, rótulos em todos os campos, `aria-*` em abas, medidores e tabelas, `prefers-reduced-motion` respeitado, áreas de toque de 44px, safe areas do celular. PWA: `manifest.webmanifest`, ícones e `sw.js` (só arquivos estáticos; a API nunca é guardada).
