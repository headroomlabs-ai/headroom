import Link from 'next/link';
import { Button } from './button';
import { CodeBlock } from './code-block';

// --- Live Stats Grid ---

const liveStats = [
  { value: '$176.6K', label: 'Cost Saved' },
  { value: '1.19M', label: 'Requests Optimized' },
  { value: '889', label: 'Active Instances' },
  { value: '14', label: 'Active Days' },
];

export function LiveStats() {
  return (
    <div className="not-prose">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 my-8">
        {liveStats.map((s) => (
          <div
            key={s.label}
            className="flex flex-col items-center p-5 rounded-xl border border-fd-border bg-fd-card"
          >
            <span className="text-2xl font-bold text-fd-foreground">
              {s.value}
            </span>
            <span className="mt-1 text-sm text-fd-muted-foreground">
              {s.label}
            </span>
          </div>
        ))}
      </div>
      <Link
        href="/docs/community-savings"
        className="text-sm font-medium hover:underline"
      >
        View detailed charts and breakdowns &rarr;
      </Link>
    </div>
  );
}

// --- Key Features Grid ---

const features: {
  title: string;
  description: string;
  href: string;
  code?: string;
  lang?: string;
}[] = [
  {
    title: 'Lossless Compression (CCR)',
    description:
      'Compresses aggressively, stores originals, gives the LLM a tool to retrieve full details. Nothing is thrown away.',
    href: '/docs/ccr',
  },
  {
    title: 'Smart Content Detection',
    description:
      'Auto-detects JSON, code, logs, text, diffs, HTML. Routes each to the best compressor. Zero configuration needed.',
    href: '/docs/how-compression-works',
  },
  {
    title: 'Cache Optimization',
    description:
      "Stabilizes prefixes so provider KV caches hit. Tracks frozen messages to preserve the 90% read discount.",
    href: '/docs/cache-optimization',
  },
  {
    title: 'Image Compression',
    description:
      '40-90% token reduction via trained ML router. Automatically selects resize/quality tradeoff per image.',
    href: '/docs/image-compression',
  },
  {
    title: 'Persistent Memory',
    description:
      'Hierarchical memory (user/session/agent/turn) with SQLite + HNSW backends. Survives across conversations.',
    href: '/docs/memory',
  },
  {
    title: 'Failure Learning',
    description:
      'Reads past sessions, finds failed tool calls, correlates with what succeeded, writes learnings to CLAUDE.md.',
    href: '/docs/failure-learning',
  },
  {
    title: 'Multi-Agent Context',
    description: 'Compress what moves between agents. Any framework.',
    href: '/docs/shared-context',
    code: 'ctx = SharedContext()\nctx.put("research", big_output)\nsummary = ctx.get("research")',
    lang: 'python',
  },
  {
    title: 'Metrics & Observability',
    description:
      'Prometheus endpoint, per-request logging, cost tracking, budget limits, pipeline timing breakdowns.',
    href: '/docs/metrics',
  },
];

export async function KeyFeatures() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 my-8 not-prose">
      {await Promise.all(
        features.map(async (f) => (
          <div
            key={f.title}
            className="flex flex-col p-5 rounded-xl border border-fd-border bg-fd-card"
          >
            <h3 className="text-base font-semibold text-fd-foreground">
              {f.title}
            </h3>
            <p className="mt-2 text-sm text-fd-muted-foreground flex-1">
              {f.description}
            </p>
            {f.code && <CodeBlock code={f.code} lang={f.lang} />}
            <Link
              href={f.href}
              className="mt-3 text-sm font-medium hover:underline"
            >
              Learn more &rarr;
            </Link>
          </div>
        )),
      )}
    </div>
  );
}

// --- Framework Integrations Bento ---

const integrations: {
  title: string;
  description: string;
  code: string;
  lang: string;
  href: string;
}[] = [
  {
    title: 'LangChain',
    description:
      'Wrap any chat model. Supports memory, retrievers, tools, streaming, async.',
    code: 'from headroom.integrations.langchain import HeadroomChatModel\nllm = HeadroomChatModel(ChatOpenAI())',
    lang: 'python',
    href: '/docs/langchain',
  },
  {
    title: 'Agno',
    description:
      'Full agent framework integration with observability hooks.',
    code: 'from headroom.integrations.agno import HeadroomAgnoModel\nmodel = HeadroomAgnoModel(Claude())\nagent = Agent(model=model)',
    lang: 'python',
    href: '/docs/agno',
  },
  {
    title: 'Strands',
    description:
      'Model wrapping + tool output hook provider for Strands Agents.',
    code: 'from headroom.integrations.strands import HeadroomStrandsModel\nmodel = HeadroomStrandsModel(...)\nagent = Agent(model=model)',
    lang: 'python',
    href: '/docs/strands',
  },
  {
    title: 'MCP Tools',
    description:
      'Three tools for Claude Code, Cursor, or any MCP client: headroom_compress, headroom_retrieve, headroom_stats.',
    code: 'headroom mcp install && claude',
    lang: 'bash',
    href: '/docs/mcp',
  },
  {
    title: 'TypeScript SDK',
    description:
      'compress(), Vercel AI SDK middleware, OpenAI and Anthropic client wrappers.',
    code: 'npm install headroom-ai',
    lang: 'bash',
    href: '/docs/vercel-ai-sdk',
  },
  {
    title: 'Vercel AI SDK',
    description:
      'One-liner withHeadroom() or headroomMiddleware() for any Vercel AI SDK model.',
    code: "import { withHeadroom } from 'headroom-ai/vercel-ai'\nconst model = withHeadroom(openai('gpt-4o'))",
    lang: 'typescript',
    href: '/docs/vercel-ai-sdk',
  },
];

export async function FrameworkIntegrations() {
  return (
    <div className="not-prose">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 my-8">
        {await Promise.all(
          integrations.map(async (i) => (
            <div
              key={i.title}
              className="flex flex-col p-5 rounded-xl border border-fd-border bg-fd-card"
            >
              <h3 className="text-base font-semibold text-fd-foreground">
                {i.title}
              </h3>
              <p className="mt-2 text-sm text-fd-muted-foreground flex-1">
                {i.description}
              </p>
              <CodeBlock code={i.code} lang={i.lang} />
              <Link
                href={i.href}
                className="mt-3 text-sm font-medium hover:underline"
              >
                {i.title} Guide &rarr;
              </Link>
            </div>
          )),
        )}
      </div>
      <Button variant="link" size="sm" asChild>
        <Link href="/docs/quickstart">
          All integration patterns &rarr;
        </Link>
      </Button>
    </div>
  );
}
