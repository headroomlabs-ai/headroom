import type { Snip, Totals } from '../types'

export const FRAMES = 24
export const FRAME_MS = 45
export const TOKENS_PER_PAGE = 600
export const MILESTONES = [10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000, 5_000_000]

const DEFAULT_PROXY = 'http://127.0.0.1:8787'
const LOOPBACK_HOSTS = new Set(['localhost', '127.0.0.1', '[::1]'])

/** The origin of `raw` when it is an http(s) URL on a loopback host with no userinfo, else undefined. */
export function loopbackOrigin(raw?: string): string | undefined {
  if (!raw) return undefined
  let url: URL
  try {
    url = new URL(raw)
  } catch {
    return undefined
  }
  const isHttp = url.protocol === 'http:' || url.protocol === 'https:'
  if (!isHttp || url.username || url.password || !LOOPBACK_HOSTS.has(url.hostname.toLowerCase())) {
    return undefined
  }

  return url.origin
}

/**
 * Where the Headroom proxy listens: the override, then the wrapped base URL, then the default port.
 * Each candidate must be a loopback URL; one that is not is skipped, so the mod never polls a remote host.
 */
export function proxyUrl(override?: string, baseUrl?: string): string {
  return loopbackOrigin(override) ?? loopbackOrigin(baseUrl) ?? DEFAULT_PROXY
}

type RawRequest = {
  request_id?: string | null
  timestamp?: string | null
  model?: string | null
  input_tokens_original?: number | null
  input_tokens_optimized?: number | null
  tokens_saved?: number | null
  savings_percent?: number | null
  transforms_applied?: string[] | null
  optimization_latency_ms?: number | null
}

/** The per-request rows of a loopback `/stats` payload, oldest first. */
export function parseRecent(text: string): Snip[] {
  let body: { recent_requests?: RawRequest[] }
  try {
    body = JSON.parse(text)
  } catch {
    return []
  }
  const rows = Array.isArray(body?.recent_requests) ? body.recent_requests : []

  return rows.flatMap(row => {
    const original = row.input_tokens_original
    const optimized = row.input_tokens_optimized
    if (typeof original !== 'number' || typeof optimized !== 'number' || !row.request_id) {
      return []
    }
    const saved = Math.max(0, row.tokens_saved ?? original - optimized)
    const percent = original > 0 ? (saved / original) * 100 : 0

    return [
      {
        id: String(row.request_id),
        at: row.timestamp ?? '',
        model: row.model ?? '',
        original,
        optimized,
        saved,
        percent,
        transforms: Array.isArray(row.transforms_applied) ? row.transforms_applied : [],
        latencyMs: typeof row.optimization_latency_ms === 'number' ? row.optimization_latency_ms : null,
      },
    ]
  })
}

type Kind = 'cut' | 'kept'

function describe(transform: string): { kind: Kind; label: string } | null {
  const t = transform.toLowerCase()
  if (t.startsWith('router:protected:')) {
    return { kind: 'kept', label: t.slice('router:protected:'.length).replace(/_/g, ' ') }
  }
  if (t.startsWith('router:excluded')) return { kind: 'kept', label: 'excluded tool' }
  if (t.startsWith('netcost:skip')) return { kind: 'kept', label: 'not worth a cut' }
  if (t.startsWith('router:netcost')) return null
  if (t.startsWith('smart:') || t.includes('smart_crush') || t.includes('json')) return { kind: 'cut', label: 'JSON crush' }
  if (t.startsWith('kompress') || t.includes('kompress')) return { kind: 'cut', label: 'Kompress text' }
  if (t.includes('cache_breakpoint') || t.includes('cache_align') || t.includes('dynamic_elements')) {
    return { kind: 'cut', label: 'cache align' }
  }
  if (t.startsWith('read_maturation')) return { kind: 'cut', label: 'stale reads' }
  if (t.includes('whitespace')) return { kind: 'cut', label: 'whitespace' }
  if (t.includes('code')) return { kind: 'cut', label: 'code AST' }
  if (t.includes('log')) return { kind: 'cut', label: 'log squash' }
  if (t.includes('search') || t.includes('grep')) return { kind: 'cut', label: 'search trim' }
  if (t.includes('diff')) return { kind: 'cut', label: 'diff trim' }
  if (t.includes('html')) return { kind: 'cut', label: 'HTML strip' }
  if (t.includes('text')) return { kind: 'cut', label: 'text trim' }
  if (t === 'processed_content_blocks') return null

  return { kind: 'cut', label: (t.split(':')[0] ?? t).replace(/_/g, ' ') }
}

/** Friendly names for what Headroom did ("JSON crush ×2") and what it left alone on purpose. */
export function explain(transforms: readonly string[]): { cuts: string[]; kept: string[] } {
  const counts = { cut: new Map<string, number>(), kept: new Map<string, number>() }
  for (const one of transforms) {
    const d = describe(one)
    if (d) counts[d.kind].set(d.label, (counts[d.kind].get(d.label) ?? 0) + 1)
  }
  const list = (m: Map<string, number>) =>
    [...m.entries()].sort((a, b) => b[1] - a[1]).map(([label, n]) => (n > 1 ? `${label} ×${n}` : label))

  return { cuts: list(counts.cut), kept: list(counts.kept) }
}

export function fmt(n: number): string {
  const abs = Math.abs(n)
  if (abs >= 1_000_000) return `${(n / 1_000_000).toFixed(abs >= 10_000_000 ? 0 : 1)}M`
  if (abs >= 10_000) return `${Math.round(n / 1000)}k`
  if (abs >= 1000) return `${(n / 1000).toFixed(1)}k`

  return String(Math.round(n))
}

export function pages(tokens: number): string {
  const p = tokens / TOKENS_PER_PAGE
  if (p < 1) return 'less than a page'

  return `${p < 10 ? p.toFixed(1) : Math.round(p)} pages`
}

export function addToTotals(totals: Totals, snips: readonly Snip[]): Totals {
  return snips.reduce<Totals>(
    (t, s) => ({
      original: t.original + s.original,
      optimized: t.optimized + s.optimized,
      saved: t.saved + s.saved,
      requests: t.requests + 1,
      biggest: !t.biggest || s.saved > t.biggest.saved ? s : t.biggest,
    }),
    totals,
  )
}

/** The milestone a step from `before` to `after` saved tokens crosses, if any. */
export function crossed(before: number, after: number): number | undefined {
  return [...MILESTONES].reverse().find(m => before < m && after >= m)
}

export type Tone = 'kept' | 'doomed' | 'blade' | 'crumb' | 'gone'
export type Segment = { tone: Tone; text: string }

const CRUMBS = ['⠁', '⠂', '⠄', '⡀', '⠠', '⠐', '⠈', '⢀']

const ease = (x: number) => 1 - Math.pow(1 - x, 3)

/**
 * One frame of the snip: the bar is the request as it arrived, `width` cells;
 * the blade travels from the right end to where the kept part ends, the cut
 * part crumbles behind it, and at `frame >= FRAMES` only dust is left.
 */
export function bar(width: number, original: number, optimized: number, frame: number): Segment[] {
  const w = Math.max(4, Math.floor(width))
  const ratio = original > 0 ? Math.min(1, Math.max(0, optimized / original)) : 1
  const kept = Math.max(original > 0 && optimized > 0 ? 1 : 0, Math.round(w * ratio))
  const p = ease(Math.min(1, Math.max(0, frame / FRAMES)))
  const done = frame >= FRAMES
  const blade = done ? -1 : Math.min(w - 1, Math.round(w - (w - kept) * p))
  const cells: Segment[] = []

  for (let i = 0; i < w; i++) {
    if (i < kept) cells.push({ tone: 'kept', text: '━' })
    else if (done) cells.push({ tone: 'gone', text: '·' })
    else if (i < blade) cells.push({ tone: 'doomed', text: '━' })
    else if (i === blade) cells.push({ tone: 'blade', text: '✂' })
    else {
      const fallen = frame - Math.round(((w - i) / Math.max(1, w - kept)) * FRAMES * 0.6)
      cells.push(fallen > 6 ? { tone: 'gone', text: '·' } : { tone: 'crumb', text: CRUMBS[(i * 7 + frame) % CRUMBS.length] ?? '·' })
    }
  }

  return cells.reduce<Segment[]>((runs, c) => {
    const last = runs[runs.length - 1]
    if (last && last.tone === c.tone) last.text += c.text
    else runs.push({ ...c })

    return runs
  }, [])
}

/** The token count shown while the blade travels: from the original down to what was sent. */
export function counter(original: number, optimized: number, frame: number): number {
  const p = ease(Math.min(1, Math.max(0, frame / FRAMES)))

  return Math.round(original - (original - optimized) * p)
}
