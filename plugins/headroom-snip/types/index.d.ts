export type Snip = {
  id: string
  at: string
  model: string
  original: number
  optimized: number
  saved: number
  percent: number
  transforms: string[]
  latencyMs: number | null
  /** The proxy's project tag for the request: a name, null when logged without one, absent when unknown. */
  project?: string | null
}

export type Totals = {
  original: number
  optimized: number
  saved: number
  requests: number
  biggest: Snip | null
}

export type Anim = { id: string; frame: number }

/** `project` is the X-Headroom-Project this session sends; null means the counts are proxy-wide. */
export type Proxy = { url: string; isUp: boolean | null; project: string | null }

declare module 'claude-code' {
  interface PluginState {
    'headroom-snip': {
      feed: Snip[]
      totals: Totals
      anim: Anim | null
      proxy: Proxy
      isHidden: boolean
      startedAt: number | null
    }
  }
}
