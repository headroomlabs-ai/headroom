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
}

export type Totals = {
  original: number
  optimized: number
  saved: number
  requests: number
  biggest: Snip | null
}

export type Anim = { id: string; frame: number }

export type Proxy = { url: string; isUp: boolean | null }

declare module 'claude-code' {
  interface PluginState {
    'headroom-snip': {
      feed: Snip[]
      totals: Totals
      anim: Anim | null
      proxy: Proxy
      isHidden: boolean
    }
  }
}
