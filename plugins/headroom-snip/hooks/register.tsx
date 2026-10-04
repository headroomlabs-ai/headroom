import { atom, read, update } from 'claude-code'
import type { EngineInterface, Register, Timer } from 'claude-code'

import type { Anim, Proxy, Snip, Totals } from '../types'
import {
  FRAMES,
  FRAME_MS,
  addToTotals,
  bar,
  counter,
  crossed,
  explain,
  fmt,
  pages,
  parseRecent,
  proxyUrl,
} from './snip'
import type { Tone } from './snip'

const PANE = 'headroom-snip'
const POLL_MS = 1000
const TAIL_TICKS = 4

const feed = atom({ plugin: 'headroom-snip', key: 'feed' } as const, [])
const totals = atom({ plugin: 'headroom-snip', key: 'totals' } as const, {
  original: 0,
  optimized: 0,
  saved: 0,
  requests: 0,
  biggest: null,
})
const anim = atom({ plugin: 'headroom-snip', key: 'anim' } as const, null)
const proxy = atom({ plugin: 'headroom-snip', key: 'proxy' } as const, { url: '', isUp: null })
const isHidden = atom({ plugin: 'headroom-snip', key: 'isHidden' } as const, false)

const TONE: Record<Tone, { color?: string; dimColor?: boolean; bold?: boolean }> = {
  kept: { color: '#4ade80' },
  doomed: { color: '#f59e0b' },
  blade: { color: '#f472b6', bold: true },
  crumb: { color: '#a8a29e' },
  gone: { dimColor: true },
}

const pct = (t: Pick<Totals, 'saved' | 'original'>) =>
  t.original > 0 ? Math.round((t.saved / t.original) * 100) : 0

// The module's own bookkeeping; a reload starts it over while $.state keeps the drawing's values.
const live = {
  seen: new Set<string>(),
  url: proxyUrl(),
  poller: undefined as Timer | undefined,
  animTimer: undefined as Timer | undefined,
  tail: 0,
  isBaselined: false,
  isPolling: false,
}

async function setProxy($: EngineInterface, next: Proxy) {
  const now = await read($, proxy)
  if (now.isUp !== next.isUp || now.url !== next.url) await update($, proxy, () => next)
}

async function snipAnimation($: EngineInterface, id: string) {
  live.animTimer?.cancel()
  let frame = 0
  await update($, anim, (): Anim => ({ id, frame }))
  live.animTimer = $.clock.every(FRAME_MS, () => {
    frame += 1
    if (frame >= FRAMES) live.animTimer?.cancel()
    void update($, anim, (): Anim => ({ id, frame }))
  })
}

async function land($: EngineInterface, fresh: Snip[]) {
  const before = (await read($, totals)).saved
  await update($, feed, list => [...list, ...fresh].slice(-50))
  const after = await update($, totals, t => addToTotals(t, fresh))
  $.ui.status(`✂ ${fmt(after.saved)} tok saved · ${pct(after)}%`)

  const milestone = crossed(before, after.saved)
  if (milestone !== undefined) {
    $.ui.toast(`✂ ${fmt(milestone)} tokens snipped this session: ${pages(milestone)} Claude didn't have to reread`)
  }

  const added = fresh.reduce((n, s) => n + s.saved, 0)
  const lifetime = Number((await $.store.get('lifetimeSaved')) ?? 0)
  await $.store.set('lifetimeSaved', lifetime + added)

  const showcase = [...fresh].reverse().find(s => s.saved > 0) ?? fresh[fresh.length - 1]
  if (showcase && showcase.saved > 0) await snipAnimation($, showcase.id)
}

async function poll($: EngineInterface) {
  if (live.isPolling) return
  live.isPolling = true
  try {
    const res = await $.http.fetch(`${live.url}/stats?cached=1`)
    if (!res.ok) {
      await setProxy($, { url: live.url, isUp: false })
      return
    }
    await setProxy($, { url: live.url, isUp: true })
    for (const s of await read($, feed)) live.seen.add(s.id)
    const fresh = parseRecent(res.text).filter(s => !live.seen.has(s.id))
    fresh.forEach(s => live.seen.add(s.id))
    // Requests the proxy served before this session are history, not this session's snips.
    if (!live.isBaselined) {
      live.isBaselined = true
      return
    }
    if (fresh.length > 0) await land($, fresh)
  } catch {
    await setProxy($, { url: live.url, isUp: false })
  } finally {
    live.isPolling = false
  }
}

function keepPolling($: EngineInterface) {
  live.tail = Number.POSITIVE_INFINITY
  live.poller ??= $.clock.every(POLL_MS, () => {
    live.tail -= 1
    if (live.tail <= 0) {
      live.poller?.cancel()
      live.poller = undefined
    }
    void poll($)
  })
}

export const register: Register = on => {
  on('session.start', async ($, e, next) => {
    live.url = proxyUrl(await $.env.get('HEADROOM_PROXY_URL'), await $.env.get('ANTHROPIC_BASE_URL'))
    await $.command.register({
      name: 'headroom',
      description: 'Show what Headroom snipped from each request (add "hide" or "show" for the band)',
    })
    void poll($)

    return next(e)
  })

  on('turn.start', async ($, e, next) => {
    keepPolling($)

    return next(e)
  })

  on('turn.complete', async ($, e, next) => {
    live.tail = TAIL_TICKS
    void poll($)

    return next(e)
  })

  on('command.run', { command: 'headroom' }, async ($, e) => {
    const arg = e.args.trim().toLowerCase()
    if (arg === 'hide' || arg === 'show') {
      await update($, isHidden, () => arg === 'hide')
      return { text: `Headroom snip band ${arg === 'hide' ? 'hidden' : 'shown'}.` }
    }
    await poll($)
    await $.ui.open({ id: PANE, title: '✂ Headroom' })

    return { text: 'Opened the Headroom snip log.' }
  })

  on('ui.render', { component: 'AbovePrompt' }, async ($, e, next) => {
    const [list, sum, now, px, hidden] = await Promise.all([
      read($, feed),
      read($, totals),
      read($, anim),
      read($, proxy),
      read($, isHidden),
    ])
    if (e.props.hasSurvey || hidden) return next(e)

    const { Box, Button, Text } = $.ui.resolve(e)
    const columns = e.props.bodyColumns ?? e.viewport?.columns ?? 80
    const hide = <Button key="hide" label="hide" plain onPress={() => update($, isHidden, () => true)} />

    if (px.isUp === false) {
      return (
        <Box>
          <Text color="#f472b6">✂ </Text>
          <Text dimColor wrap="truncate-end">
            headroom isn't in the loop: nothing at {px.url}. Run `headroom wrap claude` to start snipping.{' '}
          </Text>
          {hide}
        </Box>
      )
    }

    const last = list[list.length - 1]
    if (!last) return next(e)

    const frame = now && now.id === last.id ? now.frame : FRAMES
    const width = Math.max(10, Math.min(40, columns - 46))
    const { cuts, kept } = explain(last.transforms)
    const isPassThrough = last.saved <= 0

    return (
      <Box flexDirection="column">
        <Box>
          <Text color="#f472b6" bold>
            ✂ headroom{' '}
          </Text>
          {isPassThrough ? (
            <Text dimColor wrap="truncate-end">
              {fmt(last.original)} tok sent as-is ({kept.length > 0 ? `kept: ${kept.join(', ')}` : 'nothing worth a snip'})
            </Text>
          ) : (
            <Box>
              <Text>
                {fmt(last.original)} → <Text bold>{fmt(counter(last.original, last.optimized, frame))}</Text> tok{' '}
              </Text>
              {bar(width, last.original, last.optimized, frame).map((seg, i) => (
                <Text key={`b${i}`} {...TONE[seg.tone]}>
                  {seg.text}
                </Text>
              ))}
              <Text color="#4ade80" bold>
                {' '}
                −{Math.round(last.percent)}%
              </Text>
            </Box>
          )}
        </Box>
        <Box>
          <Text dimColor wrap="truncate-end">
            {'  '}
            {cuts.length > 0 ? `${cuts.slice(0, 3).join(' · ')}   ` : ''}session {fmt(sum.saved)} saved ({pct(sum)}%) over{' '}
            {sum.requests} req ≈ {pages(sum.saved)}{' '}
          </Text>
          <Button key="details" label="details" plain onPress={() => $.ui.open({ id: PANE, title: '✂ Headroom' })} />
          <Text dimColor> </Text>
          {hide}
        </Box>
      </Box>
    )
  })

  on('ui.render', { component: 'Pane', requestId: PANE }, async ($, e) => {
    const [list, sum, px, hidden] = await Promise.all([read($, feed), read($, totals), read($, proxy), read($, isHidden)])
    const lifetime = Number((await $.store.get('lifetimeSaved')) ?? 0)
    const { Box, Button, Text } = $.ui.resolve(e)
    const columns = e.props.bodyColumns ?? e.viewport?.columns ?? 60
    const room = Math.max(1, Math.floor(((e.viewport?.rows ?? 24) - 9) / 2))
    const width = Math.max(8, Math.min(24, columns - 34))

    return (
      <Box flexDirection="column">
        <Text>
          <Text color="#f472b6" bold>
            ✂ {fmt(sum.saved)}
          </Text>{' '}
          tokens snipped this session ({pct(sum)}% of {fmt(sum.original)}) over {sum.requests} requests
        </Text>
        <Text dimColor>
          ≈ {pages(sum.saved)} Claude didn't have to reread · {fmt(lifetime)} all-time
        </Text>
        {sum.biggest && (
          <Text dimColor wrap="truncate-end">
            biggest snip: {fmt(sum.biggest.saved)} tok (−{Math.round(sum.biggest.percent)}%)
            {explain(sum.biggest.transforms).cuts.length > 0 ? ` by ${explain(sum.biggest.transforms).cuts.join(', ')}` : ''}
          </Text>
        )}
        <Text dimColor>{px.isUp === false ? `proxy unreachable at ${px.url}` : `proxy ${px.url}`}</Text>
        <Text> </Text>
        {list.length === 0 && <Text dimColor>No requests through Headroom yet. Send a prompt.</Text>}
        {list
          .slice(-room)
          .reverse()
          .map(s => {
            const { cuts, kept } = explain(s.transforms)
            return (
              <Box key={s.id} flexDirection="column">
                <Box>
                  <Text dimColor>{(s.at.split('T')[1] ?? s.at).slice(0, 8).padEnd(9)}</Text>
                  {bar(width, s.original, s.optimized, FRAMES).map((seg, i) => (
                    <Text key={`b${i}`} {...TONE[seg.tone]}>
                      {seg.text}
                    </Text>
                  ))}
                  <Text>
                    {' '}
                    {fmt(s.original)}→{fmt(s.optimized)}
                  </Text>
                  <Text color={s.saved > 0 ? '#4ade80' : undefined} dimColor={s.saved <= 0}>
                    {' '}
                    −{Math.round(s.percent)}%
                  </Text>
                </Box>
                <Text dimColor wrap="truncate-end">
                  {'         '}
                  {cuts.length > 0 ? cuts.join(' · ') : 'sent as-is'}
                  {kept.length > 0 ? `  (kept: ${kept.join(', ')})` : ''}
                  {s.latencyMs !== null ? `  ${Math.round(s.latencyMs)}ms` : ''}
                </Text>
              </Box>
            )
          })}
        <Text> </Text>
        <Box>
          <Button
            key="band"
            label={hidden ? 'show band' : 'hide band'}
            onPress={() => update($, isHidden, h => !h)}
          />
        </Box>
      </Box>
    )
  })
}
