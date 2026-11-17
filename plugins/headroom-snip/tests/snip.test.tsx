import { expect, mock, test } from 'claude-code/testing'
import type { On } from 'claude-code'

import { bar, crossed, explain, FRAMES, isOwn, ownProject, parseRecent, proxyUrl } from '../hooks/snip'

// Session start in the tests; the proxy stamps requests in UTC ISO 8601.
const T0 = Date.parse('2026-10-04T12:00:00+00:00')
const stamp = (offsetSeconds: number) => new Date(T0 + offsetSeconds * 1000).toISOString().replace('Z', '+00:00')

const row = (id: string, original: number, optimized: number, transforms: string[], at = 5) => ({
  request_id: id,
  timestamp: stamp(at),
  model: 'claude-sonnet',
  input_tokens_original: original,
  input_tokens_optimized: optimized,
  tokens_saved: original - optimized,
  transforms_applied: transforms,
})

test('proxy url follows the wrapped base url only when it is local', async () => {
  expect(proxyUrl(undefined, 'http://127.0.0.1:9999/v1')).toBe('http://127.0.0.1:9999')
  expect(proxyUrl(undefined, 'https://api.anthropic.com')).toBe('http://127.0.0.1:8787')
  expect(proxyUrl('http://localhost:1234', 'http://127.0.0.1:9999')).toBe('http://localhost:1234')
})

test('valid loopback urls keep their origin', async () => {
  expect(proxyUrl(undefined, 'http://localhost:8787')).toBe('http://localhost:8787')
  expect(proxyUrl(undefined, 'HTTP://LOCALHOST:8787/v1')).toBe('http://localhost:8787')
  expect(proxyUrl(undefined, 'https://127.0.0.1:8443/p/my-project')).toBe('https://127.0.0.1:8443')
  expect(proxyUrl(undefined, 'http://[::1]:8787/v1')).toBe('http://[::1]:8787')
  expect(proxyUrl('http://[::1]:9000', undefined)).toBe('http://[::1]:9000')
})

test('hosts that only look local are never polled', async () => {
  const remote = [
    'https://localhost.attacker.example/v1',
    'http://127.0.0.1.attacker.example/v1',
    'http://localhost@attacker.example/v1',
    'http://127.0.0.1:8787@attacker.example',
    'http://attacker.example/?localhost',
    'http://attacker.example#127.0.0.1',
    'http://[::1].attacker.example',
    'http://localhost:8787.attacker.example',
  ]
  for (const url of remote) {
    expect(proxyUrl(undefined, url)).toBe('http://127.0.0.1:8787')
    expect(proxyUrl(url, undefined)).toBe('http://127.0.0.1:8787')
  }
})

test('userinfo, other schemes and junk are refused even on loopback', async () => {
  for (const url of [
    'http://user:pass@127.0.0.1:8787',
    'http://user@localhost:8787',
    'ftp://127.0.0.1:8787',
    'file:///tmp/sock',
    'javascript:alert(1)',
    '127.0.0.1:8787',
    'not a url',
    '',
  ]) {
    expect(proxyUrl(url, undefined)).toBe('http://127.0.0.1:8787')
  }
})

test('a remote override falls back to the local base url, not the remote host', async () => {
  expect(proxyUrl('https://headroom.example.com', 'http://127.0.0.1:9999/v1')).toBe('http://127.0.0.1:9999')
})

test('transforms read as plain words', async () => {
  const { cuts, kept } = explain(['smart:array', 'smart:dict', 'router:protected:user_message', 'inserted_2_cache_breakpoints'])
  expect(cuts).toEqual(['JSON crush ×2', 'cache align'])
  expect(kept).toEqual(['user message'])
})

test('the finished bar keeps the sent share and dusts the rest', async () => {
  const done = bar(10, 1000, 300, FRAMES)
  expect(done).toEqual([
    { tone: 'kept', text: '━━━' },
    { tone: 'gone', text: '·······' },
  ])
  const mid = bar(10, 1000, 300, FRAMES / 4)
  expect(mid.some(s => s.tone === 'blade')).toBe(true)
  expect(parseRecent('not json')).toEqual([])
})

test('rows come back oldest first, with their project tags', async () => {
  const text = JSON.stringify({
    recent_requests: [row('new', 3000, 1000, [], 9), row('old', 2000, 1000, [], 1), row('bare', 100, 100, [], 2)],
    request_logs: [
      { request_id: 'old', tags: { project: 'headroom' } },
      { request_id: 'bare', tags: {} },
    ],
  })
  const rows = parseRecent(text)
  expect(rows.map(r => r.id)).toEqual(['bare', 'old', 'new'])
  expect(rows.map(r => r.project)).toEqual([null, 'headroom', undefined])
})

test('the session project is read from the wrapped custom headers', async () => {
  expect(ownProject('X-Other: 1\nX-Headroom-Project: my%20app\nX-Headroom-Cwd: /x')).toBe('my app')
  expect(ownProject('x-headroom-project:headroom')).toBe('headroom')
  expect(ownProject('X-Headroom-Project: 100%')).toBe('100%')
  expect(ownProject(undefined)).toBe(null)
  expect(ownProject('X-Headroom-Project:   ')).toBe(null)
})

test('a request is this session\'s by its stamp and project', async () => {
  const snip = parseRecent(JSON.stringify({ recent_requests: [row('a', 10, 5, [], 3)] }))[0]!
  expect(isOwn(snip, T0, null)).toBe(true)
  expect(isOwn(snip, T0 + 60_000, null)).toBe(false)
  expect(isOwn({ ...snip, project: 'other' }, T0, 'mine')).toBe(false)
  expect(isOwn({ ...snip, project: 'mine' }, T0, 'mine')).toBe(true)
  // Logged without a project while this session sends one: another, unwrapped client.
  expect(isOwn({ ...snip, project: null }, T0, 'mine')).toBe(false)
  expect(isOwn({ ...snip, project: 'mine' }, T0, null)).toBe(true)
  // No tag known for the row (outside the proxy's tagged log tail): not counted for a tagged session.
  expect(isOwn(snip, T0, 'mine')).toBe(false)
  expect(isOwn(snip, T0, null)).toBe(true)
  // Stamped before the session started, however slightly: history.
  expect(isOwn({ ...snip, at: stamp(-0.5) }, T0, null)).toBe(false)
  expect(isOwn({ ...snip, at: stamp(0) }, T0, null)).toBe(true)
  expect(isOwn({ ...snip, at: '' }, T0, null)).toBeUndefined()
})

test('every milestone a step crosses is announced, lowest first', async () => {
  expect(crossed(0, 70_000)).toEqual([10_000, 50_000])
  expect(crossed(9_000, 9_500)).toEqual([])
  expect(crossed(10_000, 10_001)).toEqual([])
})

type Stage = {
  rows?: () => unknown[]
  logs?: () => unknown[]
  env?: Record<string, string>
  isDown?: () => boolean
}

const stage = (on: On, { rows = () => [], logs = () => [], env = {}, isDown = () => false }: Stage) => {
  const clock = mock.clock(on, { now: T0 })
  const seen = { fetches: 0, toasts: [] as string[] }
  mock.store(on)
  mock.env(on, env)
  on('http.fetch', () => {
    seen.fetches += 1
    if (isDown()) throw new Error('ECONNREFUSED')
    // The proxy lists recent_requests newest first.
    const text = JSON.stringify({ recent_requests: [...rows()].reverse(), request_logs: logs() })
    return { value: { status: 200, ok: true, headers: {}, text } }
  })
  on('session.start', (_$, e) => ({ cwd: e.cwd }))
  on('turn.start', (_$, e) => ({ turnId: e.turnId }))
  on('turn.complete', () => ({ text: '' }))
  on('command.register', (_$, e) => ({ value: { command: e.name } }))
  on('ui.open', () => ({ value: undefined as never }))
  on('ui.status', () => ({ value: undefined }))
  on('ui.toast', (_$, e) => {
    seen.toasts.push(e.text)
    return { value: undefined }
  })
  on('ui.render', ($, e) => {
    const { Box } = $.ui.resolve(e)
    return <Box />
  })

  return { clock, seen }
}

const START = { cwd: '/w', surface: 'terminal', isInteractive: true } as never
const TURN_DONE = { answer: '', durationMs: 1, isAborted: false, turnId: 't1', reason: 'answer' } as never

const BAND = {
  component: 'AbovePrompt',
  props: { hasSurvey: false, isWorking: false, maxRows: 10, bodyColumns: 100 } as never,
} as const

const PANE = {
  component: 'Pane',
  requestId: 'headroom-snip',
  props: { title: '✂ Headroom', isFocused: false, bodyColumns: 80, placement: 'dock' } as never,
  viewport: { columns: 80, rows: 40 },
} as const

const settleSnip = (clock: { advance: (ms: number) => Promise<void> }) => clock.advance(1000 + 45 * (FRAMES + 2))

test('a request made during a turn is snipped in the band', async ($, on) => {
  let rows: unknown[] = [row('old', 5000, 5000, [], -30)]
  const { clock } = stage(on, { rows: () => rows })

  await $.session.start(START)
  await clock.settle()
  rows = [...rows, row('r1', 20_000, 4_000, ['smart:array', 'kompress:tool:0.40'])]
  await $.turn.start({ text: 'hi', turnId: 't1' })
  await settleSnip(clock)

  for (const surface of ['terminal', 'desktop'] as const) {
    const ui = await $.ui.mount({ plugin: 'headroom-snip', surface, ...BAND })
    expect(await ui.find({ text: /20k → 4\.0k/ })).toBeDefined()
    expect(await ui.find({ text: /−80%/ })).toBeDefined()
    expect(await ui.find({ text: /JSON crush · Kompress text/ })).toBeDefined()
    // No project header, so the total is labelled proxy-wide; the pre-session request is not in it.
    expect(await ui.find({ text: /proxy 16k saved \(80%\) over 1 req/ })).toBeDefined()
    await ui.press({ key: 'hide' })
    expect(await ui.find({ text: /headroom/ })).toBeUndefined()
    await ui.unmount()
    await $.command.run({ command: 'headroom', args: 'show' } as never)
  }
})

test('two new requests in one poll show the newest in the band and newest first in the pane', async ($, on) => {
  let rows: unknown[] = []
  const { clock } = stage(on, { rows: () => rows })

  await $.session.start(START)
  await clock.settle()
  rows = [row('first', 10_000, 9_000, [], 5), row('second', 30_000, 3_000, ['smart:array'], 6)]
  await $.turn.start({ text: 'hi', turnId: 't1' })
  await settleSnip(clock)

  const band = await $.ui.mount({ plugin: 'headroom-snip', surface: 'terminal', ...BAND })
  expect(await band.find({ text: /30k → 3\.0k/ })).toBeDefined()
  expect(await band.find({ text: /10k → 9\.0k/ })).toBeUndefined()
  await band.unmount()

  const pane = await $.ui.mount({ plugin: 'headroom-snip', surface: 'terminal', ...PANE })
  const lines = (await pane.findAll({ type: 'Text', text: /→/ })).map(t => t.text)
  const newest = lines.findIndex(t => /30k→3\.0k/.test(t))
  const oldest = lines.findIndex(t => /10k→9\.0k/.test(t))
  expect(newest).toBeGreaterThanOrEqual(0)
  expect(oldest).toBeGreaterThan(newest)
  expect(await pane.find({ text: /28k\b|28\.0k|29k/ })).toBeDefined()
  expect(await pane.find({ text: /over 2 requests/ })).toBeDefined()
  expect(await pane.find({ text: /counting every client on this proxy/ })).toBeDefined()
  await pane.unmount()
})

test('a proxy that comes up after the session started still counts the session\'s requests', async ($, on) => {
  let isDown = true
  let rows: unknown[] = []
  const { clock } = stage(on, { rows: () => rows, isDown: () => isDown })

  await $.session.start(START)
  await clock.settle()
  const down = await $.ui.mount({ plugin: 'headroom-snip', surface: 'terminal', ...BAND })
  expect(await down.find({ text: /headroom wrap claude/ })).toBeDefined()
  await down.unmount()

  // The proxy starts mid-turn; its first successful poll already holds an in-session request
  // alongside one it served before this session.
  await $.turn.start({ text: 'hi', turnId: 't1' })
  await clock.advance(1000)
  isDown = false
  rows = [row('before', 8_000, 2_000, [], -120), row('during', 12_000, 3_000, ['smart:array'], 2)]
  await settleSnip(clock)

  const ui = await $.ui.mount({ plugin: 'headroom-snip', surface: 'terminal', ...BAND })
  expect(await ui.find({ text: /12k → 3\.0k/ })).toBeDefined()
  expect(await ui.find({ text: /proxy 9\.0k saved \(75%\) over 1 req/ })).toBeDefined()
  await ui.unmount()
})

test('with a project header, other clients on the proxy are left out', async ($, on) => {
  let rows: unknown[] = []
  let logs: unknown[] = []
  const { clock } = stage(on, {
    rows: () => rows,
    logs: () => logs,
    env: { ANTHROPIC_CUSTOM_HEADERS: 'X-Headroom-Project: headroom\nX-Headroom-Cwd: /w' },
  })

  await $.session.start(START)
  await clock.settle()
  rows = [
    row('untagged', 50_000, 5_000, ['smart:array'], 2),
    row('mine', 20_000, 4_000, ['smart:array'], 3),
    row('theirs', 40_000, 4_000, ['smart:array'], 4),
  ]
  // 'untagged' fell outside the proxy's ten-row tagged log tail, so its project is unknown.
  logs = [
    { request_id: 'mine', tags: { project: 'headroom' } },
    { request_id: 'theirs', tags: { project: 'other-app' } },
  ]
  await $.turn.start({ text: 'hi', turnId: 't1' })
  await settleSnip(clock)

  const band = await $.ui.mount({ plugin: 'headroom-snip', surface: 'terminal', ...BAND })
  expect(await band.find({ text: /20k → 4\.0k/ })).toBeDefined()
  expect(await band.find({ text: /project 16k saved \(80%\) over 1 req/ })).toBeDefined()
  expect(await band.find({ text: /session/ })).toBeUndefined()
  await band.unmount()

  const pane = await $.ui.mount({ plugin: 'headroom-snip', surface: 'terminal', ...PANE })
  expect(await pane.find({ text: /tokens snipped in headroom since this session started/ })).toBeDefined()
  expect(await pane.find({ text: /counting every request tagged headroom, other sessions in that project included/ })).toBeDefined()
  expect(await pane.find({ text: /40k→4\.0k/ })).toBeUndefined()
  expect(await pane.find({ text: /50k→5\.0k/ })).toBeUndefined()
  await pane.unmount()
})

test('two sessions in one project share a count, and every label says project, not session', async ($, on) => {
  let rows: unknown[] = []
  let logs: unknown[] = []
  const { clock, seen } = stage(on, {
    rows: () => rows,
    logs: () => logs,
    env: { ANTHROPIC_CUSTOM_HEADERS: 'X-Headroom-Project: headroom' },
  })

  await $.session.start(START)
  await clock.settle()
  // 'this-session' and 'other-session' come from two Claude Code sessions launched in directories
  // named headroom; the proxy tags both the same, so both count.
  rows = [row('this-session', 40_000, 10_000, ['smart:array'], 3), row('other-session', 30_000, 6_000, ['smart:array'], 4)]
  logs = [
    { request_id: 'this-session', tags: { project: 'headroom' } },
    { request_id: 'other-session', tags: { project: 'headroom' } },
  ]
  await $.turn.start({ text: 'hi', turnId: 't1' })
  await settleSnip(clock)

  const band = await $.ui.mount({ plugin: 'headroom-snip', surface: 'terminal', ...BAND })
  expect(await band.find({ text: /project 54k saved \(77%\) over 2 req/ })).toBeDefined()
  expect(await band.find({ text: /session/ })).toBeUndefined()
  await band.unmount()

  const pane = await $.ui.mount({ plugin: 'headroom-snip', surface: 'terminal', ...PANE })
  expect(await pane.find({ text: /tokens snipped in headroom since this session started/ })).toBeDefined()
  expect(await pane.find({ text: /over 2 requests/ })).toBeDefined()
  await pane.unmount()

  const milestones = seen.toasts.filter(t => /tokens snipped/.test(t))
  expect(milestones).toEqual([
    expect.stringMatching(/^✂ 10k tokens snipped in headroom since this session started/),
    expect.stringMatching(/^✂ 50k tokens snipped in headroom since this session started/),
  ])
})

test('one big snip announces each milestone it crosses', async ($, on) => {
  let rows: unknown[] = []
  const { clock, seen } = stage(on, { rows: () => rows })

  await $.session.start(START)
  await clock.settle()
  rows = [row('huge', 90_000, 20_000, ['smart:array'])]
  await $.turn.start({ text: 'hi', turnId: 't1' })
  await settleSnip(clock)

  const milestones = seen.toasts.filter(t => /tokens snipped/.test(t))
  expect(milestones.length).toBe(2)
  expect(milestones[0]).toMatch(/^✂ 10k tokens snipped/)
  expect(milestones[1]).toMatch(/^✂ 50k tokens snipped/)
})

test('polling picks up a request that lands just after the turn, then stops', async ($, on) => {
  let rows: unknown[] = []
  const { clock, seen } = stage(on, { rows: () => rows })

  await $.session.start(START)
  await clock.settle()
  await $.turn.start({ text: 'hi', turnId: 't1' })
  await clock.advance(2000)
  await $.turn.complete(TURN_DONE)
  rows = [row('late', 6_000, 1_500, ['smart:array'])]
  await settleSnip(clock)

  const ui = await $.ui.mount({ plugin: 'headroom-snip', surface: 'terminal', ...BAND })
  expect(await ui.find({ text: /6\.0k → 1\.5k/ })).toBeDefined()
  await ui.unmount()

  // The tail runs out four seconds after the turn; after that nothing polls.
  await clock.advance(5000)
  const before = seen.fetches
  await clock.advance(10_000)
  expect(seen.fetches).toBe(before)
})
