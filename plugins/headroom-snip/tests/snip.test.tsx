import { expect, mock, test } from 'claude-code/testing'
import type { On } from 'claude-code'

import { bar, explain, FRAMES, parseRecent, proxyUrl } from '../hooks/snip'

const row = (id: string, original: number, optimized: number, transforms: string[]) => ({
  request_id: id,
  timestamp: '2026-10-04T12:00:00',
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

const stage = (on: On, rows: () => unknown[]) => {
  const clock = mock.clock(on)
  mock.store(on)
  mock.env(on, {})
  on('http.fetch', () => ({
    value: { status: 200, ok: true, headers: {}, text: JSON.stringify({ recent_requests: rows() }) },
  }))
  on('session.start', (_$, e) => ({ cwd: e.cwd }))
  on('turn.start', (_$, e) => ({ turnId: e.turnId }))
  on('command.register', (_$, e) => ({ value: { command: e.name } }))
  on('ui.status', () => ({ value: undefined }))
  on('ui.toast', () => ({ value: undefined }))
  on('ui.render', ($, e) => {
    const { Box } = $.ui.resolve(e)
    return <Box />
  })

  return clock
}

const BAND = {
  component: 'AbovePrompt',
  props: { hasSurvey: false, isWorking: false, maxRows: 10, bodyColumns: 100 } as never,
} as const

test('a request made during a turn is snipped in the band', async ($, on) => {
  let rows: unknown[] = [row('old', 5000, 5000, [])]
  const clock = stage(on, () => rows)

  await $.session.start({ cwd: '/w', surface: 'terminal', isInteractive: true } as never)
  await clock.settle()
  rows = [...rows, row('r1', 20_000, 4_000, ['smart:array', 'kompress:tool:0.40'])]
  await $.turn.start({ text: 'hi', turnId: 't1' })
  await clock.advance(1000 + 45 * (FRAMES + 2))

  for (const surface of ['terminal', 'desktop'] as const) {
    const ui = await $.ui.mount({ plugin: 'headroom-snip', surface, ...BAND })
    expect((await ui.find({ text: /20k → 4\.0k/ }))).toBeDefined()
    expect((await ui.find({ text: /−80%/ }))).toBeDefined()
    expect((await ui.find({ text: /JSON crush · Kompress text/ }))).toBeDefined()
    expect((await ui.find({ text: /session 16k saved/ }))).toBeDefined()
    await ui.press({ key: 'hide' })
    expect(await ui.find({ text: /headroom/ })).toBeUndefined()
    await ui.unmount()
    await $.command.run({ command: 'headroom', args: 'show' } as never)
  }
})

test('a missing proxy says how to start one', async ($, on) => {
  const clock = mock.clock(on)
  mock.store(on)
  mock.env(on, {})
  on('http.fetch', () => {
    throw new Error('ECONNREFUSED')
  })
  on('session.start', (_$, e) => ({ cwd: e.cwd }))
  on('command.register', (_$, e) => ({ value: { command: e.name } }))
  on('ui.render', ($, e) => {
    const { Box } = $.ui.resolve(e)
    return <Box />
  })

  await $.session.start({ cwd: '/w', surface: 'terminal', isInteractive: true } as never)
  await clock.settle()
  const ui = await $.ui.mount({ plugin: 'headroom-snip', surface: 'terminal', ...BAND })
  expect(await ui.find({ text: /headroom wrap claude/ })).toBeDefined()
})
