'use client'

import { useState } from 'react'
import {
  AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from 'recharts'

// --- Embedded telemetry data ---
const DATA = {
  total_tokens_saved: 41750654085,
  total_cost_saved: 176635.62,
  total_requests: 1194154,
  unique_instances: 889,
  active_days: 14,
  daily_stats: [
    { date: '2026-03-30', requests: 1050, instances: 17, cost_saved: 42.29, tokens_saved: 7560502 },
    { date: '2026-03-31', requests: 26526, instances: 149, cost_saved: 1904.15, tokens_saved: 1004245860 },
    { date: '2026-04-01', requests: 53480, instances: 117, cost_saved: 3353.50, tokens_saved: 860455498 },
    { date: '2026-04-02', requests: 64906, instances: 123, cost_saved: 18007.10, tokens_saved: 3431477414 },
    { date: '2026-04-03', requests: 99872, instances: 162, cost_saved: 29238.20, tokens_saved: 6159073542 },
    { date: '2026-04-04', requests: 90992, instances: 147, cost_saved: 12863.30, tokens_saved: 1864667449 },
    { date: '2026-04-05', requests: 162541, instances: 142, cost_saved: 49802.60, tokens_saved: 11990261722 },
    { date: '2026-04-06', requests: 127494, instances: 174, cost_saved: 11533.70, tokens_saved: 3850552409 },
    { date: '2026-04-07', requests: 158840, instances: 201, cost_saved: 14260.90, tokens_saved: 4113326538 },
    { date: '2026-04-08', requests: 84459, instances: 186, cost_saved: 8383.12, tokens_saved: 2317153362 },
    { date: '2026-04-09', requests: 137856, instances: 208, cost_saved: 10663.80, tokens_saved: 2058570508 },
    { date: '2026-04-10', requests: 111043, instances: 192, cost_saved: 9893.92, tokens_saved: 1979992305 },
    { date: '2026-04-11', requests: 65851, instances: 147, cost_saved: 5853.92, tokens_saved: 1521911387 },
    { date: '2026-04-12', requests: 9244, instances: 27, cost_saved: 835.12, tokens_saved: 591405589 },
  ],
  hourly_stats: [
    { hour: '2026-04-10 06:00', requests: 8548, instances: 9, cost_saved: 577.12, tokens_saved: 198669219 },
    { hour: '2026-04-10 07:00', requests: 8289, instances: 18, cost_saved: 456.34, tokens_saved: 172834374 },
    { hour: '2026-04-10 08:00', requests: 8066, instances: 23, cost_saved: 559.56, tokens_saved: 196181344 },
    { hour: '2026-04-10 09:00', requests: 4890, instances: 16, cost_saved: 206.95, tokens_saved: 124914685 },
    { hour: '2026-04-10 10:00', requests: 9531, instances: 21, cost_saved: 421.51, tokens_saved: 236470420 },
    { hour: '2026-04-10 11:00', requests: 5170, instances: 13, cost_saved: 174.45, tokens_saved: 114113494 },
    { hour: '2026-04-10 12:00', requests: 10982, instances: 20, cost_saved: 388.05, tokens_saved: 166308102 },
    { hour: '2026-04-10 13:00', requests: 8001, instances: 16, cost_saved: 228.53, tokens_saved: 134921875 },
    { hour: '2026-04-10 14:00', requests: 4950, instances: 15, cost_saved: 208.56, tokens_saved: 119763470 },
    { hour: '2026-04-10 15:00', requests: 9000, instances: 17, cost_saved: 1739.51, tokens_saved: 166911747 },
    { hour: '2026-04-10 16:00', requests: 12569, instances: 17, cost_saved: 1130.82, tokens_saved: 312958735 },
    { hour: '2026-04-10 17:00', requests: 12057, instances: 17, cost_saved: 443.07, tokens_saved: 203763560 },
    { hour: '2026-04-10 18:00', requests: 14011, instances: 18, cost_saved: 1018.06, tokens_saved: 294928335 },
    { hour: '2026-04-10 19:00', requests: 12599, instances: 17, cost_saved: 915.21, tokens_saved: 327535343 },
    { hour: '2026-04-10 20:00', requests: 10522, instances: 16, cost_saved: 2484.87, tokens_saved: 206735552 },
    { hour: '2026-04-10 21:00', requests: 7125, instances: 16, cost_saved: 928.27, tokens_saved: 289265979 },
    { hour: '2026-04-10 22:00', requests: 4481, instances: 9, cost_saved: 190.60, tokens_saved: 137301814 },
    { hour: '2026-04-10 23:00', requests: 4033, instances: 5, cost_saved: 154.82, tokens_saved: 126989500 },
    { hour: '2026-04-11 00:00', requests: 8998, instances: 12, cost_saved: 770.90, tokens_saved: 254824094 },
    { hour: '2026-04-11 01:00', requests: 13729, instances: 12, cost_saved: 668.12, tokens_saved: 330717273 },
    { hour: '2026-04-11 02:00', requests: 3738, instances: 2, cost_saved: 152.18, tokens_saved: 126547280 },
    { hour: '2026-04-11 03:00', requests: 4449, instances: 4, cost_saved: 185.03, tokens_saved: 134597314 },
    { hour: '2026-04-11 04:00', requests: 5961, instances: 9, cost_saved: 303.65, tokens_saved: 157734893 },
    { hour: '2026-04-11 05:00', requests: 6550, instances: 8, cost_saved: 248.84, tokens_saved: 158834743 },
    { hour: '2026-04-11 06:00', requests: 5429, instances: 8, cost_saved: 268.11, tokens_saved: 155129772 },
    { hour: '2026-04-11 07:00', requests: 5671, instances: 11, cost_saved: 225.25, tokens_saved: 143244185 },
    { hour: '2026-04-11 08:00', requests: 7258, instances: 12, cost_saved: 264.80, tokens_saved: 158618368 },
    { hour: '2026-04-11 09:00', requests: 8732, instances: 7, cost_saved: 301.46, tokens_saved: 157720064 },
    { hour: '2026-04-11 10:00', requests: 6058, instances: 6, cost_saved: 230.22, tokens_saved: 155936384 },
    { hour: '2026-04-11 11:00', requests: 7615, instances: 13, cost_saved: 444.21, tokens_saved: 288328690 },
    { hour: '2026-04-11 12:00', requests: 6749, instances: 10, cost_saved: 716.35, tokens_saved: 542054609 },
    { hour: '2026-04-11 13:00', requests: 6276, instances: 10, cost_saved: 2259.74, tokens_saved: 572384133 },
    { hour: '2026-04-11 14:00', requests: 8858, instances: 16, cost_saved: 846.04, tokens_saved: 602153567 },
    { hour: '2026-04-11 15:00', requests: 9284, instances: 12, cost_saved: 811.16, tokens_saved: 581585147 },
    { hour: '2026-04-11 16:00', requests: 7390, instances: 16, cost_saved: 749.69, tokens_saved: 566925112 },
    { hour: '2026-04-11 17:00', requests: 11558, instances: 13, cost_saved: 1023.99, tokens_saved: 623709492 },
    { hour: '2026-04-11 18:00', requests: 6154, instances: 10, cost_saved: 709.22, tokens_saved: 546162461 },
    { hour: '2026-04-11 19:00', requests: 5711, instances: 11, cost_saved: 660.33, tokens_saved: 548094492 },
    { hour: '2026-04-11 20:00', requests: 9287, instances: 13, cost_saved: 800.20, tokens_saved: 575078608 },
    { hour: '2026-04-11 21:00', requests: 5858, instances: 9, cost_saved: 675.77, tokens_saved: 547657183 },
    { hour: '2026-04-11 22:00', requests: 5523, instances: 13, cost_saved: 1133.82, tokens_saved: 638333840 },
    { hour: '2026-04-11 23:00', requests: 7668, instances: 8, cost_saved: 903.46, tokens_saved: 592286174 },
    { hour: '2026-04-12 00:00', requests: 5723, instances: 9, cost_saved: 655.44, tokens_saved: 543699655 },
    { hour: '2026-04-12 01:00', requests: 5291, instances: 8, cost_saved: 673.29, tokens_saved: 552129148 },
    { hour: '2026-04-12 02:00', requests: 6689, instances: 9, cost_saved: 698.83, tokens_saved: 551696076 },
    { hour: '2026-04-12 03:00', requests: 6532, instances: 8, cost_saved: 749.06, tokens_saved: 569436595 },
    { hour: '2026-04-12 04:00', requests: 4805, instances: 3, cost_saved: 644.29, tokens_saved: 540846848 },
    { hour: '2026-04-12 05:00', requests: 5181, instances: 6, cost_saved: 642.90, tokens_saved: 539173642 },
    { hour: '2026-04-12 06:00', requests: 4739, instances: 1, cost_saved: 637.17, tokens_saved: 537999942 },
  ],
  top_instances: [
    { os: 'Windows', version: '0.5.18', cost_saved: 36325.40, instance_id: '1d5d8ed0', tokens_saved: 8852060567 },
    { os: 'Linux', version: '0.5.19', cost_saved: 2188.80, instance_id: '96d9632f', tokens_saved: 2395311403 },
    { os: 'Windows', version: '0.5.17', cost_saved: 11878.30, instance_id: '1e850cb3', tokens_saved: 2375786993 },
    { os: 'Windows', version: '0.5.17', cost_saved: 9080.36, instance_id: '0cdfb8e9', tokens_saved: 1816351278 },
    { os: 'Linux', version: '0.5.18', cost_saved: 7580.30, instance_id: '7456b0f9', tokens_saved: 1635744469 },
    { os: 'Darwin', version: '0.5.16', cost_saved: 1772.99, instance_id: '1661b732', tokens_saved: 1488844495 },
    { os: 'Darwin', version: '0.5.17', cost_saved: 4667.23, instance_id: '08bf5ae1', tokens_saved: 933450700 },
    { os: 'Darwin', version: '0.5.18', cost_saved: 0, instance_id: 'e20f01b6', tokens_saved: 565038755 },
    { os: 'Linux', version: '0.5.18', cost_saved: 538.37, instance_id: '5b0795b2', tokens_saved: 557489086 },
    { os: 'Windows', version: '0.5.18', cost_saved: 1773.69, instance_id: 'eff9e644', tokens_saved: 503362483 },
    { os: 'Windows', version: '0.5.18', cost_saved: 1812.66, instance_id: '388102c7', tokens_saved: 485305324 },
    { os: 'Linux', version: '0.5.17', cost_saved: 2297.45, instance_id: '3ee70444', tokens_saved: 468622655 },
    { os: 'Darwin', version: '0.5.16', cost_saved: 319.65, instance_id: '6e758d1e', tokens_saved: 437502391 },
    { os: 'Linux', version: '0.5.18', cost_saved: 1461.02, instance_id: 'b2e71a28', tokens_saved: 415224504 },
    { os: 'Darwin', version: '0.5.21', cost_saved: 509.44, instance_id: '8b3795aa', tokens_saved: 353583838 },
    { os: 'Linux', version: '0.5.17', cost_saved: 1663.48, instance_id: 'b6a1a735', tokens_saved: 334438796 },
    { os: 'Linux', version: '0.5.21', cost_saved: 1603.77, instance_id: 'd0c16fa0', tokens_saved: 322890921 },
    { os: 'Linux', version: '0.5.18', cost_saved: 1580.12, instance_id: '4140eb00', tokens_saved: 316804284 },
    { os: 'Darwin', version: '0.5.21', cost_saved: 1052.77, instance_id: '2b11b55c', tokens_saved: 308601543 },
    { os: 'Darwin', version: '0.5.17', cost_saved: 1473.78, instance_id: '1ede777b', tokens_saved: 296950447 },
  ],
}

// --- Formatters ---

function fmt(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`
  return n.toFixed(0)
}

function fmtUsd(n: number): string {
  if (n >= 1e3) return `$${(n / 1e3).toFixed(1)}K`
  return `$${n.toFixed(0)}`
}

const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

function fmtDateDaily(d: string): string {
  const [, m, day] = d.split('-')
  return `${MONTHS[parseInt(m, 10) - 1]} ${parseInt(day, 10)}`
}

function fmtDateHourly(d: string): string {
  // "2026-04-10 06:00" → "Apr 10 6am"
  const [date, time] = d.split(' ')
  const [, m, day] = date.split('-')
  const hour = parseInt(time.split(':')[0], 10)
  const ampm = hour >= 12 ? 'pm' : 'am'
  const h12 = hour === 0 ? 12 : hour > 12 ? hour - 12 : hour
  return `${MONTHS[parseInt(m, 10) - 1]} ${parseInt(day, 10)} ${h12}${ampm}`
}

// --- Components ---

const PURPLE = 'hsl(262, 52%, 56%)'
const PURPLE_LIGHT = 'hsl(262, 60%, 65%)'

type Metric = 'cost_saved' | 'requests' | 'tokens_saved'
type TimeRange = 'daily' | 'hourly'

function ToggleButton({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      className={`px-3 py-1 text-xs font-medium rounded-md transition ${
        active
          ? 'bg-fd-primary text-fd-primary-foreground'
          : 'bg-fd-muted text-fd-muted-foreground hover:text-fd-foreground'
      }`}
    >
      {children}
    </button>
  )
}

function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null
  return (
    <div className="rounded-lg border border-fd-border bg-fd-card px-3 py-2 text-xs shadow-lg">
      <p className="font-medium text-fd-foreground mb-1">{label}</p>
      {payload.map((p: any) => (
        <p key={p.dataKey} className="text-fd-muted-foreground">
          {p.dataKey === 'cost_saved' ? fmtUsd(p.value) : fmt(p.value)}
        </p>
      ))}
    </div>
  )
}

function StatsCards() {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 not-prose">
      {[
        { label: 'Cost Saved', value: fmtUsd(DATA.total_cost_saved) },
        { label: 'Requests Optimized', value: fmt(DATA.total_requests) },
        { label: 'Active Instances', value: fmt(DATA.unique_instances) },
        { label: 'Active Days', value: String(DATA.active_days) },
      ].map(s => (
        <div key={s.label} className="flex flex-col items-center p-5 rounded-xl border border-fd-border bg-fd-card">
          <span className="text-2xl font-bold text-fd-foreground">{s.value}</span>
          <span className="mt-1 text-sm text-fd-muted-foreground">{s.label}</span>
        </div>
      ))}
    </div>
  )
}

function AreaChartSection() {
  const [metric, setMetric] = useState<Metric>('tokens_saved')
  const [timeRange, setTimeRange] = useState<TimeRange>('hourly')

  const isHourly = timeRange === 'hourly'
  const chartData: { label: string; requests: number; instances: number; cost_saved: number; tokens_saved: number }[] = !isHourly
    ? DATA.daily_stats.map(d => ({ label: fmtDateDaily(d.date), requests: d.requests, instances: d.instances, cost_saved: d.cost_saved, tokens_saved: d.tokens_saved }))
    : DATA.hourly_stats.map(d => ({ label: fmtDateHourly(d.hour), requests: d.requests, instances: d.instances, cost_saved: d.cost_saved, tokens_saved: d.tokens_saved }))

  const metricLabels: Record<Metric, string> = {
    cost_saved: 'Cost Saved',
    requests: 'Requests',
    tokens_saved: 'Tokens Saved',
  }

  const tickFormatter = (v: number) =>
    metric === 'cost_saved' ? fmtUsd(v) : fmt(v)

  return (
    <div className="rounded-xl border border-fd-border bg-fd-card p-5 not-prose">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-5">
        <div className="flex gap-1">
          {(['cost_saved', 'requests', 'tokens_saved'] as Metric[]).map(m => (
            <ToggleButton key={m} active={metric === m} onClick={() => setMetric(m)}>
              {metricLabels[m]}
            </ToggleButton>
          ))}
        </div>
        <div className="flex gap-1">
          {(['hourly', 'daily'] as TimeRange[]).map(t => (
            <ToggleButton key={t} active={timeRange === t} onClick={() => setTimeRange(t)}>
              {t === 'hourly' ? 'Last 48h' : 'Daily'}
            </ToggleButton>
          ))}
        </div>
      </div>
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={0}>
          <AreaChart data={chartData} margin={{ left: 0, right: 10, top: 5, bottom: isHourly ? 40 : 5 }}>
            <defs>
              <linearGradient id="purpleGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={PURPLE} stopOpacity={0.3} />
                <stop offset="100%" stopColor={PURPLE} stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-fd-border)" />
            <XAxis
              dataKey="label"
              tick={{ fontSize: 9, fill: 'var(--color-fd-muted-foreground)' }}
              interval={isHourly ? 3 : 0}
              angle={isHourly ? -45 : 0}
              textAnchor={isHourly ? 'end' : 'middle'}
              height={isHourly ? 60 : 30}
            />
            <YAxis tickFormatter={tickFormatter} tick={{ fontSize: 10, fill: 'var(--color-fd-muted-foreground)' }} width={45} />
            <Tooltip content={<CustomTooltip />} />
            <Area
              type="monotone"
              dataKey={metric}
              stroke={PURPLE}
              strokeWidth={2}
              fill="url(#purpleGrad)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

function BarChartSection() {
  const barData = DATA.top_instances.slice(0, 10).map(i => ({
    name: `${i.instance_id} (${i.os})`,
    tokens_saved: i.tokens_saved,
    cost_saved: i.cost_saved,
  }))

  return (
    <div className="rounded-xl border border-fd-border bg-fd-card p-5 not-prose">
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={0}>
          <BarChart data={barData} layout="vertical" margin={{ left: 0, right: 10, top: 5, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-fd-border)" horizontal={false} />
            <XAxis type="number" tickFormatter={fmt} tick={{ fontSize: 10, fill: 'var(--color-fd-muted-foreground)' }} />
            <YAxis type="category" dataKey="name" tick={{ fontSize: 9, fill: 'var(--color-fd-muted-foreground)' }} width={85} />
            <Tooltip content={({ active, payload }: any) => {
              if (!active || !payload?.length) return null
              const d = payload[0].payload
              return (
                <div className="rounded-lg border border-fd-border bg-fd-card px-3 py-2 text-xs shadow-lg">
                  <p className="font-medium text-fd-foreground">{d.name}</p>
                  <p className="text-fd-muted-foreground">{fmt(d.tokens_saved)} tokens</p>
                  <p className="text-fd-muted-foreground">{fmtUsd(d.cost_saved)} saved</p>
                </div>
              )
            }} />
            <Bar dataKey="tokens_saved" fill={PURPLE} radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

function DataTable() {
  return (
    <div className="rounded-xl border border-fd-border bg-fd-card overflow-hidden not-prose">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-fd-border text-left text-fd-muted-foreground">
              <th className="px-5 py-2 font-medium">Instance</th>
              <th className="px-5 py-2 font-medium">OS</th>
              <th className="px-5 py-2 font-medium">Version</th>
              <th className="px-5 py-2 font-medium text-right">Tokens Saved</th>
              <th className="px-5 py-2 font-medium text-right">Cost Saved</th>
            </tr>
          </thead>
          <tbody>
            {DATA.top_instances.map((inst, i) => (
              <tr
                key={inst.instance_id}
                className={`border-b border-fd-border last:border-0 ${i % 2 === 0 ? 'bg-fd-muted/30' : ''}`}
              >
                <td className="px-5 py-2.5 font-mono text-xs text-fd-foreground">{inst.instance_id}</td>
                <td className="px-5 py-2.5 text-fd-muted-foreground">{inst.os}</td>
                <td className="px-5 py-2.5 text-fd-muted-foreground">{inst.version}</td>
                <td className="px-5 py-2.5 text-right text-fd-foreground font-medium">{fmt(inst.tokens_saved)}</td>
                <td className="px-5 py-2.5 text-right text-fd-foreground font-medium">{fmtUsd(inst.cost_saved)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export function CommunityCharts({ section }: { section?: 'stats' | 'area' | 'bar' | 'table' }) {
  if (section === 'stats') return <StatsCards />
  if (section === 'area') return <AreaChartSection />
  if (section === 'bar') return <BarChartSection />
  if (section === 'table') return <DataTable />

  // Render all if no section specified
  return (
    <div className="space-y-10">
      <StatsCards />
      <AreaChartSection />
      <BarChartSection />
      <DataTable />
    </div>
  )
}
