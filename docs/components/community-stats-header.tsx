import { fetchCommunityStats, fmtNum, fmtUsd } from '@/lib/telemetry';

/**
 * Server component that renders the top-level stats for the community savings page.
 * Fetches live data from Supabase at render time.
 */
export async function CommunityStatsHeader() {
  const data = await fetchCommunityStats();

  const stats = [
    { value: fmtNum(data.total_tokens_saved), label: 'Tokens Saved' },
    { value: fmtUsd(data.total_cost_saved), label: 'Cost Saved' },
    { value: fmtNum(data.total_requests), label: 'Requests Optimized' },
    { value: fmtNum(data.unique_instances), label: 'Active Instances' },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 not-prose">
      {stats.map((s) => (
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
  );
}
