import Link from 'next/link';
import { fetchCommunityStats, fmtNum, fmtUsd } from '@/lib/telemetry';

/**
 * Server component that fetches live stats from Supabase at render time.
 * Falls back to hardcoded data if the API is unreachable.
 */
export async function LiveStats() {
  const data = await fetchCommunityStats();

  const stats = [
    { value: fmtNum(data.total_tokens_saved), label: 'Tokens Saved' },
    { value: fmtUsd(data.total_cost_saved), label: 'Cost Saved' },
    { value: fmtNum(data.total_requests), label: 'Requests Optimized' },
    { value: fmtNum(data.unique_instances), label: 'Active Instances' },
  ];

  return (
    <div className="not-prose">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 my-8">
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
      <Link
        href="/docs/community-savings"
        className="text-sm font-medium hover:underline"
      >
        View detailed charts and breakdowns &rarr;
      </Link>
    </div>
  );
}
