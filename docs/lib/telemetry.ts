export interface CommunityStats {
  total_tokens_saved: number;
  total_cost_saved: number;
  total_requests: number;
  unique_instances: number;
}

const fallbackStats: CommunityStats = {
  total_tokens_saved: 0,
  total_cost_saved: 0,
  total_requests: 0,
  unique_instances: 0,
};

export function fmtNum(value: number) {
  return new Intl.NumberFormat('en-US', {
    notation: value >= 10000 ? 'compact' : 'standard',
    maximumFractionDigits: 1,
  }).format(value);
}

export function fmtUsd(value: number) {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    notation: value >= 10000 ? 'compact' : 'standard',
    maximumFractionDigits: value >= 1000 ? 1 : 2,
  }).format(value);
}

export async function fetchCommunityStats(): Promise<CommunityStats> {
  const endpoint = process.env.NEXT_PUBLIC_COMMUNITY_STATS_URL;
  if (!endpoint) return fallbackStats;

  try {
    const response = await fetch(endpoint, { next: { revalidate: 300 } });
    if (!response.ok) return fallbackStats;

    return {
      ...fallbackStats,
      ...(await response.json()),
    };
  } catch {
    return fallbackStats;
  }
}
