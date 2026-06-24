-- Simplify cache bust tracking in proxy_telemetry_v2
-- Run in Supabase SQL Editor

-- 1. Drop the over-engineered columns from the previous version
ALTER TABLE proxy_telemetry_v2
  DROP COLUMN IF EXISTS cache_bust_count,
  DROP COLUMN IF EXISTS cache_bust_net_negative,
  DROP COLUMN IF EXISTS cache_bust_verdict;

-- 2. Keep just one column: tokens that lost their cache discount due to compression.
-- Compare with tokens_saved to see if compression is net-positive:
--   tokens_saved > cache_bust_tokens → compression wins
--   tokens_saved < cache_bust_tokens → should freeze more of the prefix
ALTER TABLE proxy_telemetry_v2
  ADD COLUMN IF NOT EXISTS cache_bust_tokens bigint DEFAULT 0;

-- 3. Drop the over-engineered dashboard column too
ALTER TABLE dashboard_summary
  DROP COLUMN IF EXISTS cache_bust_stats;
