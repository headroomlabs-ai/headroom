-- Upgrade dashboard_summary: add hourly_stats column + update refresh function
-- Run this in Supabase SQL Editor (safe to run on existing table)

-- 1. Add hourly_stats column if missing
ALTER TABLE dashboard_summary
  ADD COLUMN IF NOT EXISTS hourly_stats jsonb DEFAULT '[]'::jsonb;

-- 2. Replace the refresh function with hourly support
CREATE OR REPLACE FUNCTION refresh_dashboard_summary()
RETURNS void AS $$
DECLARE
  _daily jsonb;
  _hourly jsonb;
  _top jsonb;
  _os jsonb;
  _versions jsonb;
  _total_tokens bigint;
  _total_cost numeric;
  _total_requests int;
  _unique_instances int;
  _active_days int;
BEGIN
  -- Daily totals: MAX per instance per day (beacon is cumulative), then SUM across instances
  WITH instance_daily AS (
    SELECT
      instance_id,
      created_at::date AS day,
      MAX(COALESCE(tokens_saved, 0)) AS tokens_saved,
      MAX(COALESCE(cost_saved_usd, 0)) AS cost_saved,
      MAX(COALESCE(requests, 0)) AS requests
    FROM proxy_telemetry_v2
    GROUP BY instance_id, created_at::date
  ),
  daily_agg AS (
    SELECT
      day,
      SUM(tokens_saved) AS tokens_saved,
      SUM(cost_saved)::numeric(12,2) AS cost_saved,
      SUM(requests) AS requests,
      COUNT(DISTINCT instance_id) AS instances
    FROM instance_daily
    GROUP BY day
    ORDER BY day
  )
  SELECT
    COALESCE(jsonb_agg(jsonb_build_object(
      'date', day,
      'tokens_saved', tokens_saved,
      'cost_saved', cost_saved,
      'requests', requests,
      'instances', instances
    ) ORDER BY day), '[]'::jsonb),
    COALESCE(SUM(tokens_saved), 0),
    COALESCE(SUM(cost_saved), 0),
    COALESCE(SUM(requests), 0),
    COUNT(DISTINCT day)
  INTO _daily, _total_tokens, _total_cost, _total_requests, _active_days
  FROM daily_agg;

  -- Hourly totals: last 48 hours, MAX per instance per hour, then SUM across instances
  WITH instance_hourly AS (
    SELECT
      instance_id,
      date_trunc('hour', created_at) AS hour,
      MAX(COALESCE(tokens_saved, 0)) AS tokens_saved,
      MAX(COALESCE(cost_saved_usd, 0)) AS cost_saved,
      MAX(COALESCE(requests, 0)) AS requests
    FROM proxy_telemetry_v2
    WHERE created_at >= now() - interval '48 hours'
    GROUP BY instance_id, date_trunc('hour', created_at)
  ),
  hourly_agg AS (
    SELECT
      hour,
      SUM(tokens_saved) AS tokens_saved,
      SUM(cost_saved)::numeric(12,2) AS cost_saved,
      SUM(requests) AS requests,
      COUNT(DISTINCT instance_id) AS instances
    FROM instance_hourly
    GROUP BY hour
    ORDER BY hour
  )
  SELECT COALESCE(jsonb_agg(jsonb_build_object(
    'hour', to_char(hour, 'YYYY-MM-DD HH24:MI'),
    'tokens_saved', tokens_saved,
    'cost_saved', cost_saved,
    'requests', requests,
    'instances', instances
  ) ORDER BY hour), '[]'::jsonb)
  INTO _hourly
  FROM hourly_agg;

  -- Unique instances
  SELECT COUNT(DISTINCT instance_id) INTO _unique_instances FROM proxy_telemetry_v2;

  -- Top 20 instances by total tokens saved
  WITH instance_totals AS (
    SELECT
      instance_id,
      SUM(max_tokens) AS tokens_saved,
      SUM(max_cost)::numeric(12,2) AS cost_saved,
      MAX(os) AS os,
      MAX(version) AS version
    FROM (
      SELECT
        instance_id,
        created_at::date,
        MAX(COALESCE(tokens_saved, 0)) AS max_tokens,
        MAX(COALESCE(cost_saved_usd, 0)) AS max_cost,
        MAX(os) AS os,
        MAX(headroom_version) AS version
      FROM proxy_telemetry_v2
      GROUP BY instance_id, created_at::date
    ) sub
    GROUP BY instance_id
    ORDER BY tokens_saved DESC
    LIMIT 20
  )
  SELECT COALESCE(jsonb_agg(jsonb_build_object(
    'instance_id', LEFT(instance_id, 8),
    'tokens_saved', tokens_saved,
    'cost_saved', cost_saved,
    'os', SPLIT_PART(COALESCE(os, '?'), ' ', 1),
    'version', version
  ) ORDER BY tokens_saved DESC), '[]'::jsonb)
  INTO _top
  FROM instance_totals;

  -- OS breakdown
  SELECT COALESCE(jsonb_object_agg(os_name, cnt), '{}'::jsonb)
  INTO _os
  FROM (
    SELECT SPLIT_PART(COALESCE(os, '?'), ' ', 1) AS os_name, COUNT(*) AS cnt
    FROM proxy_telemetry_v2
    GROUP BY os_name
  ) sub;

  -- Version breakdown
  SELECT COALESCE(jsonb_object_agg(COALESCE(headroom_version, '?'), cnt), '{}'::jsonb)
  INTO _versions
  FROM (
    SELECT headroom_version, COUNT(*) AS cnt
    FROM proxy_telemetry_v2
    GROUP BY headroom_version
  ) sub;

  -- Upsert the single summary row
  INSERT INTO dashboard_summary (id, updated_at, total_tokens_saved, total_cost_saved,
    total_requests, unique_instances, active_days, daily_stats, hourly_stats,
    top_instances, os_breakdown, version_breakdown)
  VALUES ('current', now(), _total_tokens, _total_cost, _total_requests,
    _unique_instances, _active_days, _daily, _hourly, _top, _os, _versions)
  ON CONFLICT (id) DO UPDATE SET
    updated_at = EXCLUDED.updated_at,
    total_tokens_saved = EXCLUDED.total_tokens_saved,
    total_cost_saved = EXCLUDED.total_cost_saved,
    total_requests = EXCLUDED.total_requests,
    unique_instances = EXCLUDED.unique_instances,
    active_days = EXCLUDED.active_days,
    daily_stats = EXCLUDED.daily_stats,
    hourly_stats = EXCLUDED.hourly_stats,
    top_instances = EXCLUDED.top_instances,
    os_breakdown = EXCLUDED.os_breakdown,
    version_breakdown = EXCLUDED.version_breakdown;
END;
$$ LANGUAGE plpgsql;

-- 3. Refresh now to populate hourly data
SELECT refresh_dashboard_summary();
