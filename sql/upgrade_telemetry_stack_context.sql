-- Add deployment-context columns to proxy_telemetry_v2
-- Run in Supabase SQL Editor.
--
-- headroom_stack:      how Headroom is being invoked — e.g. "proxy",
--                      "wrap_claude", "adapter_ts_openai", "mixed", "unknown".
-- install_mode:        how the proxy process is deployed — one of
--                      "wrapped", "persistent", "on_demand", "unknown".
-- requests_by_stack:   JSONB dict {stack_slug: count} for sessions that see
--                      multiple integration surfaces (e.g. a persistent proxy
--                      serving both wrap_claude and TS adapter callers).

ALTER TABLE proxy_telemetry_v2
  ADD COLUMN IF NOT EXISTS headroom_stack text,
  ADD COLUMN IF NOT EXISTS install_mode text,
  ADD COLUMN IF NOT EXISTS requests_by_stack jsonb;
