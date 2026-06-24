-- Supabase SQL: Create proxy_telemetry_v2 table
-- Run this in the Supabase SQL Editor (https://supabase.com/dashboard → SQL Editor)
-- This table matches every field the beacon code writes in headroom/telemetry/beacon.py

CREATE TABLE IF NOT EXISTS proxy_telemetry_v2 (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    created_at timestamptz DEFAULT now() NOT NULL,

    -- Core identity (always present)
    session_id text NOT NULL,
    instance_id text,
    headroom_version text,
    python_version text,
    os text,
    sdk text,
    backend text,
    session_minutes integer,
    headroom_stack text,
    install_mode text,
    requests_by_stack jsonb,

    -- Effectiveness metrics
    tokens_saved bigint,
    requests integer,
    compression_percent real,
    cache_hit_rate real,
    cost_saved_usd real,
    cache_saved_usd real,
    models_used jsonb,

    -- Performance overhead
    overhead_avg_ms real,
    overhead_max_ms real,

    -- TTFB
    ttfb_avg_ms real,

    -- Pipeline timing (JSONB: {transform_name: avg_ms})
    pipeline_timing jsonb,

    -- Request patterns
    avg_tokens_before integer,
    avg_tokens_after integer,

    -- Compression cache
    compression_cache jsonb,

    -- CCR usage
    ccr jsonb,

    -- Waste signals
    waste_signals jsonb
);

-- Index for querying by session and time
CREATE INDEX IF NOT EXISTS idx_ptv2_session_id ON proxy_telemetry_v2(session_id);
CREATE INDEX IF NOT EXISTS idx_ptv2_created_at ON proxy_telemetry_v2(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ptv2_version ON proxy_telemetry_v2(headroom_version);

-- Enable Row Level Security
ALTER TABLE proxy_telemetry_v2 ENABLE ROW LEVEL SECURITY;

-- RLS policies: anon can INSERT (telemetry beacon) and SELECT (for debugging)
CREATE POLICY "anon_insert" ON proxy_telemetry_v2
    FOR INSERT TO anon
    WITH CHECK (true);

CREATE POLICY "anon_select" ON proxy_telemetry_v2
    FOR SELECT TO anon
    USING (true);

-- Grant permissions to anon role
GRANT INSERT, SELECT ON proxy_telemetry_v2 TO anon;
