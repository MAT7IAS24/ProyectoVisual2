-- Modelo campeón persistente y eventos de drift.
CREATE TABLE IF NOT EXISTS public.model_registry (
    registry_id BIGSERIAL PRIMARY KEY,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    is_champion BOOLEAN NOT NULL DEFAULT FALSE,
    activated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_retrained_at TIMESTAMPTZ,
    last_accuracy NUMERIC(12, 6),
    git_commit TEXT,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (model_name, model_version)
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_model_registry_one_champion
    ON public.model_registry (is_champion)
    WHERE is_champion = TRUE;

CREATE TABLE IF NOT EXISTS public.drift_events (
    drift_event_id BIGSERIAL PRIMARY KEY,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    baseline_accuracy NUMERIC(12, 6),
    recent_accuracy NUMERIC(12, 6),
    accuracy_drop NUMERIC(12, 6),
    threshold NUMERIC(12, 6) NOT NULL,
    drift_detected BOOLEAN NOT NULL,
    action_taken TEXT NOT NULL DEFAULT 'none',
    cycle_id TEXT,
    git_commit TEXT,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_drift_events_model_time
    ON public.drift_events (model_name, detected_at DESC);
