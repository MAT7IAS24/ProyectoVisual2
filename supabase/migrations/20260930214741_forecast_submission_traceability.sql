CREATE TABLE IF NOT EXISTS public.forecast_submissions (
    submission_id BIGSERIAL PRIMARY KEY,
    cycle_id TEXT NOT NULL UNIQUE,
    client_run_id TEXT NOT NULL UNIQUE,
    model_version TEXT NOT NULL,
    submitted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    api_status INTEGER NOT NULL,
    api_response JSONB NOT NULL DEFAULT '{}'::jsonb,
    git_commit TEXT
);

CREATE TABLE IF NOT EXISTS public.forecast_predictions (
    prediction_id BIGSERIAL PRIMARY KEY,
    submission_id BIGINT NOT NULL REFERENCES public.forecast_submissions(submission_id) ON DELETE CASCADE,
    cycle_id TEXT NOT NULL,
    station_id TEXT NOT NULL REFERENCES public.stations(station_id) ON DELETE RESTRICT,
    target_at TIMESTAMPTZ NOT NULL,
    horizon_minutes INTEGER,
    predicted_demand NUMERIC(12, 4) NOT NULL,
    actual_demand NUMERIC(12, 4),
    error_value NUMERIC(12, 4),
    evaluated_at TIMESTAMPTZ,
    UNIQUE (cycle_id, station_id, target_at)
);

CREATE INDEX IF NOT EXISTS idx_forecast_predictions_horizon
    ON public.forecast_predictions (horizon_minutes, target_at);
