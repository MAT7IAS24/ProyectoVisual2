-- Proyecto: Proyectovisual2
-- Modelo de datos para Pulso TransMi / MLOps de predicción de demanda

CREATE TABLE IF NOT EXISTS stations (
    station_id TEXT PRIMARY KEY,
    station_name TEXT NOT NULL,
    corridor TEXT,
    latitude NUMERIC(10, 8) NOT NULL,
    longitude NUMERIC(11, 8) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS demand_observations (
    observation_id BIGSERIAL PRIMARY KEY,
    station_id TEXT NOT NULL REFERENCES stations(station_id) ON DELETE RESTRICT,
    observed_at TIMESTAMPTZ NOT NULL,
    demand INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (station_id, observed_at)
);

CREATE TABLE IF NOT EXISTS context_observations (
    context_id BIGSERIAL PRIMARY KEY,
    observed_at TIMESTAMPTZ NOT NULL UNIQUE,
    rain_mm NUMERIC(10, 4) NOT NULL DEFAULT 0,
    rain_forecast NUMERIC(10, 4) NOT NULL DEFAULT 0,
    temperature_c NUMERIC(10, 4) NOT NULL DEFAULT 0,
    temperature_forecast NUMERIC(10, 4) NOT NULL DEFAULT 0,
    event_intensity NUMERIC(10, 4) NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS models (
    model_id BIGSERIAL PRIMARY KEY,
    model_name TEXT NOT NULL UNIQUE,
    algorithm TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS model_versions (
    model_version_id BIGSERIAL PRIMARY KEY,
    model_id BIGINT NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
    version_tag TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    artifact_path TEXT,
    config_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    status TEXT NOT NULL DEFAULT 'draft',
    UNIQUE (model_id, version_tag)
);

CREATE TABLE IF NOT EXISTS training_runs (
    training_run_id BIGSERIAL PRIMARY KEY,
    model_version_id BIGINT NOT NULL REFERENCES model_versions(model_version_id) ON DELETE CASCADE,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'running',
    train_start TIMESTAMPTZ,
    train_end TIMESTAMPTZ,
    validation_start TIMESTAMPTZ,
    validation_end TIMESTAMPTZ,
    metrics_json JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id BIGSERIAL PRIMARY KEY,
    model_version_id BIGINT NOT NULL REFERENCES model_versions(model_version_id) ON DELETE CASCADE,
    station_id TEXT NOT NULL REFERENCES stations(station_id) ON DELETE RESTRICT,
    prediction_time TIMESTAMPTZ NOT NULL,
    target_time TIMESTAMPTZ NOT NULL,
    horizon_minutes INTEGER NOT NULL,
    predicted_demand NUMERIC(12, 4) NOT NULL,
    actual_demand NUMERIC(12, 4),
    error_value NUMERIC(12, 4),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (model_version_id, station_id, target_time, horizon_minutes)
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    pipeline_run_id BIGSERIAL PRIMARY KEY,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'running',
    trigger_type TEXT NOT NULL DEFAULT 'manual',
    notes TEXT
);

CREATE TABLE IF NOT EXISTS submissions (
    submission_id BIGSERIAL PRIMARY KEY,
    pipeline_run_id BIGINT NOT NULL REFERENCES pipeline_runs(pipeline_run_id) ON DELETE CASCADE,
    model_version_id BIGINT NOT NULL REFERENCES model_versions(model_version_id) ON DELETE RESTRICT,
    submitted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    score_wape NUMERIC(12, 6),
    accuracy NUMERIC(12, 6),
    rank_position INTEGER,
    status TEXT NOT NULL DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS leaderboard (
    leaderboard_id BIGSERIAL PRIMARY KEY,
    submission_id BIGINT NOT NULL REFERENCES submissions(submission_id) ON DELETE CASCADE,
    metric_name TEXT NOT NULL,
    metric_value NUMERIC(12, 6) NOT NULL,
    rank_position INTEGER,
    snapshot_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (submission_id, metric_name)
);

CREATE INDEX IF NOT EXISTS idx_demand_observations_station_time
    ON demand_observations (station_id, observed_at);

CREATE INDEX IF NOT EXISTS idx_demand_observations_time
    ON demand_observations (observed_at);

CREATE INDEX IF NOT EXISTS idx_context_observations_time
    ON context_observations (observed_at);

CREATE INDEX IF NOT EXISTS idx_predictions_model_station_time
    ON predictions (model_version_id, station_id, target_time);

CREATE INDEX IF NOT EXISTS idx_submissions_model_run
    ON submissions (model_version_id, pipeline_run_id);

CREATE INDEX IF NOT EXISTS idx_leaderboard_metric_rank
    ON leaderboard (metric_name, rank_position);
