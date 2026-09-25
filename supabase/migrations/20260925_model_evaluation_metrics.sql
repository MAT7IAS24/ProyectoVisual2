-- Métricas históricas de evaluación de modelos por estación.
-- Permite comparar ejecuciones y detectar degradación/drift posteriormente.

CREATE TABLE IF NOT EXISTS public.model_evaluation_metrics (
    metric_id BIGSERIAL PRIMARY KEY,
    evaluated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_version TEXT NOT NULL,
    model_name TEXT NOT NULL,
    station_id TEXT NOT NULL REFERENCES public.stations(station_id) ON DELETE RESTRICT,
    station_name TEXT,
    corridor TEXT,
    evaluation_window TEXT NOT NULL DEFAULT 'recent_temporal_validation',
    cycle_id TEXT,
    data_cutoff TIMESTAMPTZ,
    accuracy NUMERIC(12, 6) NOT NULL,
    wape NUMERIC(12, 6),
    train_rows INTEGER,
    evaluation_rows INTEGER,
    git_commit TEXT,
    mlflow_run_id TEXT,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_model_metrics_station_time
    ON public.model_evaluation_metrics (station_id, evaluated_at DESC);

CREATE INDEX IF NOT EXISTS idx_model_metrics_model_time
    ON public.model_evaluation_metrics (model_name, evaluated_at DESC);

CREATE INDEX IF NOT EXISTS idx_model_metrics_cycle
    ON public.model_evaluation_metrics (cycle_id, evaluated_at DESC);
