ALTER TABLE public.demand_observations
    ADD COLUMN IF NOT EXISTS released_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_demand_observations_released_at
    ON public.demand_observations (released_at DESC);
