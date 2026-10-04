ALTER TABLE public.model_registry
    ADD COLUMN IF NOT EXISTS baseline_accuracy NUMERIC(12, 6),
    ADD COLUMN IF NOT EXISTS baseline_set_at TIMESTAMPTZ;

UPDATE public.model_registry
SET baseline_accuracy = COALESCE(baseline_accuracy, last_accuracy),
    baseline_set_at = COALESCE(baseline_set_at, last_retrained_at, NOW())
WHERE is_champion = TRUE;
