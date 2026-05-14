-- Coach Insights summaries written by the pipeline. Each is a self-contained
-- JSONB object so the recording detail page can read with one round-trip and
-- doesn't recompute per request. Shapes are defined in coach-insights-spec.md
-- and produced by build_position_summary / build_net_approach_summary /
-- build_error_summary in backend/pipeline/run.py.
ALTER TABLE public.matches
    ADD COLUMN IF NOT EXISTS position_summary JSONB DEFAULT '{}'::jsonb;

ALTER TABLE public.matches
    ADD COLUMN IF NOT EXISTS net_approach_summary JSONB DEFAULT '{}'::jsonb;

ALTER TABLE public.matches
    ADD COLUMN IF NOT EXISTS error_summary JSONB DEFAULT '{}'::jsonb;
