-- Rally state-machine output. `rallies` is the full per-rally array
-- (one object per rally with shots[]); `rally_summary` is the aggregate
-- (avg/median length, win counts, end-reason histogram) used by the
-- match-detail stats tiles and Coach Insights summary card.
-- Shapes are produced by build_rallies / build_rally_summary in
-- backend/pipeline/rallies.py.
ALTER TABLE public.matches
    ADD COLUMN IF NOT EXISTS rallies JSONB DEFAULT '[]'::jsonb;

ALTER TABLE public.matches
    ADD COLUMN IF NOT EXISTS rally_summary JSONB DEFAULT '{}'::jsonb;
