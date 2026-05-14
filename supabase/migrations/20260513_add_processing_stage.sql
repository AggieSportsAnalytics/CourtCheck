-- Human-readable processing stage label written by the pipeline alongside the
-- 0..1 progress float. Lets the upload + recording-detail pages render the
-- actual phase Modal is in (e.g. "Following the ball and players") instead of
-- deriving stage labels client-side from progress thresholds, which lags
-- whenever the percent write is delayed.
ALTER TABLE public.matches
    ADD COLUMN IF NOT EXISTS processing_stage TEXT;
