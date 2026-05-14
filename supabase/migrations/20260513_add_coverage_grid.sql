-- 12-row x 8-col player coverage occupancy grid for the courtmap Coverage viz.
-- Bottom half-court (player's side), normalized to peak cell = 1.0.
-- Shape: number[][] (12 rows, 8 cols each), row 0 = nearest to net, row 11 = baseline.
-- Computed from per-frame player positions in the pipeline; only the aggregated
-- 96 floats are persisted so coverage stays interactive without storing the full track.
ALTER TABLE public.matches
    ADD COLUMN IF NOT EXISTS coverage_grid JSONB DEFAULT '[]'::jsonb;
