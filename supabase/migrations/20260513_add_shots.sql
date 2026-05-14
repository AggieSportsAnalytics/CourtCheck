-- Per-shot court-coordinate data for interactive courtmaps (shot map / spacing / coverage).
-- Stored as JSONB on the matches table so the pipeline can emit and the
-- frontend can consume in a single round-trip with the rest of recording data.
--
-- Each element shape (court units 0-27 width x 0-78 length, net at y=39):
--   {
--     "frame": int,            // frame index in source video
--     "time_s": float,         // seconds from start
--     "stroke": "forehand"|"backhand"|"serve"|"volley"|"unknown",
--     "player": 1 | 2,         // 1=near, 2=far
--     "court_x": float,        // bounce x in 0..27 court units
--     "court_y": float,        // bounce y in 0..78 court units
--     "in": bool,              // bounce inside singles boundaries
--     "ball_court_x": float|null,    // ball x at contact
--     "ball_court_y": float|null,
--     "player_court_x": float|null,  // hitter x at contact
--     "player_court_y": float|null
--   }
ALTER TABLE public.matches
    ADD COLUMN IF NOT EXISTS shots JSONB DEFAULT '[]'::jsonb;

-- GIN index for any future filter-by-stroke queries
CREATE INDEX IF NOT EXISTS matches_shots_gin_idx
    ON public.matches USING GIN (shots);
