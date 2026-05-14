-- Per-player handedness so the stroke classifier can mirror lefty pose
-- sequences at inference time. The TCN was trained on right-handed canonical
-- poses (HANDOFF_2026-05-13 — horizontal-flip disabled because a flipped FH
-- visually IS a BH). For a left-handed player we mirror x-axis + swap
-- left/right keypoint pairs in extract_pose_sequence so the classifier sees
-- the canonical right-handed orientation it knows.
--
-- Default 'right' — roughly 90% of players. Coach edits via the player
-- profile page (segmented control). Storage is a TEXT enum, not a boolean,
-- so we can add 'ambidextrous' later without another migration.

ALTER TABLE public.players
    ADD COLUMN IF NOT EXISTS handedness TEXT
    DEFAULT 'right'
    CHECK (handedness IN ('right', 'left'));

-- Backfill any existing rows with the default.
UPDATE public.players SET handedness = 'right' WHERE handedness IS NULL;
