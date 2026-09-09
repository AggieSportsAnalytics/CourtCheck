-- Per-recording "favorite" flag so coaches can star the sessions they care
-- about and filter the recordings library down to them. Plain boolean: a
-- recording is either favorited or not, toggled from the recordings list row
-- and the recording detail header.
--
-- Default false; backfill leaves existing rows unfavorited.

ALTER TABLE public.matches
    ADD COLUMN IF NOT EXISTS favorited BOOLEAN NOT NULL DEFAULT false;

-- Partial index: filtering "favorites only" touches just the starred rows,
-- which are the minority, so an index on the true subset stays small and fast.
CREATE INDEX IF NOT EXISTS matches_favorited_idx
    ON public.matches (user_id)
    WHERE favorited;
