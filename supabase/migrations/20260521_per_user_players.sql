-- Per-user player ownership (2026-05-21).
--
-- Model:
--   players.user_id IS NULL  → demo data (UC Davis roster), readable by all authed users
--   players.user_id = <uuid> → that user's private roster
--
-- Replaces the env-based admin allowlist (ADMIN_EMAILS) with row-level ownership.
-- Existing 9 UC Davis players keep user_id = NULL so every signed-up user
-- continues to see them on landing.

BEGIN;

-- 1. Add the owner column. ON DELETE SET NULL converts a deleted user's
--    roster into orphaned demo data rather than cascading deletes (the
--    matches data tied to those players would still need cleanup, but
--    that's a separate concern).
ALTER TABLE public.players
  ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS players_user_id_idx ON public.players(user_id);

-- 2. Drop the broad authenticated_read_players policy and replace with the
--    full per-user policy set.
DROP POLICY IF EXISTS "authenticated_read_players" ON public.players;

-- SELECT: anyone authed can see demo (user_id IS NULL) OR their own.
CREATE POLICY "players_read_demo_or_own"
  ON public.players
  FOR SELECT
  TO authenticated
  USING (user_id IS NULL OR user_id = (select auth.uid()));

-- INSERT: must set user_id to themselves (prevents claiming a row as someone else).
CREATE POLICY "players_insert_own"
  ON public.players
  FOR INSERT
  TO authenticated
  WITH CHECK (user_id = (select auth.uid()));

-- UPDATE: only their own rows; cannot mutate demo data.
CREATE POLICY "players_update_own"
  ON public.players
  FOR UPDATE
  TO authenticated
  USING (user_id = (select auth.uid()))
  WITH CHECK (user_id = (select auth.uid()));

-- DELETE: only their own rows.
CREATE POLICY "players_delete_own"
  ON public.players
  FOR DELETE
  TO authenticated
  USING (user_id = (select auth.uid()));

COMMIT;
