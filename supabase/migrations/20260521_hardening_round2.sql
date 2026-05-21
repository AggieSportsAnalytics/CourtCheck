-- Hardening round 2 (2026-05-21) — repo getting attention, locking down for public traffic.
--
-- Changes:
--   1. rate_limit_events: per-user/per-IP sliding-window counter table for app-side rate limiting
--   2. audit_log: append-only log of sensitive operations + triggers on matches/players
--   3. Fix auth_rls_initplan perf advisor on players.authenticated_read_players
--   4. storage.buckets file_size_limit: cap raw-videos uploads at 2 GB (matches Modal 30-min ceiling)
--   5. storage.buckets allowed_mime_types: lock raw-videos to video/*

BEGIN;

-- =========================================================================
-- 1. Rate-limit events table
-- =========================================================================
CREATE TABLE IF NOT EXISTS public.rate_limit_events (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID,
  ip TEXT,
  bucket TEXT NOT NULL,
  at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Composite indexes for the COUNT queries the helper runs
CREATE INDEX IF NOT EXISTS rate_limit_events_user_bucket_at_idx
  ON public.rate_limit_events (user_id, bucket, at DESC)
  WHERE user_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS rate_limit_events_ip_bucket_at_idx
  ON public.rate_limit_events (ip, bucket, at DESC)
  WHERE ip IS NOT NULL;

-- BRIN index on `at` for the cleanup job — keeps storage low without a full B-tree
CREATE INDEX IF NOT EXISTS rate_limit_events_at_brin
  ON public.rate_limit_events USING BRIN (at);

ALTER TABLE public.rate_limit_events ENABLE ROW LEVEL SECURITY;
-- No policies — only service_role can write/read. The API uses supabaseAdmin
-- (service_role) which bypasses RLS. Anon REST cannot reach this table.

-- Cleanup function: drop events older than 24h. Called by a cron in app code
-- or you can wire pg_cron with a schedule.
CREATE OR REPLACE FUNCTION public.rate_limit_prune()
RETURNS void
LANGUAGE plpgsql
SECURITY INVOKER
AS $$
BEGIN
  DELETE FROM public.rate_limit_events WHERE at < NOW() - INTERVAL '24 hours';
END;
$$;
REVOKE EXECUTE ON FUNCTION public.rate_limit_prune() FROM PUBLIC, anon, authenticated;

-- =========================================================================
-- 2. Audit log + triggers
-- =========================================================================
CREATE TABLE IF NOT EXISTS public.audit_log (
  id BIGSERIAL PRIMARY KEY,
  at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  actor UUID,            -- auth.uid() when available
  schema_name TEXT NOT NULL,
  table_name TEXT NOT NULL,
  op TEXT NOT NULL,      -- INSERT | UPDATE | DELETE
  row_id TEXT,
  old_row JSONB,
  new_row JSONB
);

CREATE INDEX IF NOT EXISTS audit_log_at_idx ON public.audit_log (at DESC);
CREATE INDEX IF NOT EXISTS audit_log_actor_at_idx ON public.audit_log (actor, at DESC);
CREATE INDEX IF NOT EXISTS audit_log_table_at_idx ON public.audit_log (table_name, at DESC);

ALTER TABLE public.audit_log ENABLE ROW LEVEL SECURITY;
-- Only service_role can write/read via supabaseAdmin. No anon access.

CREATE OR REPLACE FUNCTION public.audit_trigger()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
DECLARE
  v_actor UUID;
  v_row_id TEXT;
BEGIN
  -- auth.uid() is null when triggered by service_role; that's fine, we log NULL.
  BEGIN
    v_actor := auth.uid();
  EXCEPTION WHEN OTHERS THEN
    v_actor := NULL;
  END;

  v_row_id := COALESCE(
    (CASE WHEN TG_OP = 'DELETE' THEN OLD.id::text ELSE NEW.id::text END),
    NULL
  );

  INSERT INTO public.audit_log (actor, schema_name, table_name, op, row_id, old_row, new_row)
  VALUES (
    v_actor,
    TG_TABLE_SCHEMA,
    TG_TABLE_NAME,
    TG_OP,
    v_row_id,
    CASE WHEN TG_OP IN ('UPDATE', 'DELETE') THEN to_jsonb(OLD) ELSE NULL END,
    CASE WHEN TG_OP IN ('UPDATE', 'INSERT') THEN to_jsonb(NEW) ELSE NULL END
  );

  RETURN COALESCE(NEW, OLD);
END;
$$;

REVOKE EXECUTE ON FUNCTION public.audit_trigger() FROM PUBLIC, anon, authenticated;

-- Attach triggers to sensitive tables.
DROP TRIGGER IF EXISTS audit_matches_changes ON public.matches;
CREATE TRIGGER audit_matches_changes
  AFTER INSERT OR UPDATE OR DELETE ON public.matches
  FOR EACH ROW EXECUTE FUNCTION public.audit_trigger();

DROP TRIGGER IF EXISTS audit_players_changes ON public.players;
CREATE TRIGGER audit_players_changes
  AFTER INSERT OR UPDATE OR DELETE ON public.players
  FOR EACH ROW EXECUTE FUNCTION public.audit_trigger();

-- =========================================================================
-- 3. Perf advisor fix: re-wrap auth.role() in a subselect so it's evaluated
--    once per query, not once per row.
-- =========================================================================
DROP POLICY IF EXISTS "authenticated_read_players" ON public.players;
CREATE POLICY "authenticated_read_players"
  ON public.players
  FOR SELECT
  TO authenticated
  USING ((select auth.role()) = 'authenticated');

-- =========================================================================
-- 4. Cap raw-videos uploads + restrict mime types
-- =========================================================================
UPDATE storage.buckets
SET file_size_limit = 2147483648,  -- 2 GiB
    allowed_mime_types = ARRAY['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/webm']
WHERE id = 'raw-videos';

UPDATE storage.buckets
SET file_size_limit = 524288000  -- 500 MiB for processed outputs
WHERE id = 'results';

UPDATE storage.buckets
SET file_size_limit = 104857600  -- 100 MiB
WHERE id = 'swing-clips';

COMMIT;
