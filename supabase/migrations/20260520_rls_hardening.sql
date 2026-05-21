-- RLS hardening pass (2026-05-20).
--
-- Findings from Supabase advisors + manual policy audit:
--   1. public.swing_labels has a "full_access" policy with USING (true) for ALL
--      operations — any unauthenticated REST client could read/write/delete the
--      3479 labeled training rows. The annotation app uses the service_role key,
--      which bypasses RLS, so removing the public policy doesn't break it.
--   2. public.players has a "service_full_access_players" policy with USING (true)
--      for ALL operations, granted to the public role. Same issue — the API uses
--      supabaseAdmin (service_role) for writes, so revoking the public policy
--      doesn't break the app.
--   3. public.rls_auto_enable() is SECURITY DEFINER and executable by anon +
--      authenticated via PostgREST. It's an event trigger function — Postgres
--      calls it internally on DDL, never via REST. Revoking EXECUTE is safe.
--   4. Storage buckets raw-videos and results are public. All app access goes
--      through createSignedUrl() (1-hour expiry), so flipping them private has
--      no functional impact and closes URL-guessing as an attack vector.
--      assets stays public — brand logos served via plain <img src>.

BEGIN;

-- 1. Drop the always-true ALL policy on swing_labels.
DROP POLICY IF EXISTS "full_access" ON public.swing_labels;

-- 2. Drop the always-true ALL policy on players, keep the read policy.
DROP POLICY IF EXISTS "service_full_access_players" ON public.players;

-- 3. Lock down rls_auto_enable from REST exposure.
REVOKE EXECUTE ON FUNCTION public.rls_auto_enable() FROM anon, authenticated, PUBLIC;

-- 4. Flip user-data buckets to private. Signed URLs continue to work.
UPDATE storage.buckets
SET public = false
WHERE id IN ('raw-videos', 'results');

COMMIT;
