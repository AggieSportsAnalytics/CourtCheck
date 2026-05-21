-- Onboarding flag backfill (2026-05-21).
--
-- New users land on /onboarding to pick a starter roster (clone UC Davis or
-- start empty). Existing users with matches are auto-onboarded so they don't
-- get bounced through the flow on next login.

BEGIN;

-- Mark every existing user who has touched the product as onboarded.
-- Uses raw_user_meta_data (user-editable, exposed via supabase.auth.getUser().user_metadata).
UPDATE auth.users
SET raw_user_meta_data =
  COALESCE(raw_user_meta_data, '{}'::jsonb) || jsonb_build_object('onboarded', true)
WHERE id IN (SELECT DISTINCT user_id FROM public.matches WHERE user_id IS NOT NULL)
   OR id IN (SELECT DISTINCT user_id FROM public.players WHERE user_id IS NOT NULL);

COMMIT;
