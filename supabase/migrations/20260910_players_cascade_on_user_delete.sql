-- Brian: apply this in the Supabase SQL editor; no Supabase CLI on this machine.
-- Delete owned players with their auth account instead of turning them into templates.
BEGIN;

ALTER TABLE public.players DROP CONSTRAINT IF EXISTS players_user_id_fkey;
ALTER TABLE public.players ADD CONSTRAINT players_user_id_fkey
  FOREIGN KEY (user_id) REFERENCES auth.users(id) ON DELETE CASCADE;

COMMIT;
