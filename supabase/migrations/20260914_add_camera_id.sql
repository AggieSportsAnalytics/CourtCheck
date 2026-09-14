-- Brian applies this migration in the Supabase SQL editor.
ALTER TABLE public.matches ADD COLUMN IF NOT EXISTS camera_id text;
ALTER TABLE public.matches ADD COLUMN IF NOT EXISTS camera_match jsonb;
