-- Keypoints: coach-annotated timestamps for set starts, side switches, and cuts
ALTER TABLE public.matches ADD COLUMN IF NOT EXISTS keypoints JSONB DEFAULT '[]'::jsonb;
