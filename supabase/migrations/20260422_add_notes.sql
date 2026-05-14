-- Timestamped coach notes on matches: [{timestamp_sec, text}]
ALTER TABLE public.matches ADD COLUMN IF NOT EXISTS notes JSONB DEFAULT '[]'::jsonb;
