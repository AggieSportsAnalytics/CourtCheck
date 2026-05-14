-- player_shot_map_path: storage path for the player position dot map at shot moments
ALTER TABLE public.matches ADD COLUMN IF NOT EXISTS player_shot_map_path TEXT;
