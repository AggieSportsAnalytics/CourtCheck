-- players: UC Davis Women's Tennis roster
CREATE TABLE IF NOT EXISTS public.players (
  id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name       TEXT NOT NULL,
  position   TEXT,
  year       TEXT,
  photo_url  TEXT,
  team_id    UUID,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Index for name lookups (deduplication)
CREATE INDEX IF NOT EXISTS players_name_idx ON public.players (name);

-- RLS: enabled, service role bypasses anyway
ALTER TABLE public.players ENABLE ROW LEVEL SECURITY;

CREATE POLICY "authenticated_read_players"
  ON public.players
  FOR SELECT
  USING (auth.role() = 'authenticated');

CREATE POLICY "service_full_access_players"
  ON public.players
  FOR ALL
  USING (true)
  WITH CHECK (true);

-- Add player_id foreign key to matches table
ALTER TABLE public.matches ADD COLUMN IF NOT EXISTS player_id UUID REFERENCES public.players(id);

-- Index for efficient player -> matches lookups
CREATE INDEX IF NOT EXISTS matches_player_id_idx ON public.matches (player_id);
