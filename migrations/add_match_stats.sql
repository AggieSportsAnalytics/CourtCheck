-- Migration: Add tennis player stats columns to matches table
-- Run this in the Supabase SQL editor (https://supabase.com/dashboard → SQL Editor)

ALTER TABLE matches ADD COLUMN IF NOT EXISTS bounce_count     integer;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS shot_count       integer;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS rally_count      integer;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS forehand_count   integer;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS backhand_count   integer;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS serve_count      integer;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS in_bounds_bounces  integer;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS out_bounds_bounces integer;
