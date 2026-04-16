-- swing_labels: one row per candidate swing clip.
-- Seeded from queue.csv via seed_swing_labels.py.
-- label / annotator / labeled_at are updated in-place as annotators work.

create table if not exists public.swing_labels (
    clip_id        text primary key,
    supabase_path  text        not null default '',
    source_video   text        not null default '',
    peak_frame     integer     not null default 0,
    window_start   integer     not null default 0,
    window_end     integer     not null default 0,
    player_idx     integer     not null default 0,
    wrist_velocity double precision not null default 0,
    label          text        not null default '',
    annotator      text        not null default '',
    labeled_at     timestamptz
);

-- Index for fast "unlabeled" queries (label = '')
create index if not exists swing_labels_label_idx on public.swing_labels (label);

-- RLS: enabled but open (annotation app uses service role key which bypasses RLS anyway)
alter table public.swing_labels enable row level security;

create policy "full_access"
    on public.swing_labels
    for all
    using (true)
    with check (true);
