-- Per-annotator assignment for swing_labels.
-- Each annotator labels only the clips assigned to them.
-- '' (empty) = unassigned; queryable in the labeling UI as the "any" filter.

alter table public.swing_labels
    add column if not exists assigned_to text not null default '';

create index if not exists swing_labels_assigned_to_idx
    on public.swing_labels (assigned_to);
