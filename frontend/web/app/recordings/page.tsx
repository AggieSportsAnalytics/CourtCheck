'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import EditableName from '@/components/recordings/EditableName';
import BounceLoader from '@/components/upload/BounceLoader';
import { useRecordingsData } from '@/lib/hooks/useApiData';

/**
 * Recordings list. Ported from docs/brand-drop/mocks/matches-list.html.
 *
 * Layout:
 *   - h1 "Every recording, scrubbable." (clay italic on "scrubbable.")
 *   - Mono meta line: results count ("12 recordings.")
 *   - Filter bar: search + date/filter chips
 *   - Match table: .cc-match-row grid (checkbox? | 110px date | 1fr title | actions)
 *   - Empty state CTA → /upload
 *   - Bulk action bar (fixed, slides up when items selected)
 *
 * Wires to /api/recordings. A recording is identified by its own title
 * (`name`), NOT associated to a player profile.
 */

interface Recording {
  id: string;
  status: 'pending' | 'processing' | 'done' | 'failed';
  createdAt: string;
  /** User-facing title (e.g. "test5-player", "StMarys Court2 4950 clip"). */
  name: string;
  filename: string;
}

const AVATAR_COLORS = [
  'var(--color-court)',
  'var(--color-clay)',
  'var(--color-plum)',
  'var(--color-amber)',
  'var(--color-slate)',
];

function hashColor(seed: string): string {
  let h = 0;
  for (let i = 0; i < seed.length; i++) h = (h * 31 + seed.charCodeAt(i)) | 0;
  const idx = Math.abs(h) % AVATAR_COLORS.length;
  return AVATAR_COLORS[idx];
}

/**
 * Tidy a raw recording title for display — strips the file extension and
 * collapses underscores. Used when the upload didn't supply an explicit title
 * so the filename is the title (e.g. "StMarys_Court2_4950_clip.mp4" →
 * "StMarys Court2 4950 clip").
 */
function cleanTitle(name: string): string {
  return name.replace(/\.[a-z0-9]+$/i, '').replace(/_/g, ' ').trim();
}

function initials(s: string): string {
  return s
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((p) => p[0]?.toUpperCase() ?? '')
    .join('');
}

function fmtDate(iso: string): string {
  return new Date(iso).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
  });
}

export default function RecordingsPage() {
  const router = useRouter();
  // SWR-cached recordings: instant render on revisit, background revalidation,
  // and auto-polling while anything is still processing (see useRecordingsData).
  const { data, error: swrError, isLoading, mutate } = useRecordingsData();
  const recordings: Recording[] = useMemo(
    () =>
      (data?.recordings ?? []).map((r) => ({
        id: r.id,
        status: r.status as Recording['status'],
        createdAt: r.createdAt,
        name: r.name,
        filename: r.filename,
      })),
    [data],
  );
  const loading = isLoading;
  const error = swrError ? (swrError as Error).message : null;
  const [query, setQuery] = useState('');
  const [thisSeasonActive, setThisSeasonActive] = useState(false);
  const [filterOpen, setFilterOpen] = useState(false);
  const [statusFilter, setStatusFilter] = useState<'all' | 'done' | 'processing' | 'failed'>('all');
  const filterRef = useRef<HTMLDivElement>(null);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  // Multi-select state
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [bulkState, setBulkState] = useState<'idle' | 'confirming' | 'deleting'>('idle');
  const [bulkError, setBulkError] = useState<string | null>(null);

  async function deleteRecording(id: string) {
    setDeletingId(id);
    setDeleteError(null);
    try {
      const res = await fetch(`/api/recordings/${id}`, { method: 'DELETE' });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || `Delete failed (${res.status})`);
      }
      setConfirmDelete(null);
      // Drop from the SWR cache immediately, then revalidate to confirm.
      mutate(
        (curr) =>
          curr ? { recordings: curr.recordings.filter((r) => r.id !== id) } : curr,
        { revalidate: true },
      );
    } catch (e) {
      setDeleteError((e as Error).message);
    } finally {
      setDeletingId(null);
    }
  }

  function toggleSelect(id: string) {
    setConfirmDelete(null);
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function selectAllFiltered() {
    setSelectedIds(new Set(filtered.map((r) => r.id)));
  }

  function clearSelection() {
    setSelectedIds(new Set());
    setBulkState('idle');
    setBulkError(null);
  }

  async function bulkDelete() {
    setBulkState('deleting');
    setBulkError(null);
    const ids = [...selectedIds];
    const results = await Promise.allSettled(
      ids.map((id) => fetch(`/api/recordings/${id}`, { method: 'DELETE' }))
    );
    const successIds = ids.filter(
      (_, i) =>
        results[i].status === 'fulfilled' &&
        (results[i] as PromiseFulfilledResult<Response>).value.ok
    );
    mutate(
      (curr) =>
        curr
          ? { recordings: curr.recordings.filter((r) => !successIds.includes(r.id)) }
          : curr,
      { revalidate: true },
    );
    setSelectedIds((prev) => {
      const next = new Set(prev);
      successIds.forEach((id) => next.delete(id));
      return next;
    });
    const failCount = ids.length - successIds.length;
    if (failCount > 0) {
      setBulkError(`${failCount} couldn't be deleted.`);
    }
    setBulkState('idle');
  }

  // Academic year starts Aug 1. Returns Aug 1 of the current or previous year.
  function currentSeasonStart(): Date {
    const now = new Date();
    const year = now.getMonth() >= 7 ? now.getFullYear() : now.getFullYear() - 1;
    return new Date(year, 7, 1);
  }

  useEffect(() => {
    if (!filterOpen) return;
    function handleClick(e: MouseEvent) {
      if (filterRef.current && !filterRef.current.contains(e.target as Node)) {
        setFilterOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [filterOpen]);

  const filtered = useMemo(() => {
    const seasonStart = thisSeasonActive ? currentSeasonStart() : null;
    return recordings.filter((r) => {
      if (query.trim()) {
        const haystack = `${cleanTitle(r.name)} ${r.filename}`.toLowerCase();
        if (!haystack.includes(query.trim().toLowerCase())) return false;
      }
      if (seasonStart && new Date(r.createdAt) < seasonStart) return false;
      if (statusFilter !== 'all' && r.status !== statusFilter) return false;
      return true;
    });
  }, [recordings, query, thisSeasonActive, statusFilter]);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] gap-2">
        <BounceLoader size={240} />
        <p className="text-sm text-ink-mute">Loading recordings.</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-[1280px] mx-auto px-6 py-12">
        <p className="text-sm text-clay">Could not load recordings: {error}</p>
      </div>
    );
  }

  const isSelectionMode = selectedIds.size > 0;
  const allFilteredSelected = filtered.length > 0 && filtered.every((r) => selectedIds.has(r.id));
  // Desktop grid columns (checkbox | date | title | actions [| ×delete]). On
  // mobile the rows fall back to flex so the title is never squeezed to zero.
  const mdGridCols = isSelectionMode
    ? 'md:grid-cols-[32px_110px_1fr_80px]'
    : 'md:grid-cols-[32px_110px_1fr_80px_36px]';

  return (
    <div className="max-w-[1280px] mx-auto px-6">
      {/* Page head */}
      <div className="flex items-end justify-between gap-6 flex-wrap" style={{ paddingTop: 52, paddingBottom: 28 }}>
        <div>
          <span className="inline-flex items-center gap-2 font-mono text-[0.72rem] uppercase tracking-[0.18em] text-court before:content-[''] before:w-1.5 before:h-1.5 before:bg-clay before:rounded-full">
            Library
          </span>
          <h1
            className="font-display font-medium tracking-tight mt-3.5"
            style={{
              fontSize: 'clamp(40px, 4.6vw, 64px)',
              lineHeight: 1.0,
              letterSpacing: '-0.022em',
            }}
          >
            Every recording, <em>scrubbable.</em>
          </h1>
          <p className="text-ink-soft text-base mt-2">
            <span className="font-display" style={{ fontFeatureSettings: '"tnum"' }}>
              {recordings.length}
            </span>{' '}
            {recordings.length === 1 ? 'recording' : 'recordings'}
          </p>
        </div>

        <Link
          href="/upload"
          className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full bg-ink text-cream dark:bg-court-deep font-medium text-[0.95rem] hover:-translate-y-px"
          style={{ transition: 'transform var(--duration-quick) var(--ease-spring), background var(--duration-base) var(--ease-out)' }}
        >
          <svg viewBox="0 0 24 24" width={16} height={16} fill="none" stroke="currentColor" strokeWidth={1.75} strokeLinecap="round" strokeLinejoin="round">
            <path d="M5 12h14" />
            <path d="M12 5v14" />
          </svg>
          Upload video
        </Link>
      </div>

      {/* Filter bar — stacks on mobile (search row, then chips row) */}
      <div className="flex flex-col md:flex-row md:items-center gap-2.5 md:gap-3 px-4 py-2.5 md:py-3 bg-paper border border-line rounded-[22px] md:rounded-full mb-7">
        <div className="flex items-center gap-2.5 flex-1 min-w-0">
          <svg viewBox="0 0 24 24" width={16} height={16} fill="none" stroke="currentColor" strokeWidth={1.75} strokeLinecap="round" strokeLinejoin="round" className="text-ink-mute shrink-0">
            <circle cx="11" cy="11" r="8" />
            <path d="m21 21-4.3-4.3" />
          </svg>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search by title or filename"
            className="flex-1 min-w-0 bg-transparent border-none text-[0.94rem] text-ink py-1 outline-none placeholder:text-ink-mute"
          />
        </div>
        {/* Divider (desktop only) + thin rule (mobile only) between the two rows */}
        <span className="hidden md:block w-px h-5 bg-line" />
        <span className="md:hidden h-px -mt-0.5 bg-line-soft" />
        {/* Chips + count. `md:contents` dissolves this wrapper on desktop so the
            chips and count flow as direct children of the bar (count → far right). */}
        <div className="flex items-center gap-2 md:contents">
        <Chip active={thisSeasonActive} onClick={() => setThisSeasonActive((v) => !v)}>
          <svg viewBox="0 0 24 24" width={13} height={13} fill="none" stroke="currentColor" strokeWidth={1.75} strokeLinecap="round" strokeLinejoin="round">
            <path d="M8 2v4" />
            <path d="M16 2v4" />
            <rect width="18" height="18" x="3" y="4" rx="2" />
            <path d="M3 10h18" />
          </svg>
          This season
        </Chip>
        <div ref={filterRef} className="relative">
          <Chip active={statusFilter !== 'all'} onClick={() => setFilterOpen((v) => !v)}>
            <svg viewBox="0 0 24 24" width={13} height={13} fill="none" stroke="currentColor" strokeWidth={1.75} strokeLinecap="round" strokeLinejoin="round">
              <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" />
            </svg>
            Filter{statusFilter !== 'all' ? ` · ${statusFilter}` : ''}
          </Chip>
          {filterOpen && (
            <div
              className="absolute left-0 top-[calc(100%+8px)] z-10 bg-paper border border-line rounded-[10px] shadow-lg py-1.5 min-w-[160px]"
              style={{ boxShadow: '0 4px 24px rgba(0,0,0,0.10)' }}
            >
              {(['all', 'done', 'processing', 'failed'] as const).map((s) => (
                <button
                  key={s}
                  type="button"
                  onClick={() => { setStatusFilter(s); setFilterOpen(false); }}
                  className={`w-full text-left px-4 py-2 text-[0.88rem] capitalize flex items-center justify-between gap-3 ${
                    statusFilter === s ? 'text-ink font-medium' : 'text-ink-soft hover:text-ink'
                  }`}
                >
                  {s === 'all' ? 'All statuses' : s.charAt(0).toUpperCase() + s.slice(1)}
                  {statusFilter === s && (
                    <svg viewBox="0 0 24 24" width={12} height={12} fill="none" stroke="currentColor" strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round">
                      <path d="M20 6 9 17l-5-5" />
                    </svg>
                  )}
                </button>
              ))}
            </div>
          )}
        </div>
        <span className="ml-auto font-mono text-[0.72rem] uppercase tracking-[0.12em] text-ink-mute">
          {filtered.length} {filtered.length === 1 ? 'recording' : 'recordings'}
        </span>
        </div>
      </div>

      {/* Match table */}
      {recordings.length === 0 ? (
        <EmptyState />
      ) : (
        <div className="bg-paper border border-line rounded-[14px] overflow-hidden mb-9">
          {/* Header — hidden on mobile (rows become stacked cards there) */}
          <div
            className={`hidden md:grid items-center px-6 py-3.5 bg-shade dark:bg-surface font-mono text-[0.66rem] uppercase tracking-[0.14em] text-ink-mute gap-4 ${mdGridCols}`}
            style={{ borderBottom: '1px solid var(--color-line-soft)' }}
          >
            {/* Select-all checkbox — always in col 1 */}
            <button
              type="button"
              onClick={allFilteredSelected ? clearSelection : selectAllFiltered}
              aria-label={allFilteredSelected ? 'Deselect all' : 'Select all'}
              className="w-5 h-5 rounded-[5px] border-2 flex items-center justify-center shrink-0"
              style={{
                opacity: isSelectionMode ? 1 : 0,
                pointerEvents: isSelectionMode ? 'auto' : 'none',
                borderColor: allFilteredSelected ? 'var(--color-clay)' : 'var(--color-ink-mute)',
                background: allFilteredSelected ? 'var(--color-clay)' : 'transparent',
                transition: 'opacity var(--duration-quick) var(--ease-out), border-color var(--duration-quick) var(--ease-out), background var(--duration-quick) var(--ease-out)',
              }}
            >
              {allFilteredSelected ? (
                <svg viewBox="0 0 24 24" width={11} height={11} fill="none" stroke="white" strokeWidth={3} strokeLinecap="round" strokeLinejoin="round">
                  <path d="M20 6 9 17l-5-5" />
                </svg>
              ) : (
                <svg viewBox="0 0 24 24" width={10} height={10} fill="none" stroke="var(--color-ink-mute)" strokeWidth={3} strokeLinecap="round">
                  <path d="M5 12h14" />
                </svg>
              )}
            </button>
            <span>Date</span>
            <span>Title</span>
            <span />
            {!isSelectionMode && <span />}
          </div>

          {filtered.length === 0 ? (
            <div className="px-6 py-10 text-center text-ink-soft text-sm">
              No recordings match your filters.
            </div>
          ) : (
            filtered.map((rec, i) => {
              const isLast = i === filtered.length - 1;
              const isConfirming = confirmDelete === rec.id;
              const isDeleting = deletingId === rec.id;
              const isSelected = selectedIds.has(rec.id);

              if (isConfirming) {
                return (
                  <div
                    key={rec.id}
                    className="px-6 py-4"
                    style={{ borderBottom: isLast ? 'none' : '1px solid var(--color-line-soft)' }}
                  >
                    <div className="flex items-center justify-between gap-3 flex-wrap">
                      <span className="text-clay font-display font-medium text-[1.0rem]">
                        Delete this recording?
                      </span>
                      <div className="flex items-center gap-2">
                        <button
                          type="button"
                          onClick={(e) => { e.stopPropagation(); setConfirmDelete(null); }}
                          className="inline-flex items-center px-4 py-1.5 rounded-full border border-line bg-paper text-ink-soft hover:text-ink hover:border-ink-mute text-[0.85rem] font-medium cursor-pointer"
                          style={{ transition: 'border-color var(--duration-quick) var(--ease-out), color var(--duration-quick) var(--ease-out)' }}
                        >
                          Cancel
                        </button>
                        <button
                          type="button"
                          disabled={isDeleting}
                          onClick={(e) => { e.stopPropagation(); deleteRecording(rec.id); }}
                          className="inline-flex items-center px-4 py-1.5 rounded-full bg-clay text-cream text-[0.85rem] font-medium hover:-translate-y-px disabled:opacity-60 disabled:cursor-not-allowed cursor-pointer"
                          style={{ transition: 'transform var(--duration-quick) var(--ease-spring)' }}
                        >
                          {isDeleting ? 'Deleting...' : 'Delete'}
                        </button>
                      </div>
                    </div>
                    {deleteError && (
                      <p className="mt-2 text-[0.82rem] text-clay">{deleteError}</p>
                    )}
                  </div>
                );
              }

              const titleSeed = cleanTitle(rec.name);
              return (
                <div
                  key={rec.id}
                  role="button"
                  tabIndex={0}
                  onClick={() => {
                    if (isSelectionMode) {
                      toggleSelect(rec.id);
                    } else {
                      router.push(`/recordings/${rec.id}`);
                    }
                  }}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      if (isSelectionMode) toggleSelect(rec.id);
                      else router.push(`/recordings/${rec.id}`);
                    }
                  }}
                  className={`cc-match-row group flex md:grid items-center px-4 md:px-6 py-3.5 md:py-4 gap-3 md:gap-4 cursor-pointer ${mdGridCols}`}
                  style={{
                    borderBottom: isLast ? 'none' : '1px solid var(--color-line-soft)',
                    background: isSelected ? 'var(--color-shade)' : undefined,
                    transition: 'background var(--duration-quick) var(--ease-out)',
                  }}
                >
                  {/* Checkbox — col 1. Always visible on mobile (no hover on
                      touch); hover-reveal on desktop until something's selected. */}
                  <div
                    className="flex items-center justify-center shrink-0"
                    onClick={(e) => { e.stopPropagation(); toggleSelect(rec.id); }}
                  >
                    <span
                      className={`w-5 h-5 rounded-[5px] border-2 flex items-center justify-center shrink-0 ${
                        !isSelectionMode && !isSelected ? 'opacity-100 md:opacity-0 md:group-hover:opacity-100' : ''
                      }`}
                      style={{
                        borderColor: isSelected ? 'var(--color-clay)' : 'var(--color-line)',
                        background: isSelected ? 'var(--color-clay)' : 'transparent',
                        transition: 'opacity var(--duration-quick) var(--ease-out), border-color var(--duration-quick) var(--ease-out), background var(--duration-quick) var(--ease-out)',
                      }}
                    >
                      {isSelected && (
                        <svg viewBox="0 0 24 24" width={11} height={11} fill="none" stroke="white" strokeWidth={3} strokeLinecap="round" strokeLinejoin="round">
                          <path d="M20 6 9 17l-5-5" />
                        </svg>
                      )}
                    </span>
                  </div>

                  {/* Date — its own column on desktop only */}
                  <span
                    className="hidden md:block font-mono text-ink-mute uppercase"
                    style={{ fontSize: '0.78rem', fontFeatureSettings: '"tnum"' }}
                  >
                    {fmtDate(rec.createdAt)}
                  </span>

                  {/* Title column (flex-1 on mobile so it never gets squeezed) */}
                  <div className="flex items-center gap-3 min-w-0 flex-1">
                    <div
                      className="w-9 h-9 md:w-8 md:h-8 rounded-full shrink-0 flex items-center justify-center text-cream font-display font-medium text-[0.9rem] md:text-[0.85rem]"
                      style={{ background: hashColor(titleSeed) }}
                    >
                      {initials(titleSeed) || '·'}
                    </div>
                    <div className="flex flex-col min-w-0 gap-px">
                      <span className="font-display font-medium text-[1.05rem] tracking-tight truncate">
                        {cleanTitle(rec.name)}
                      </span>
                      {/* Mobile meta: date (+ status). Replaces the hidden date column. */}
                      <span
                        className="md:hidden font-mono text-[0.7rem] uppercase tracking-[0.08em] text-ink-mute"
                        style={{ fontFeatureSettings: '"tnum"' }}
                      >
                        {fmtDate(rec.createdAt)}
                        {rec.status !== 'done' &&
                          ` · ${rec.status === 'failed' ? 'Needs attention' : 'Processing'}`}
                      </span>
                      {/* Desktop status line */}
                      {rec.status !== 'done' && (
                        <span className="hidden md:block text-[0.78rem] text-ink-mute">
                          {rec.status === 'failed' ? 'Needs attention' : 'Processing'}
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Actions column */}
                  <div className="justify-self-end flex items-center gap-1.5 shrink-0">
                    <EditableName
                      recordingId={rec.id}
                      initialName={rec.name}
                      variant="row"
                      onSaved={(newName) =>
                        mutate(
                          (curr) =>
                            curr
                              ? {
                                  recordings: curr.recordings.map((p) =>
                                    p.id === rec.id ? { ...p, name: newName } : p,
                                  ),
                                }
                              : curr,
                          { revalidate: false },
                        )
                      }
                    />
                    {!isSelectionMode && <span className="cc-arrow hidden md:inline text-ink-mute">→</span>}
                  </div>

                  {/* Individual delete — hidden in selection mode */}
                  {!isSelectionMode && (
                    <button
                      type="button"
                      aria-label="Delete recording"
                      onClick={(e) => {
                        e.stopPropagation();
                        setDeleteError(null);
                        setConfirmDelete(rec.id);
                      }}
                      className="w-7 h-7 shrink-0 rounded-full border border-line bg-paper text-ink-mute hover:border-clay hover:text-clay inline-flex items-center justify-center cursor-pointer transition-colors justify-self-end"
                    >
                      <svg viewBox="0 0 24 24" width={14} height={14} fill="none" stroke="currentColor" strokeWidth={1.75} strokeLinecap="round" strokeLinejoin="round">
                        <path d="M18 6 6 18" />
                        <path d="m6 6 12 12" />
                      </svg>
                    </button>
                  )}
                </div>
              );
            })
          )}
        </div>
      )}

      {/* Bulk action bar — slides up from bottom when items are selected */}
      <div
        style={{
          position: 'fixed',
          bottom: selectedIds.size > 0 ? 28 : -72,
          left: '50%',
          transform: 'translateX(-50%)',
          transition: 'bottom 0.28s cubic-bezier(0.34, 1.4, 0.64, 1)',
          zIndex: 50,
          pointerEvents: selectedIds.size > 0 ? 'auto' : 'none',
        }}
      >
        <div
          className="flex items-center gap-1.5 px-2 py-2 rounded-full"
          style={{
            background: 'var(--color-ink)',
            color: 'var(--color-cream)',
            boxShadow: '0 8px 36px rgba(0,0,0,0.28), 0 2px 8px rgba(0,0,0,0.12)',
            minWidth: 320,
          }}
        >
          {bulkState === 'confirming' ? (
            <>
              <span className="flex-1 pl-3 text-[0.9rem] font-medium" style={{ color: 'var(--color-clay)' }}>
                Delete {selectedIds.size} {selectedIds.size === 1 ? 'recording' : 'recordings'}?
              </span>
              {bulkError && (
                <span className="text-[0.78rem] shrink-0" style={{ color: 'var(--color-clay)' }}>
                  {bulkError}
                </span>
              )}
              <button
                type="button"
                onClick={() => { setBulkState('idle'); setBulkError(null); }}
                className="px-3.5 py-1.5 rounded-full text-[0.85rem] font-medium shrink-0"
                style={{ color: 'var(--color-cream)', opacity: 0.6 }}
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={bulkDelete}
                className="px-4 py-1.5 rounded-full text-[0.85rem] font-semibold shrink-0"
                style={{ background: 'var(--color-clay)', color: 'var(--color-cream)' }}
              >
                Confirm
              </button>
            </>
          ) : bulkState === 'deleting' ? (
            <span className="flex-1 pl-3 text-[0.9rem] font-medium" style={{ opacity: 0.7 }}>
              Deleting {selectedIds.size}…
            </span>
          ) : (
            <>
              {/* Count badge */}
              <span
                className="ml-1 px-2.5 py-1 rounded-full font-mono text-[0.78rem] font-semibold shrink-0"
                style={{ background: 'var(--color-clay)', color: 'var(--color-cream)' }}
              >
                {selectedIds.size}
              </span>

              <span className="flex-1 pl-1.5 text-[0.9rem] font-medium" style={{ opacity: 0.85 }}>
                {selectedIds.size === 1 ? 'recording selected' : 'recordings selected'}
              </span>

              {/* Select all shortcut — only shown when not all are selected */}
              {!allFilteredSelected && filtered.length > 0 && (
                <button
                  type="button"
                  onClick={selectAllFiltered}
                  className="px-3.5 py-1.5 rounded-full text-[0.82rem] font-medium shrink-0 hover:opacity-80"
                  style={{ color: 'var(--color-cream)', opacity: 0.6 }}
                >
                  Select all {filtered.length}
                </button>
              )}

              {/* Deselect all */}
              <button
                type="button"
                onClick={clearSelection}
                aria-label="Clear selection"
                className="w-8 h-8 rounded-full flex items-center justify-center shrink-0 hover:opacity-70"
                style={{ color: 'var(--color-cream)', opacity: 0.5 }}
              >
                <svg viewBox="0 0 24 24" width={14} height={14} fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
                  <path d="M18 6 6 18" />
                  <path d="m6 6 12 12" />
                </svg>
              </button>

              {/* Divider */}
              <span className="w-px h-5 shrink-0" style={{ background: 'rgba(255,255,255,0.15)' }} />

              {/* Delete button */}
              <button
                type="button"
                onClick={() => setBulkState('confirming')}
                className="px-4 py-1.5 rounded-full text-[0.88rem] font-semibold shrink-0 hover:-translate-y-px"
                style={{
                  background: 'var(--color-clay)',
                  color: 'var(--color-cream)',
                  transition: 'transform var(--duration-quick) var(--ease-spring)',
                }}
              >
                Delete {selectedIds.size}
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function Chip({
  active = false,
  onClick,
  children,
}: {
  active?: boolean;
  onClick?: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full border text-[0.85rem] font-medium ${
        active
          ? 'border-ink text-ink bg-shade dark:bg-surface font-semibold'
          : 'border-line bg-transparent text-ink-soft hover:text-ink hover:border-ink-mute'
      }`}
      style={{ transition: 'border-color var(--duration-quick) var(--ease-out), color var(--duration-quick) var(--ease-out), background var(--duration-quick) var(--ease-out)' }}
    >
      {children}
    </button>
  );
}

function EmptyState() {
  return (
    <div className="rounded-[14px] p-12 text-center bg-paper border border-line mb-9">
      <svg
        viewBox="0 0 24 24"
        className="w-10 h-10 mx-auto mb-4 text-ink-mute"
        fill="none"
        stroke="currentColor"
        strokeWidth={1.25}
      >
        <rect x="3" y="4" width="18" height="16" rx="2" />
        <path d="M7 4v16M17 4v16M3 9h4M3 14h4M17 9h4M17 14h4" />
      </svg>
      <p className="font-display font-medium text-[1.15rem] mb-1">
        Upload your first recording.
      </p>
      <p className="text-sm text-ink-soft mb-6">
        A match goes in, every shot and pattern comes out.
      </p>
      <Link
        href="/upload"
        className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full bg-ink text-cream dark:bg-court-deep font-medium text-[0.95rem]"
      >
        <svg viewBox="0 0 24 24" width={16} height={16} fill="none" stroke="currentColor" strokeWidth={1.75} strokeLinecap="round" strokeLinejoin="round">
          <path d="M5 12h14" />
          <path d="M12 5v14" />
        </svg>
        Upload video
      </Link>
    </div>
  );
}
