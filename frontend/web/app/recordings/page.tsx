'use client';

import { useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import EditableName from '@/components/recordings/EditableName';
import BounceLoader from '@/components/upload/BounceLoader';

/**
 * Recordings list. Ported from docs/brand-drop/mocks/matches-list.html.
 *
 * Layout:
 *   - h1 "Every recording, scrubbable." (clay italic on "scrubbable.")
 *   - Mono meta line: results count ("12 recordings.")
 *   - Filter bar: search + player chips + (placeholder) date/filter chips
 *   - Match table: .cc-match-row grid (110px date | 1fr name | 40px arrow)
 *   - Empty state CTA → /upload
 *
 * Wires to /api/recordings. Player vs opponent is derived from `name`
 * (display string, e.g. "M. Lin vs D. Park") or falls back to `filename`.
 */

interface Recording {
  id: string;
  status: 'pending' | 'processing' | 'done' | 'failed';
  createdAt: string;
  /** User-facing title (e.g. "test5-player", "StMarys Court2 4950 clip"). */
  name: string;
  filename: string;
  /** Real player name from the players table when player_id is bound,
   *  null otherwise. The list filters + displays by THIS, not by parsing
   *  the title — earlier the chips read the title as the player and ended
   *  up filtering on filenames like "test5-player". */
  playerName: string | null;
}

type RawRecording = {
  id: string;
  status: Recording['status'];
  createdAt: string;
  name: string;
  filename: string;
  playerName: string | null;
};

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
  const [recordings, setRecordings] = useState<Recording[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const [playerFilter, setPlayerFilter] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  async function deleteRecording(id: string) {
    setDeletingId(id);
    setDeleteError(null);
    try {
      const res = await fetch(`/api/recordings/${id}`, { method: 'DELETE' });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || `Delete failed (${res.status})`);
      }
      setRecordings((prev) => prev.filter((r) => r.id !== id));
      setConfirmDelete(null);
    } catch (e) {
      setDeleteError((e as Error).message);
    } finally {
      setDeletingId(null);
    }
  }

  useEffect(() => {
    fetch('/api/recordings')
      .then((r) => {
        if (!r.ok) throw new Error('Failed to fetch recordings');
        return r.json();
      })
      .then((d: { recordings: RawRecording[] }) => {
        setRecordings(
          d.recordings.map((r) => ({
            id: r.id,
            status: r.status,
            createdAt: r.createdAt,
            name: r.name,
            filename: r.filename,
            playerName: r.playerName ?? null,
          }))
        );
      })
      .catch((e) => setError((e as Error).message))
      .finally(() => setLoading(false));
  }, []);

  // Filter chips from REAL bound player names (matches.player_id → players.name).
  // Recordings without a player binding don't contribute a chip — the title
  // isn't a player, so showing "test5-player" or "new insights" as a chip
  // was misleading.
  const uniquePlayers = useMemo(() => {
    const seen = new Set<string>();
    recordings.forEach((r) => {
      if (r.playerName) seen.add(r.playerName);
    });
    return Array.from(seen).sort().slice(0, 8);
  }, [recordings]);

  const filtered = useMemo(() => {
    return recordings.filter((r) => {
      if (playerFilter && r.playerName !== playerFilter) return false;
      if (query.trim()) {
        const haystack = `${cleanTitle(r.name)} ${r.playerName ?? ''} ${r.filename}`.toLowerCase();
        if (!haystack.includes(query.trim().toLowerCase())) return false;
      }
      return true;
    });
  }, [recordings, query, playerFilter]);

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
            {' · '}
            <span className="font-display" style={{ fontFeatureSettings: '"tnum"' }}>
              {new Set(recordings.map((r) => r.playerName).filter(Boolean)).size}
            </span>{' '}
            players
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

      {/* Filter bar */}
      <div className="flex items-center gap-3 px-4 py-3 bg-paper border border-line rounded-full mb-7 flex-wrap">
        <div className="flex items-center gap-2.5 flex-1 min-w-[200px]" style={{ flexBasis: 240 }}>
          <svg viewBox="0 0 24 24" width={16} height={16} fill="none" stroke="currentColor" strokeWidth={1.75} strokeLinecap="round" strokeLinejoin="round" className="text-ink-mute shrink-0">
            <circle cx="11" cy="11" r="8" />
            <path d="m21 21-4.3-4.3" />
          </svg>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search by player, opponent, or filename"
            className="flex-1 bg-transparent border-none text-[0.94rem] text-ink py-1 outline-none placeholder:text-ink-mute"
          />
        </div>
        <span className="w-px h-5 bg-line" />
        <Chip
          active={playerFilter === null}
          onClick={() => setPlayerFilter(null)}
        >
          All players
        </Chip>
        {uniquePlayers.map((p) => (
          <Chip
            key={p}
            active={playerFilter === p}
            onClick={() => setPlayerFilter(playerFilter === p ? null : p)}
          >
            {p}
          </Chip>
        ))}
        <span className="w-px h-5 bg-line" />
        <Chip>
          <svg viewBox="0 0 24 24" width={13} height={13} fill="none" stroke="currentColor" strokeWidth={1.75} strokeLinecap="round" strokeLinejoin="round">
            <path d="M8 2v4" />
            <path d="M16 2v4" />
            <rect width="18" height="18" x="3" y="4" rx="2" />
            <path d="M3 10h18" />
          </svg>
          This season
        </Chip>
        <Chip>
          <svg viewBox="0 0 24 24" width={13} height={13} fill="none" stroke="currentColor" strokeWidth={1.75} strokeLinecap="round" strokeLinejoin="round">
            <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" />
          </svg>
          Filter
        </Chip>
        <span className="ml-auto font-mono text-[0.72rem] uppercase tracking-[0.12em] text-ink-mute">
          {filtered.length} {filtered.length === 1 ? 'recording' : 'recordings'}
        </span>
      </div>

      {/* Match table */}
      {recordings.length === 0 ? (
        <EmptyState />
      ) : (
        <div className="bg-paper border border-line rounded-[14px] overflow-hidden mb-9">
          <div
            className="grid items-center px-6 py-3.5 bg-shade dark:bg-surface font-mono text-[0.66rem] uppercase tracking-[0.14em] text-ink-mute gap-4"
            style={{ gridTemplateColumns: '110px 1fr 160px 80px 36px', borderBottom: '1px solid var(--color-line-soft)' }}
          >
            <span>Date</span>
            <span>Title</span>
            <span>Player</span>
            <span />
            <span />
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

              if (isConfirming) {
                return (
                  <div
                    key={rec.id}
                    className="px-6 py-4"
                    style={{
                      borderBottom: isLast ? 'none' : '1px solid var(--color-line-soft)',
                    }}
                  >
                    <div className="flex items-center justify-between gap-3 flex-wrap">
                      <span className="text-clay font-display font-medium text-[1.0rem]">
                        Delete this recording?
                      </span>
                      <div className="flex items-center gap-2">
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            setConfirmDelete(null);
                          }}
                          className="inline-flex items-center px-4 py-1.5 rounded-full border border-line bg-paper text-ink-soft hover:text-ink hover:border-ink-mute text-[0.85rem] font-medium cursor-pointer"
                          style={{ transition: 'border-color var(--duration-quick) var(--ease-out), color var(--duration-quick) var(--ease-out)' }}
                        >
                          Cancel
                        </button>
                        <button
                          type="button"
                          disabled={isDeleting}
                          onClick={(e) => {
                            e.stopPropagation();
                            deleteRecording(rec.id);
                          }}
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

              const titleSeed = rec.playerName ?? cleanTitle(rec.name);
              return (
                <div
                  key={rec.id}
                  role="button"
                  tabIndex={0}
                  onClick={() => router.push(`/recordings/${rec.id}`)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      router.push(`/recordings/${rec.id}`);
                    }
                  }}
                  className="cc-match-row grid items-center px-6 py-4 gap-4 cursor-pointer"
                  style={{
                    gridTemplateColumns: '110px 1fr 160px 80px 36px',
                    borderBottom: isLast ? 'none' : '1px solid var(--color-line-soft)',
                  }}
                >
                  <span
                    className="font-mono text-ink-mute uppercase"
                    style={{ fontSize: '0.78rem', fontFeatureSettings: '"tnum"' }}
                  >
                    {fmtDate(rec.createdAt)}
                  </span>
                  {/* Title column — actual recording title (rec.name), no
                      "player vs opponent" parsing. The avatar is keyed off the
                      real player when bound, otherwise the title text. */}
                  <div className="flex items-center gap-3 min-w-0">
                    <div
                      className="w-8 h-8 rounded-full shrink-0 flex items-center justify-center text-cream font-display font-medium text-[0.85rem]"
                      style={{ background: hashColor(titleSeed) }}
                    >
                      {initials(titleSeed) || '·'}
                    </div>
                    <div className="flex flex-col min-w-0 gap-px">
                      <span className="font-display font-medium text-[1.05rem] tracking-tight truncate">
                        {cleanTitle(rec.name)}
                      </span>
                      {rec.status !== 'done' && (
                        <span className="text-[0.78rem] text-ink-mute">
                          {rec.status === 'failed' ? 'Needs attention' : 'Processing'}
                        </span>
                      )}
                    </div>
                  </div>
                  {/* Player column — real bound player name or em-dash. */}
                  <span className="font-display text-[0.95rem] text-ink-soft truncate">
                    {rec.playerName ?? <span className="text-ink-mute">—</span>}
                  </span>
                  <div className="justify-self-end flex items-center gap-1.5">
                    <EditableName
                      recordingId={rec.id}
                      initialName={rec.name}
                      variant="row"
                      onSaved={(newName) =>
                        setRecordings((prev) =>
                          prev.map((p) => (p.id === rec.id ? { ...p, name: newName } : p)),
                        )
                      }
                    />
                    <span className="cc-arrow text-ink-mute">→</span>
                  </div>
                  <button
                    type="button"
                    aria-label="Delete recording"
                    onClick={(e) => {
                      e.stopPropagation();
                      setDeleteError(null);
                      setConfirmDelete(rec.id);
                    }}
                    className="w-7 h-7 rounded-full border border-line bg-paper text-ink-mute hover:border-clay hover:text-clay inline-flex items-center justify-center cursor-pointer transition-colors justify-self-end"
                  >
                    <svg viewBox="0 0 24 24" width={14} height={14} fill="none" stroke="currentColor" strokeWidth={1.75} strokeLinecap="round" strokeLinejoin="round">
                      <path d="M18 6 6 18" />
                      <path d="m6 6 12 12" />
                    </svg>
                  </button>
                </div>
              );
            })
          )}
        </div>
      )}
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
    <div
      className="rounded-[14px] p-12 text-center bg-paper border border-line mb-9"
    >
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
