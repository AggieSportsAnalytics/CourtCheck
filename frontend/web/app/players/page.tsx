'use client'

import { useEffect, useMemo, useState } from 'react'
import Link from 'next/link'

import { PlayerCard, type PlayerCardData } from '@/components/players/PlayerCard'
import { Eyebrow } from '@/components/ui/eyebrow'
import { Display, Num } from '@/components/ui/display'
import { Button } from '@/components/ui/button'

interface ApiPlayer {
  id: string
  name: string
  year: string | null
  position: string | null
  photo_url: string | null
}

interface ApiRecording {
  id: string
  name: string
  status: string
  createdAt: string
  player_id?: string | null
  fps: number | null
  numFrames: number | null
  shotCount: number | null
  forehandCount: number | null
  backhandCount: number | null
  serveCount: number | null
  inBoundsBounces: number | null
  outBoundsBounces: number | null
}

type SortKey = 'name' | 'recordings' | 'first-serve'

const YEAR_FILTERS = [
  { key: 'all', label: 'All players' },
  { key: 'senior', label: 'Seniors' },
  { key: 'junior', label: 'Juniors' },
  { key: 'sophomore', label: 'Sophomores' },
  { key: 'freshman', label: 'Freshmen' },
] as const

// Map any year string the DB might hold to one of the canonical filter keys.
// Accepts "Sr." / "Sr" / "senior" / "Senior" / "SR" — all collapse to "senior".
function normalizeYear(raw: string | null | undefined): string {
  if (!raw) return ''
  const v = raw.trim().toLowerCase().replace(/\./g, '')
  if (v === 'sr' || v === 'senior') return 'senior'
  if (v === 'jr' || v === 'junior') return 'junior'
  if (v === 'so' || v === 'soph' || v === 'sophomore') return 'sophomore'
  if (v === 'fr' || v === 'fresh' || v === 'freshman') return 'freshman'
  return v
}

function inPct(inB: number | null, outB: number | null): number | null {
  if (inB == null || outB == null) return null
  const total = inB + outB
  return total > 0 ? Math.round((inB / total) * 100) : null
}

// Roll up recordings into a per-player card payload. Real fields where they
// exist, null/derived where they don't.
function buildCards(
  players: ApiPlayer[],
  recordings: ApiRecording[],
): PlayerCardData[] {
  const byPlayer = new Map<string, ApiRecording[]>()
  for (const rec of recordings) {
    if (!rec.player_id) continue
    const arr = byPlayer.get(rec.player_id) ?? []
    arr.push(rec)
    byPlayer.set(rec.player_id, arr)
  }

  return players.map<PlayerCardData>((p) => {
    const recs = (byPlayer.get(p.id) ?? []).slice().sort(
      (a, b) =>
        new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime(),
    )

    // Hours recorded = sum of (numFrames / fps) / 3600.
    let totalSecs = 0
    let inB = 0
    let outB = 0
    let fh = 0
    let bh = 0
    let sv = 0
    for (const r of recs) {
      if (r.fps && r.numFrames) totalSecs += r.numFrames / r.fps
      inB += r.inBoundsBounces ?? 0
      outB += r.outBoundsBounces ?? 0
      fh += r.forehandCount ?? 0
      bh += r.backhandCount ?? 0
      sv += r.serveCount ?? 0
    }

    // Baseline % = in-bounds rate (proxy until we have a real baseline stat).
    const baselinePct = inPct(inB, outB)
    // Net win % — no backend field yet. Surface dominant-stroke share as the
    // "non-baseline" proxy so the slot isn't dead.
    const totalStrokes = fh + bh + sv
    const netPct =
      totalStrokes > 0 ? Math.round(((bh + sv) / totalStrokes) * 100) : null
    // 1st serve % — no first-serve-in field yet. Use serve share as best proxy.
    const firstServePct =
      totalStrokes > 0 ? Math.round((sv / totalStrokes) * 100) : null

    return {
      id: p.id,
      name: p.name,
      year: p.year,
      position: p.position,
      photo_url: p.photo_url,
      matchCount: recs.length,
      hoursRecorded: totalSecs > 0 ? totalSecs / 3600 : null,
      baselinePct,
      netPct,
      firstServePct,
      recent: recs.slice(0, 3).map((r) => ({
        id: r.id,
        label: r.name,
        date: r.createdAt,
        // No win/loss field on recordings yet. Leave null so the pill renders
        // an em-dash instead of fabricating a result.
        result: null,
      })),
    }
  })
}

export default function PlayersPage() {
  const [players, setPlayers] = useState<ApiPlayer[]>([])
  const [recordings, setRecordings] = useState<ApiRecording[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const [query, setQuery] = useState('')
  const [yearFilter, setYearFilter] = useState<string>('all')
  const [sort, setSort] = useState<SortKey>('name')

  useEffect(() => {
    let cancelled = false
    Promise.all([
      fetch('/api/players').then((r) =>
        r.ok ? r.json() : Promise.reject(new Error('Failed to load players')),
      ),
      fetch('/api/recordings').then((r) =>
        r.ok ? r.json() : Promise.reject(new Error('Failed to load recordings')),
      ),
    ])
      .then(([pData, rData]) => {
        if (cancelled) return
        setPlayers((pData.players ?? []) as ApiPlayer[])
        setRecordings((rData.recordings ?? []) as ApiRecording[])
      })
      .catch((e: Error) => {
        if (!cancelled) setError(e.message)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  const cards = useMemo(
    () => buildCards(players, recordings),
    [players, recordings],
  )

  const filtered = useMemo(() => {
    let xs = cards
    if (yearFilter !== 'all') {
      xs = xs.filter((c) => normalizeYear(c.year) === yearFilter)
    }
    if (query.trim()) {
      const q = query.trim().toLowerCase()
      xs = xs.filter((c) => c.name.toLowerCase().includes(q))
    }
    const sorted = xs.slice()
    if (sort === 'name') {
      sorted.sort((a, b) => a.name.localeCompare(b.name))
    } else if (sort === 'recordings') {
      sorted.sort((a, b) => b.matchCount - a.matchCount)
    } else if (sort === 'first-serve') {
      sorted.sort((a, b) => (b.firstServePct ?? -1) - (a.firstServePct ?? -1))
    }
    return sorted
  }, [cards, yearFilter, query, sort])

  const totalRecordings = recordings.length

  return (
    <div className="mx-auto max-w-[1200px] px-6 py-12">
      {/* HEAD */}
      <header className="pb-7 pt-3">
        <div className="flex flex-wrap items-end justify-between gap-6">
          <div>
            <Eyebrow>Roster</Eyebrow>
            <Display as="h1" size="lg" className="mt-3.5 mb-2">
              Your <em>team</em>.
            </Display>
            <p className="text-base text-ink-soft">
              <Num size="sm">{players.length}</Num> player{players.length === 1 ? '' : 's'}{' '}
              · <Num size="sm">{totalRecordings}</Num> recording
              {totalRecordings === 1 ? '' : 's'} analyzed this season
            </p>
          </div>
          <div className="flex items-end gap-2.5">
            <Button variant="ghost" size="sm" asChild>
              <Link href="/recordings">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.75} strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="7 10 12 15 17 10" />
                  <line x1="12" y1="15" x2="12" y2="3" />
                </svg>
                Recordings
              </Link>
            </Button>
            <Button variant="ink" size="sm" asChild>
              <Link href="/upload">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.75} strokeLinecap="round" strokeLinejoin="round">
                  <path d="M5 12h14" />
                  <path d="M12 5v14" />
                </svg>
                Upload video
              </Link>
            </Button>
          </div>
        </div>
      </header>

      {/* FILTER + SEARCH */}
      <div className="mb-7 flex flex-wrap items-center gap-3">
        <div className="flex flex-wrap gap-2">
          {YEAR_FILTERS.map((f) => {
            const active = yearFilter === f.key
            return (
              <button
                key={f.key}
                onClick={() => setYearFilter(f.key)}
                className={`cc-legend-chip inline-flex items-center gap-2 rounded-full border px-4 py-2 text-[0.88rem] font-medium ${
                  active
                    ? 'border-ink text-ink font-semibold'
                    : 'border-line text-ink-soft hover:border-ink-mute hover:text-ink'
                } bg-paper`}
              >
                {f.label}
              </button>
            )
          })}
        </div>

        <div className="ml-auto flex items-center gap-2.5">
          <label className="relative">
            <span className="sr-only">Search players</span>
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth={1.75}
              strokeLinecap="round"
              strokeLinejoin="round"
              className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-ink-mute"
            >
              <circle cx="11" cy="11" r="8" />
              <line x1="21" y1="21" x2="16.65" y2="16.65" />
            </svg>
            <input
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search by name"
              className="h-9 rounded-full border border-line bg-paper pl-9 pr-4 text-[0.88rem] text-ink placeholder:text-ink-mute focus:border-ink focus:outline-none"
            />
          </label>

          <label className="flex items-center gap-2 text-[0.78rem] uppercase tracking-[0.12em] text-ink-mute">
            <span className="sr-only">Sort by</span>
            <select
              value={sort}
              onChange={(e) => setSort(e.target.value as SortKey)}
              className="h-9 rounded-full border border-line bg-paper px-3 text-[0.88rem] font-medium text-ink focus:border-ink focus:outline-none"
            >
              <option value="name">Name A–Z</option>
              <option value="recordings">Most recorded</option>
              <option value="first-serve">Serve share</option>
            </select>
          </label>
        </div>
      </div>

      {/* CONTENT */}
      {error && (
        <div
          className="cc-card mb-6 p-5"
          style={{ borderColor: 'color-mix(in srgb, var(--color-clay) 30%, var(--color-line))' }}
        >
          <p className="text-sm text-clay">{error}</p>
        </div>
      )}

      {loading ? (
        <SkeletonGrid />
      ) : filtered.length === 0 ? (
        <EmptyState hasPlayers={players.length > 0} hasFilter={yearFilter !== 'all' || query.trim() !== ''} />
      ) : (
        <div className="mb-9 grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {filtered.map((c) => (
            <PlayerCard key={c.id} player={c} />
          ))}
        </div>
      )}
    </div>
  )
}

function SkeletonGrid() {
  return (
    <div className="mb-9 grid gap-5 sm:grid-cols-2 lg:grid-cols-3" aria-hidden>
      {Array.from({ length: 6 }).map((_, i) => (
        <div
          key={i}
          className="flex h-[260px] flex-col gap-4 rounded-[14px] border border-line bg-paper/60 p-7"
        >
          <div className="flex items-center gap-4">
            <div className="size-14 shrink-0 animate-pulse rounded-full bg-shade" />
            <div className="flex flex-1 flex-col gap-2">
              <div className="h-5 w-2/3 animate-pulse rounded bg-shade" />
              <div className="h-3 w-1/2 animate-pulse rounded bg-shade" />
            </div>
          </div>
          <div className="h-[68px] animate-pulse rounded-[10px] bg-shade" />
          <div className="flex flex-col gap-2">
            <div className="h-3 w-1/3 animate-pulse rounded bg-shade" />
            <div className="h-3 w-full animate-pulse rounded bg-shade" />
            <div className="h-3 w-5/6 animate-pulse rounded bg-shade" />
          </div>
        </div>
      ))}
    </div>
  )
}

function EmptyState({ hasPlayers, hasFilter }: { hasPlayers: boolean; hasFilter: boolean }) {
  if (hasPlayers && hasFilter) {
    return (
      <div className="cc-card mb-9 flex flex-col items-center gap-3 px-6 py-16 text-center">
        <p className="font-display text-2xl text-ink">No players match those filters.</p>
        <p className="text-sm text-ink-soft">Try clearing the search or switching years.</p>
      </div>
    )
  }
  return (
    <div className="cc-card mb-9 flex flex-col items-center gap-4 px-6 py-16 text-center">
      <p className="font-display text-3xl text-ink">No players yet.</p>
      <p className="max-w-md text-sm text-ink-soft">
        Add players to your roster, then upload match footage to start tracking
        per-player analytics.
      </p>
      <div className="mt-2 flex flex-wrap justify-center gap-2.5">
        <Button variant="ink" size="sm" asChild>
          <Link href="/upload">Upload first video</Link>
        </Button>
        <Button variant="ghost" size="sm" asChild>
          <Link href="/recordings">Browse recordings</Link>
        </Button>
      </div>
    </div>
  )
}
