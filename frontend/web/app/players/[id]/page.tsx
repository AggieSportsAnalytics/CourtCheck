'use client'

import { useEffect, useMemo, useState } from 'react'
import { useParams } from 'next/navigation'
import Link from 'next/link'

import { CountUp } from '@/components/players/CountUp'
import { StrokeBars, type StrokeRow } from '@/components/players/StrokeBars'
import { Eyebrow } from '@/components/ui/eyebrow'
import { Display } from '@/components/ui/display'
import { Button } from '@/components/ui/button'
import { isDemoMode, DEMO_PLAYERS, DEMO_RECORDINGS } from '@/lib/demo/demoData'
import { playerPhotoProxyUrl } from '@/lib/utils'

interface ApiPlayer {
  id: string
  name: string
  year: string | null
  position: string | null
  photo_url: string | null
  handedness?: 'right' | 'left' | null
}

type Handedness = 'right' | 'left'

interface ApiRecording {
  id: string
  name: string
  status: string
  createdAt: string
  player_id?: string | null
  fps: number | null
  numFrames: number | null
  shotCount: number | null
  bounceCount: number | null
  rallyCount: number | null
  forehandCount: number | null
  backhandCount: number | null
  serveCount: number | null
  inBoundsBounces: number | null
  outBoundsBounces: number | null
}

function initialsOf(name: string): string {
  return name
    .trim()
    .split(/\s+/)
    .slice(0, 2)
    .map((p) => p[0]?.toUpperCase() ?? '')
    .join('')
}

function splitName(name: string): { lead: string; tail: string } {
  const parts = name.trim().split(/\s+/)
  if (parts.length === 1) return { lead: parts[0], tail: '' }
  const tail = parts[parts.length - 1]
  const lead = parts.slice(0, -1).join(' ')
  return { lead, tail }
}

function formatDateShort(iso: string): string {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ''
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}

function inPct(inB: number | null, outB: number | null): number | null {
  if (inB == null || outB == null) return null
  const t = inB + outB
  return t > 0 ? Math.round((inB / t) * 100) : null
}

export default function PlayerDetailPage() {
  const { id } = useParams<{ id: string }>()
  const [player, setPlayer] = useState<ApiPlayer | null>(null)
  const [recordings, setRecordings] = useState<ApiRecording[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  // Falls the headshot back to initials on a failed/rate-limited proxy fetch,
  // matching the PlayerCard behavior (instead of leaving a blank circle).
  const [photoFailed, setPhotoFailed] = useState(false)

  useEffect(() => {
    if (!id) return

    // Demo mode: a demo-* player resolves from the fabricated roster +
    // recordings so the player screen shows a full stat read with no network.
    if (
      isDemoMode(typeof window !== 'undefined' ? window.location.search : null) &&
      id.startsWith('demo-')
    ) {
      const dp = DEMO_PLAYERS.find((p) => p.id === id) ?? null
      setPlayer(dp as ApiPlayer | null)
      const mine = DEMO_RECORDINGS.filter((r) => r.player_id === id).sort(
        (a, b) =>
          new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime(),
      )
      setRecordings(mine as unknown as ApiRecording[])
      setLoading(false)
      return
    }

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
        const found =
          (pData.players as ApiPlayer[]).find((p) => p.id === id) ?? null
        setPlayer(found)
        const all = (rData.recordings ?? []) as ApiRecording[]
        const mine = all
          .filter((r) => r.player_id === id)
          .sort(
            (a, b) =>
              new Date(b.createdAt).getTime() -
              new Date(a.createdAt).getTime(),
          )
        setRecordings(mine)
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
  }, [id])

  const totals = useMemo(() => {
    let shots = 0
    let bounces = 0
    let rallies = 0
    let secs = 0
    let inB = 0
    let outB = 0
    let fh = 0
    let bh = 0
    let sv = 0
    for (const r of recordings) {
      shots += r.shotCount ?? 0
      bounces += r.bounceCount ?? 0
      rallies += r.rallyCount ?? 0
      if (r.fps && r.numFrames) secs += r.numFrames / r.fps
      inB += r.inBoundsBounces ?? 0
      outB += r.outBoundsBounces ?? 0
      fh += r.forehandCount ?? 0
      bh += r.backhandCount ?? 0
      sv += r.serveCount ?? 0
    }
    return {
      shots,
      bounces,
      rallies,
      hours: secs / 3600,
      accuracyPct: inPct(inB, outB),
      fh,
      bh,
      sv,
    }
  }, [recordings])

  const strokeRows: StrokeRow[] = useMemo(
    () => [
      { key: 'forehand', label: 'Forehand', count: totals.fh },
      { key: 'backhand', label: 'Backhand', count: totals.bh },
      { key: 'serve', label: 'Serve', count: totals.sv },
      // Volley count not in backend yet. Renders as 0% bar with "0 shots".
      { key: 'volley', label: 'Volley', count: 0 },
    ],
    [totals],
  )

  // Last 5 recordings = trend slice
  const last5 = useMemo(() => recordings.slice(0, 5), [recordings])

  if (loading) {
    return (
      <div className="mx-auto max-w-[1200px] px-6 py-12">
        <ProfileSkeleton />
      </div>
    )
  }

  if (error) {
    return (
      <div className="mx-auto max-w-[1200px] px-6 py-12">
        <Crumb name="Player" />
        <div
          className="cc-card p-6"
          style={{ borderColor: 'color-mix(in srgb, var(--color-clay) 30%, var(--color-line))' }}
        >
          <p className="text-sm text-clay">{error}</p>
          <Link href="/players" className="mt-3 inline-block text-sm text-court hover:underline">
            ← Back to roster
          </Link>
        </div>
      </div>
    )
  }

  if (!player) {
    return (
      <div className="mx-auto max-w-[1200px] px-6 py-12">
        <Crumb name="Not found" />
        <div className="cc-card flex flex-col items-start gap-3 p-8">
          <p className="font-display text-2xl text-ink">Player not found.</p>
          <p className="text-sm text-ink-soft">
            This player may have been removed, or the link is stale.
          </p>
          <Link href="/players" className="mt-2 text-sm text-court hover:underline">
            ← Back to roster
          </Link>
        </div>
      </div>
    )
  }

  const { lead, tail } = splitName(player.name)
  const meta = [player.year, player.position, 'UC Davis Tennis'].filter(Boolean)

  return (
    <div className="mx-auto max-w-[1200px] px-6 py-12">
      {/* Breadcrumb */}
      <Crumb name={player.name} />

      {/* PROFILE HEAD */}
      <section className="grid items-center gap-9 pb-8 pt-6 md:grid-cols-[auto_1fr_auto]">
        {player.photo_url && !photoFailed ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={playerPhotoProxyUrl(player.photo_url) ?? undefined}
            alt=""
            className="size-36 shrink-0 rounded-full object-cover object-top"
            onError={() => setPhotoFailed(true)}
          />
        ) : (
          <div
            className="flex size-36 shrink-0 items-center justify-center rounded-full font-display text-[3.4rem] font-medium tracking-[-0.02em] text-cream"
            style={{
              background:
                'linear-gradient(140deg, var(--color-court), var(--color-court-deep))',
              boxShadow:
                '0 1px 0 rgba(26,24,21,0.06), 0 24px 48px -24px rgba(46,83,65,0.45)',
            }}
          >
            {initialsOf(player.name) || '?'}
          </div>
        )}

        <div className="flex flex-col gap-3">
          <Display as="h1" size="lg">
            {lead}
            {tail && (
              <>
                {' '}
                <em>{tail}</em>
              </>
            )}
          </Display>
          <div className="flex flex-wrap gap-3 text-base text-ink-soft">
            {meta.map((m, i) => (
              <span key={`${m}-${i}`} className="flex items-center gap-3">
                {i > 0 && <span className="text-ink-mute">·</span>}
                <span className={i === 0 ? 'font-medium text-ink' : ''}>{m}</span>
              </span>
            ))}
          </div>
          <HandednessControl
            playerId={player.id}
            value={(player.handedness as Handedness | undefined) ?? 'right'}
            onChange={(next) =>
              setPlayer((prev) => (prev ? { ...prev, handedness: next } : prev))
            }
          />
        </div>

        <div className="flex flex-wrap gap-2.5">
          <Button variant="ghost" size="sm" asChild>
            <Link href="/recordings">All recordings</Link>
          </Button>
          {recordings[0] && (
            <Button variant="ink" size="sm" asChild>
              <Link href={`/recordings/${recordings[0].id}`}>Open latest</Link>
            </Button>
          )}
        </div>
      </section>

      {/* STAT STRIP */}
      <section className="mb-9 grid grid-cols-2 gap-5 sm:grid-cols-3 lg:grid-cols-5">
        <StatCard label="Recordings" value={recordings.length} unit="" />
        <StatCard
          label="Total shots"
          value={totals.shots}
          unit=""
          format={(n) => n.toLocaleString()}
        />
        <StatCard
          label="Bounce accuracy"
          value={totals.accuracyPct ?? 0}
          unit={totals.accuracyPct == null ? null : '%'}
          missing={totals.accuracyPct == null}
        />
        <StatCard
          label="Rallies"
          value={totals.rallies}
          unit=""
          format={(n) => n.toLocaleString()}
        />
        <StatCard
          label="Hours recorded"
          value={Math.round(totals.hours * 10)}
          unit="h"
          format={(n) => (n / 10).toFixed(1)}
          missing={totals.hours <= 0}
        />
      </section>

      {/* STROKE BREAKDOWN */}
      <section className="mb-10">
        <header className="mb-6 flex flex-wrap items-end justify-between gap-4">
          <div>
            <Eyebrow>Stroke breakdown</Eyebrow>
            <Display as="h2" size="md" className="mt-3">
              How <em>{player.name.split(/\s+/)[0]}</em> hits.
            </Display>
          </div>
          <p className="max-w-[38ch] text-sm text-ink-mute">
            Counts and share of total tracked shots across every recording.
          </p>
        </header>

        <div className="cc-card p-7" data-countup-card>
          <StrokeBars rows={strokeRows} />
        </div>
      </section>

      {/* RECENT TREND */}
      {last5.length > 0 && (
        <section className="mb-10">
          <header className="mb-6 flex flex-wrap items-end justify-between gap-4">
            <div>
              <Eyebrow>Recent trend</Eyebrow>
              <Display as="h2" size="md" className="mt-3">
                Last <em>{last5.length} recording{last5.length === 1 ? '' : 's'}</em>.
              </Display>
            </div>
            <p className="max-w-[38ch] text-sm text-ink-mute">
              In-bounds rate per recording. Higher = cleaner shot selection.
            </p>
          </header>

          <div className="cc-card p-7">
            <RecentTrendBars recordings={last5} />
          </div>
        </section>
      )}

      {/* MATCH LIST */}
      <section className="mb-10">
        <header className="mb-6 flex flex-wrap items-end justify-between gap-4">
          <div>
            <Eyebrow>Recording history</Eyebrow>
            <Display as="h2" size="md" className="mt-3">
              Every recording, <em>scrubbable</em>.
            </Display>
          </div>
        </header>

        {recordings.length === 0 ? (
          <div className="cc-card flex flex-col items-center gap-3 p-12 text-center">
            <p className="font-display text-xl text-ink">No recordings yet.</p>
            <p className="text-sm text-ink-soft">
              Upload match footage and assign it to {player.name.split(/\s+/)[0]} to see it here.
            </p>
            <Button variant="ink" size="sm" asChild>
              <Link href="/upload">Upload video</Link>
            </Button>
          </div>
        ) : (
          <div className="cc-card overflow-hidden p-0">
            <div className="grid grid-cols-[100px_1fr_120px_120px_60px] items-center gap-4 border-b border-line-soft bg-shade px-6 py-3.5 font-mono text-[0.66rem] uppercase tracking-[0.14em] text-ink-mute dark:bg-surface">
              <span>Date</span>
              <span>Recording</span>
              <span>Shots</span>
              <span>Accuracy</span>
              <span />
            </div>
            {recordings.map((r) => (
              <Link
                key={r.id}
                href={`/recordings/${r.id}`}
                className="cc-match-row block border-b border-line-soft last:border-b-0"
              >
                <div className="grid grid-cols-[100px_1fr_120px_120px_60px] items-center gap-4 px-6 py-4">
                  <span className="font-mono text-[0.78rem] tabular-nums text-ink-mute">
                    {formatDateShort(r.createdAt)}
                  </span>
                  <div className="min-w-0">
                    <div className="truncate font-display text-[1.05rem] font-medium tracking-[-0.01em] text-ink">
                      {r.name}
                    </div>
                    <div className="text-[0.85rem] text-ink-mute">
                      {r.status === 'done'
                        ? r.shotCount != null
                          ? `${r.shotCount} shots · ${r.bounceCount ?? 0} bounces`
                          : 'Processed'
                        : r.status === 'processing'
                          ? 'Processing'
                          : r.status === 'failed'
                            ? 'Failed'
                            : 'Pending'}
                    </div>
                  </div>
                  <span className="font-display text-[1rem] font-medium tabular-nums text-ink">
                    {r.shotCount != null ? r.shotCount.toLocaleString() : '—'}
                  </span>
                  <span className="font-display text-[1rem] font-medium tabular-nums text-ink">
                    {(() => {
                      const a = inPct(r.inBoundsBounces, r.outBoundsBounces)
                      return a == null ? (
                        <span className="text-ink-mute">—</span>
                      ) : (
                        <>
                          {a}
                          <span className="ml-0.5 text-[0.62em] text-ink-mute">%</span>
                        </>
                      )
                    })()}
                  </span>
                  <span className="cc-arrow text-right text-ink-mute">→</span>
                </div>
              </Link>
            ))}
          </div>
        )}
      </section>
    </div>
  )
}

function Crumb({ name }: { name: string }) {
  return (
    <nav className="flex items-center gap-2.5 pt-1 font-mono text-[0.72rem] uppercase tracking-[0.14em] text-ink-mute">
      <Link href="/players" className="hover:text-ink">
        Players
      </Link>
      <span className="opacity-50">/</span>
      <span className="text-ink">{name}</span>
    </nav>
  )
}

function StatCard({
  label,
  value,
  unit,
  missing,
  format,
}: {
  label: string
  value: number
  unit: string | null
  missing?: boolean
  format?: (n: number) => string
}) {
  return (
    <div className="cc-stat-tile" data-countup-card>
      <div className="font-mono text-[0.66rem] uppercase tracking-[0.14em] text-ink-mute">
        {label}
      </div>
      <div className="mt-2 font-display text-[2.4rem] font-medium leading-none tabular-nums tracking-[-0.018em] text-ink">
        {missing ? (
          <span className="text-ink-mute">—</span>
        ) : (
          <>
            <CountUp to={value} format={format} />
            {unit && (
              <span className="ml-0.5 text-[0.5em] text-ink-mute">{unit}</span>
            )}
          </>
        )}
      </div>
    </div>
  )
}

function RecentTrendBars({ recordings }: { recordings: ApiRecording[] }) {
  // Reverse so chronological left → right.
  const rows = recordings.slice().reverse()
  return (
    <div className="flex items-end gap-3 h-44">
      {rows.map((r) => {
        const acc = inPct(r.inBoundsBounces, r.outBoundsBounces)
        const height = acc == null ? 4 : Math.max(6, acc) // % maps directly to height %
        return (
          <Link
            key={r.id}
            href={`/recordings/${r.id}`}
            className="group flex flex-1 flex-col items-center justify-end gap-2"
          >
            <div className="relative flex w-full flex-1 items-end">
              <div
                className="w-full rounded-t-md transition-[height,background] duration-[480ms] ease-[cubic-bezier(0.2,0.8,0.2,1)]"
                style={{
                  height: `${height}%`,
                  background:
                    'color-mix(in srgb, var(--color-court) 70%, transparent)',
                }}
              />
              <div className="pointer-events-none absolute -top-7 left-1/2 -translate-x-1/2 whitespace-nowrap rounded-md border border-line bg-paper px-2 py-1 text-[0.7rem] font-medium text-ink opacity-0 shadow-sm transition-opacity group-hover:opacity-100">
                {acc == null ? '—' : `${acc}%`}
              </div>
            </div>
            <span className="font-mono text-[0.62rem] uppercase tracking-[0.12em] text-ink-mute">
              {formatDateShort(r.createdAt)}
            </span>
          </Link>
        )
      })}
    </div>
  )
}

/**
 * Right/Left segmented control. Optimistic update + PATCH /api/players/[id].
 * Why this matters: the stroke classifier was trained on right-handed canonical
 * poses. For lefties the pipeline mirrors x-axis + swaps L/R keypoints in
 * extract_pose_sequence so the model sees a righty-equivalent pose. Without
 * this flag a lefty's forehands silently get mislabeled as backhands.
 */
function HandednessControl({
  playerId,
  value,
  onChange,
}: {
  playerId: string
  value: Handedness
  onChange: (next: Handedness) => void
}) {
  const [saving, setSaving] = useState<Handedness | null>(null)
  const [error, setError] = useState<string | null>(null)

  const update = async (next: Handedness) => {
    if (next === value) return
    setSaving(next)
    setError(null)
    onChange(next) // optimistic
    try {
      const res = await fetch(`/api/players/${playerId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ handedness: next }),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        setError(body?.error || 'Failed to save')
        onChange(value) // rollback
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save')
      onChange(value) // rollback
    } finally {
      setSaving(null)
    }
  }

  return (
    <div className="flex items-center gap-3">
      <span className="font-mono text-[0.66rem] uppercase tracking-[0.14em] text-ink-mute">
        Handedness
      </span>
      <div
        role="radiogroup"
        aria-label="Player handedness"
        className="inline-flex gap-0.5 rounded-full border border-line bg-paper p-0.5"
      >
        {(['right', 'left'] as Handedness[]).map((opt) => {
          const active = value === opt
          const isSaving = saving === opt
          return (
            <button
              key={opt}
              type="button"
              role="radio"
              aria-checked={active}
              disabled={isSaving}
              onClick={() => update(opt)}
              className={`rounded-full px-3.5 py-1 text-[0.82rem] font-medium transition-colors duration-150 ${
                active
                  ? 'bg-ink text-cream dark:bg-court-deep'
                  : 'text-ink-soft hover:text-ink'
              } ${isSaving ? 'opacity-60' : ''}`}
            >
              {opt === 'right' ? 'Right-handed' : 'Left-handed'}
            </button>
          )
        })}
      </div>
      {error && <span className="text-[0.78rem] text-clay">{error}</span>}
    </div>
  )
}

function ProfileSkeleton() {
  return (
    <div aria-hidden>
      <div className="h-3 w-32 animate-pulse rounded bg-shade" />
      <div className="mt-6 grid grid-cols-[auto_1fr] items-center gap-9">
        <div className="size-36 animate-pulse rounded-full bg-shade" />
        <div className="flex flex-col gap-3">
          <div className="h-12 w-2/3 animate-pulse rounded bg-shade" />
          <div className="h-4 w-1/2 animate-pulse rounded bg-shade" />
        </div>
      </div>
      <div className="mt-10 grid grid-cols-2 gap-5 sm:grid-cols-5">
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className="h-24 animate-pulse rounded-[14px] bg-paper/60 border border-line" />
        ))}
      </div>
      <div className="mt-10 h-72 animate-pulse rounded-[14px] border border-line bg-paper/60" />
    </div>
  )
}
