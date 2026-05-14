'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { useAuth } from '@/contexts/AuthContext'

type Summary = {
  totals: { done: number; total: number }
  tennisStats: {
    totalShots: number
    totalBounces: number
    totalRallies: number
    totalForehands: number
    totalBackhands: number
    totalServes: number
    totalInBounds: number
    totalOutBounds: number
  }
  hasTennisStats: boolean
}

const STROKES = [
  { key: 'forehand', label: 'Forehand', tokenColor: 'var(--color-court)' },
  { key: 'backhand', label: 'Backhand', tokenColor: 'var(--color-plum)' },
  { key: 'serve', label: 'Serve / smash', tokenColor: 'var(--color-amber)' },
] as const

export default function ProfilePage() {
  const { user } = useAuth()
  const [summary, setSummary] = useState<Summary | null>(null)
  const [loading, setLoading] = useState(true)
  const [photoFailed, setPhotoFailed] = useState(false)

  useEffect(() => {
    let cancelled = false
    fetch('/api/dashboard/summary')
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (!cancelled) setSummary(d ?? null)
      })
      .catch(() => {})
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  const displayName = user?.user_metadata?.name || user?.email?.split('@')[0] || 'Player'
  const email = user?.email || ''
  const role = user?.user_metadata?.role || 'Coach'
  const avatarUrl: string | undefined = user?.user_metadata?.avatar_url
  const joined = user?.created_at
    ? new Date(user.created_at).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
      })
    : null
  const initials = displayName
    .split(' ')
    .filter(Boolean)
    .slice(0, 2)
    .map((p: string) => p[0]?.toUpperCase())
    .join('')

  const ts = summary?.tennisStats
  const totalStrokes = ts ? ts.totalForehands + ts.totalBackhands + ts.totalServes : 0
  const totalBounced = ts ? ts.totalInBounds + ts.totalOutBounds : 0
  const inPct = totalBounced > 0 ? Math.round((ts!.totalInBounds / totalBounced) * 100) : 0

  const stats = [
    { label: 'Sessions analyzed', value: summary?.totals?.done ?? 0 },
    { label: 'Total shots', value: ts?.totalShots ?? 0 },
    { label: 'Total bounces', value: ts?.totalBounces ?? 0 },
    { label: 'Total rallies', value: ts?.totalRallies ?? 0 },
  ]

  const photoSrc =
    avatarUrl && !photoFailed
      ? `/api/proxy-image?url=${encodeURIComponent(avatarUrl)}`
      : null

  return (
    <div className="container mx-auto max-w-[960px] px-6 md:px-14 py-10">
      {/* Header */}
      <section className="pt-6 pb-7">
        <span
          aria-hidden
          className="inline-flex items-center gap-2 font-mono uppercase tracking-[0.18em] text-[0.72rem] text-court dark:text-court-light"
        >
          <span className="w-1.5 h-1.5 rounded-full bg-clay dark:bg-clay-soft" />
          Profile
        </span>
        <h1
          className="text-ink mt-4"
          style={{
            fontFamily: 'var(--font-display)',
            fontWeight: 500,
            letterSpacing: '-0.022em',
            lineHeight: 1.15,
            fontSize: 'clamp(40px, 4.8vw, 64px)',
            paddingTop: '0.08em',
          }}
        >
          Your <em>profile</em>.
        </h1>
      </section>

      {/* Identity card */}
      <section className="cc-card flex flex-col sm:flex-row items-start sm:items-center gap-6 p-7 mb-6">
        <div
          className="w-20 h-20 rounded-full overflow-hidden flex items-center justify-center text-cream flex-shrink-0"
          style={{
            background: 'var(--color-court)',
            fontFamily: 'var(--font-display)',
            fontWeight: 500,
            fontSize: '1.85rem',
            letterSpacing: '-0.012em',
          }}
        >
          {photoSrc ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={photoSrc}
              alt=""
              className="w-full h-full object-cover"
              onError={() => setPhotoFailed(true)}
            />
          ) : (
            initials || 'P'
          )}
        </div>
        <div className="flex-1 min-w-0">
          <h2
            className="text-ink"
            style={{
              fontFamily: 'var(--font-display)',
              fontWeight: 500,
              fontSize: '2rem',
              letterSpacing: '-0.014em',
              lineHeight: 1.25,
              paddingTop: '0.1em',
            }}
          >
            {displayName}
          </h2>
          <p className="text-ink-soft font-mono text-[0.78rem] mt-1 truncate">{email}</p>
          <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-ink-mute font-mono uppercase tracking-[0.14em] text-[0.66rem]">
            <span className="text-court dark:text-court-light">{role}</span>
            {joined && (
              <>
                <span aria-hidden>·</span>
                <span>Joined {joined}</span>
              </>
            )}
          </div>
        </div>
        <Link
          href="/settings"
          className="self-stretch sm:self-auto inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-full border border-line text-ink-soft hover:border-ink hover:text-ink transition-colors font-medium text-sm"
        >
          Edit profile
          <span aria-hidden>→</span>
        </Link>
      </section>

      {/* Career stats strip */}
      <section
        className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6"
        aria-label="Career stats"
      >
        {stats.map((s) => (
          <div key={s.label} className="cc-stat-tile">
            <div className="font-mono uppercase tracking-[0.14em] text-[0.7rem] text-ink-mute mb-3">
              {s.label}
            </div>
            <div
              className="text-ink"
              style={{
                fontFamily: 'var(--font-display)',
                fontWeight: 500,
                fontFeatureSettings: "'tnum'",
                fontSize: 'clamp(32px, 3.4vw, 44px)',
                lineHeight: 1.0,
                letterSpacing: '-0.02em',
              }}
            >
              {loading ? '·' : s.value.toLocaleString()}
            </div>
          </div>
        ))}
      </section>

      {/* Stroke breakdown */}
      {!loading && summary?.hasTennisStats && totalStrokes > 0 && (
        <section className="cc-card p-7 mb-6">
          <h3 className="font-mono uppercase tracking-[0.16em] text-[0.7rem] text-court dark:text-court-light mb-5">
            Stroke breakdown
          </h3>
          <div className="flex flex-col gap-4">
            {STROKES.map((s) => {
              const count =
                s.key === 'forehand'
                  ? ts!.totalForehands
                  : s.key === 'backhand'
                    ? ts!.totalBackhands
                    : ts!.totalServes
              const pct = totalStrokes > 0 ? Math.round((count / totalStrokes) * 100) : 0
              return (
                <div key={s.key} className="flex flex-col gap-1.5">
                  <div className="flex items-center justify-between text-sm">
                    <span className="inline-flex items-center gap-2 text-ink-soft">
                      <span
                        aria-hidden
                        className="w-2 h-2 rounded-sm"
                        style={{ background: s.tokenColor }}
                      />
                      {s.label}
                    </span>
                    <span
                      className="text-ink font-medium"
                      style={{
                        fontFamily: 'var(--font-display)',
                        fontFeatureSettings: "'tnum'",
                      }}
                    >
                      {count.toLocaleString()} <span className="text-ink-mute">({pct}%)</span>
                    </span>
                  </div>
                  <div className="h-1.5 rounded-full bg-shade overflow-hidden">
                    <div
                      className="h-full rounded-full transition-[width]"
                      style={{
                        width: `${pct}%`,
                        background: s.tokenColor,
                        transitionDuration: '760ms',
                        transitionTimingFunction: 'cubic-bezier(0.165, 0.84, 0.44, 1)',
                      }}
                    />
                  </div>
                </div>
              )
            })}
          </div>
        </section>
      )}

      {/* Court accuracy */}
      {!loading && totalBounced > 0 && (
        <section className="cc-card p-7 mb-6">
          <h3 className="font-mono uppercase tracking-[0.16em] text-[0.7rem] text-court dark:text-court-light mb-5">
            Court accuracy
          </h3>
          <div className="flex items-center gap-7 flex-wrap">
            <div className="shrink-0 w-24 h-24 relative flex items-center justify-center">
              <svg viewBox="0 0 36 36" className="w-24 h-24 -rotate-90">
                <circle
                  cx="18"
                  cy="18"
                  r="15.9"
                  fill="none"
                  stroke="var(--color-line)"
                  strokeWidth="2.6"
                />
                <circle
                  cx="18"
                  cy="18"
                  r="15.9"
                  fill="none"
                  stroke="var(--color-court)"
                  strokeWidth="2.6"
                  strokeDasharray={`${inPct} ${100 - inPct}`}
                  strokeLinecap="round"
                />
              </svg>
              <span
                className="absolute text-ink"
                style={{
                  fontFamily: 'var(--font-display)',
                  fontWeight: 500,
                  fontSize: '1.15rem',
                  fontFeatureSettings: "'tnum'",
                }}
              >
                {inPct}%
              </span>
            </div>
            <div className="min-w-0">
              <p className="font-mono uppercase tracking-[0.14em] text-[0.66rem] text-ink-mute mb-1">
                In-bounds bounces
              </p>
              <p
                className="text-ink"
                style={{
                  fontFamily: 'var(--font-display)',
                  fontWeight: 500,
                  fontSize: 'clamp(28px, 3vw, 40px)',
                  letterSpacing: '-0.018em',
                  lineHeight: 1.0,
                  fontFeatureSettings: "'tnum'",
                }}
              >
                {ts!.totalInBounds.toLocaleString()}
              </p>
              <p className="text-ink-mute text-sm mt-1">
                of {totalBounced.toLocaleString()} tracked bounces
              </p>
            </div>
          </div>
        </section>
      )}

      {/* Quick actions */}
      <section
        className="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-2"
        aria-label="Quick actions"
      >
        <Link
          href="/upload"
          className="inline-flex items-center justify-center gap-2 py-3 px-4 rounded-full bg-court text-cream font-medium text-sm hover:-translate-y-px transition-transform dark:bg-court-deep dark:hover:bg-court"
        >
          New recording
          <span aria-hidden>→</span>
        </Link>
        <Link
          href="/recordings"
          className="inline-flex items-center justify-center gap-2 py-3 px-4 rounded-full border border-line text-ink-soft hover:border-ink hover:text-ink transition-colors font-medium text-sm"
        >
          View recordings
        </Link>
        <Link
          href="/settings"
          className="inline-flex items-center justify-center gap-2 py-3 px-4 rounded-full border border-line text-ink-soft hover:border-ink hover:text-ink transition-colors font-medium text-sm"
        >
          Settings
        </Link>
      </section>
    </div>
  )
}
