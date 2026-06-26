'use client'

import Link from 'next/link'
import { useState } from 'react'
import { CountUp } from './CountUp'
import { playerPhotoProxyUrl } from '@/lib/utils'

export interface PlayerCardData {
  id: string
  name: string
  year: string | null
  position: string | null
  photo_url: string | null
  matchCount: number
  hoursRecorded: number | null
  // Stat row (percentages, 0-100). null = no data.
  baselinePct: number | null
  netPct: number | null
  firstServePct: number | null
  // Last 3 recordings (oldest → newest is fine; we display top-down).
  recent: {
    id: string
    label: string // "vs OPP · School" or just recording name
    date: string // ISO
    result: 'W' | 'L' | null
  }[]
}

// Stable gradient per player id, picked from the brand palette pairs used in the mock.
const AVATAR_GRADIENTS: Array<[string, string]> = [
  ['var(--color-court)', 'var(--color-court-deep)'],
  ['var(--color-clay)', 'var(--color-clay-soft)'],
  ['var(--color-plum)', '#5C3852'],
  ['var(--color-slate)', '#3E4A56'],
  ['var(--color-amber)', '#9A6E20'],
  ['var(--color-court-light)', 'var(--color-court)'],
]

function gradientFor(id: string): [string, string] {
  // Tiny deterministic hash so the same player always gets the same gradient.
  let h = 0
  for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) >>> 0
  return AVATAR_GRADIENTS[h % AVATAR_GRADIENTS.length]
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

function metaLine(p: PlayerCardData): string {
  const bits = [p.year, p.position].filter(Boolean)
  return bits.length ? bits.join(' · ') : 'UC Davis Tennis'
}

export function PlayerCard({ player }: { player: PlayerCardData }) {
  const [g1, g2] = gradientFor(player.id)
  const [photoFailed, setPhotoFailed] = useState(false)
  const { lead, tail } = splitName(player.name)

  const photoSrc = photoFailed ? null : playerPhotoProxyUrl(player.photo_url)

  return (
    <Link href={`/players/${player.id}`} className="group block h-full">
      <div
        data-countup-card
        className="cc-card flex h-full cursor-pointer flex-col gap-[18px] p-7"
      >
        {/* Header */}
        <div className="flex items-center gap-4">
          {photoSrc ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={photoSrc}
              alt=""
              className="size-14 shrink-0 rounded-full object-cover object-top"
              onError={() => setPhotoFailed(true)}
            />
          ) : (
            <div
              className="flex size-14 shrink-0 items-center justify-center rounded-full font-display text-[1.45rem] font-medium tracking-[-0.012em] text-cream"
              style={{ background: `linear-gradient(140deg, ${g1}, ${g2})` }}
            >
              {initialsOf(player.name) || '?'}
            </div>
          )}
          <div className="flex min-w-0 flex-col gap-0.5">
            <span className="font-display text-[1.4rem] font-medium leading-[1.05] tracking-[-0.014em] text-ink">
              {lead}
              {tail && (
                <>
                  {' '}
                  <em>{tail}</em>
                </>
              )}
            </span>
            <span className="truncate text-[0.85rem] text-ink-soft">
              {metaLine(player)}
            </span>
          </div>
        </div>

        {/* Stats row */}
        <div className="grid grid-cols-3 gap-2 rounded-[10px] bg-shade px-4 py-3.5 dark:bg-surface">
          <Stat label="Baseline" pct={player.baselinePct} />
          <Stat label="Net win" pct={player.netPct} />
          <Stat label="1st serve" pct={player.firstServePct} />
        </div>

        {/* Recent recordings */}
        <div className="flex flex-col gap-1.5">
          <span className="font-mono text-[0.66rem] uppercase tracking-[0.14em] text-ink-mute">
            {player.recent.length > 0
              ? `Last ${player.recent.length} recording${player.recent.length === 1 ? '' : 's'}`
              : 'No recordings yet'}
          </span>
          {player.recent.length === 0 ? (
            <span className="text-[0.85rem] text-ink-mute">
              Upload a video to start.
            </span>
          ) : (
            player.recent.map((r) => (
              <div
                key={r.id}
                className="grid grid-cols-[auto_1fr_auto] items-center gap-3 text-[0.88rem]"
              >
                <span className="font-mono text-[0.72rem] tabular-nums text-ink-mute">
                  {formatDateShort(r.date)}
                </span>
                <span className="truncate text-ink-soft">{r.label}</span>
                <ResultPill result={r.result} />
              </div>
            ))
          )}
        </div>

        {/* Foot */}
        <div className="mt-auto flex items-center justify-between border-t border-line-soft pt-3.5">
          <span className="font-mono text-[0.7rem] uppercase tracking-[0.1em] text-ink-mute">
            {player.hoursRecorded != null
              ? `${player.hoursRecorded.toFixed(1)} hrs recorded`
              : `${player.matchCount} recording${player.matchCount === 1 ? '' : 's'}`}
          </span>
          <span className="cc-arrow text-ink-mute group-hover:text-ink">
            →
          </span>
        </div>
      </div>
    </Link>
  )
}

function Stat({ label, pct }: { label: string; pct: number | null }) {
  return (
    <div className="flex flex-col gap-px">
      <span className="font-display text-[1.3rem] font-medium leading-[1.05] tabular-nums tracking-[-0.012em] text-ink">
        {pct == null ? (
          <span className="text-ink-mute">—</span>
        ) : (
          <>
            <CountUp to={pct} />
            <span className="ml-0.5 text-[0.62em] text-ink-mute">%</span>
          </>
        )}
      </span>
      <span className="font-mono text-[0.6rem] uppercase tracking-[0.1em] text-ink-mute">
        {label}
      </span>
    </div>
  )
}

function ResultPill({ result }: { result: 'W' | 'L' | null }) {
  if (!result) {
    return (
      <span className="font-mono text-[0.66rem] uppercase tracking-[0.12em] text-ink-mute">
        —
      </span>
    )
  }
  const isWin = result === 'W'
  return (
    <span
      className="rounded-full px-2 py-0.5 font-mono text-[0.66rem] font-semibold uppercase tracking-[0.12em]"
      style={
        isWin
          ? {
              background: 'color-mix(in srgb, var(--color-court) 10%, transparent)',
              color: 'var(--color-court)',
            }
          : {
              background: 'color-mix(in srgb, var(--color-clay) 10%, transparent)',
              color: 'var(--color-clay)',
            }
      }
    >
      {result}
    </span>
  )
}
