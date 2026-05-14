'use client'

import { useEffect, useRef, useState } from 'react'

/**
 * Horizontal stroke bars (FH / BH / Serve / Volley).
 * Adapted from the buildShotBars pattern in
 * docs/brand-drop/mocks/visuals.html (stroke breakdown section).
 *
 * Brand stroke colors (LOCKED):
 *   forehand = court
 *   backhand = plum
 *   serve    = amber
 *   volley   = clay
 */
export interface StrokeRow {
  key: 'forehand' | 'backhand' | 'serve' | 'volley'
  label: string
  count: number
}

const STROKE_COLOR: Record<StrokeRow['key'], string> = {
  forehand: 'var(--color-stroke-forehand)',
  backhand: 'var(--color-stroke-backhand)',
  serve: 'var(--color-stroke-serve)',
  volley: 'var(--color-stroke-volley)',
}

export function StrokeBars({ rows }: { rows: StrokeRow[] }) {
  const totalShots = rows.reduce((s, r) => s + r.count, 0)
  const maxShare =
    Math.max(1, ...rows.map((r) => (totalShots > 0 ? r.count / totalShots : 0))) || 1

  return (
    <div className="flex flex-col gap-4">
      {rows.map((row) => {
        const share = totalShots > 0 ? row.count / totalShots : 0
        // Bar fills to the SHARE of total shots (so dominant stroke = longest
        // bar). Accuracy is shown as a label on the right.
        const widthPct = (share / maxShare) * 100
        return (
          <StrokeBarRow
            key={row.key}
            row={row}
            widthPct={widthPct}
            share={share}
          />
        )
      })}
      {totalShots === 0 && (
        <p className="text-sm text-ink-mute">
          No stroke data yet. Process a recording to populate this.
        </p>
      )}
    </div>
  )
}

function StrokeBarRow({
  row,
  widthPct,
  share,
}: {
  row: StrokeRow
  widthPct: number
  share: number
}) {
  const ref = useRef<HTMLDivElement>(null)
  const [drawn, setDrawn] = useState(false)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    if (reduce) {
      setDrawn(true)
      return
    }
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) setDrawn(true)
          else setDrawn(false)
        })
      },
      { threshold: 0.4 },
    )
    io.observe(el)
    return () => io.disconnect()
  }, [])

  return (
    <div ref={ref} className="flex flex-col gap-1.5">
      <div className="flex items-baseline justify-between gap-3">
        <span className="font-display text-[1.05rem] font-medium tracking-[-0.012em] text-ink">
          {row.label}
        </span>
        <div className="flex items-baseline gap-3 font-mono text-[0.72rem] tabular-nums text-ink-mute">
          <span>{row.count.toLocaleString()} shots</span>
          <span className="text-ink-soft">
            {Math.round(share * 100)}%
          </span>
        </div>
      </div>
      <div
        className="h-2 w-full overflow-hidden rounded-full"
        style={{ background: 'color-mix(in srgb, var(--color-ink) 7%, transparent)' }}
      >
        <div
          className="h-full rounded-full transition-[width] duration-[720ms] ease-[cubic-bezier(0.2,0.8,0.2,1)]"
          style={{
            width: drawn ? `${Math.max(2, widthPct)}%` : '0%',
            background: STROKE_COLOR[row.key],
          }}
        />
      </div>
    </div>
  )
}
