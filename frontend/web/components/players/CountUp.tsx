'use client'

import { useEffect, useRef } from 'react'

/**
 * Scroll-triggered count-up number.
 * Ported from docs/brand-drop/mocks/visuals.html lines 1264-1301
 * (the [data-count-to] IntersectionObserver pattern).
 *
 * Counts from 0 to `to` over `duration`ms when its nearest card-ish ancestor
 * enters the viewport. Resets silently when it leaves so re-entry replays.
 * Respects prefers-reduced-motion.
 */
export function CountUp({
  to,
  duration = 720,
  className,
  format,
}: {
  to: number
  duration?: number
  className?: string
  format?: (n: number) => string
}) {
  const ref = useRef<HTMLSpanElement>(null)

  useEffect(() => {
    const el = ref.current
    if (!el) return

    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const fmt = format ?? ((n: number) => String(n))
    let rafId: number | null = null
    let isInView = false

    const runCount = () => {
      if (rafId) cancelAnimationFrame(rafId)
      if (reduce) {
        el.textContent = fmt(to)
        return
      }
      const start = performance.now()
      const step = (now: number) => {
        const t = Math.min(1, (now - start) / duration)
        el.textContent = fmt(Math.round(to * t))
        if (t < 1) rafId = requestAnimationFrame(step)
        else {
          rafId = null
          el.textContent = fmt(to)
        }
      }
      rafId = requestAnimationFrame(step)
    }

    const reset = () => {
      if (rafId) {
        cancelAnimationFrame(rafId)
        rafId = null
      }
      el.textContent = fmt(0)
    }

    // Observe the nearest card-ish ancestor so the trigger fires when the
    // whole card enters view, not just the number itself (would fire too
    // late on tall cards). Falls back to the span itself.
    const card =
      el.closest('[data-countup-card]') ??
      el.closest('.cc-card') ??
      el.closest('.cc-stat-tile') ??
      el

    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting && !isInView) {
            isInView = true
            runCount()
          } else if (!entry.isIntersecting && isInView) {
            isInView = false
            reset()
          }
        })
      },
      { threshold: 0.35 },
    )

    io.observe(card)

    return () => {
      io.disconnect()
      if (rafId) cancelAnimationFrame(rafId)
    }
  }, [to, duration, format])

  return (
    <span ref={ref} className={className}>
      0
    </span>
  )
}
