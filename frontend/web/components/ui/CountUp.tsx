'use client';

import { useEffect, useRef, useState } from 'react';

type Props = {
  /** Number of decimal places to render. Defaults to 0. */
  decimals?: number;
  /** Animation duration in ms. */
  duration?: number;
  durationMs?: number;
  /** If provided, the caller controls playback instead of scroll visibility. */
  play?: boolean;
  format?: (n: number) => string;
  /** Optional suffix rendered after the number (NOT animated). */
  suffix?: React.ReactNode;
  className?: string;
} & ({ to: number; value?: never } | { value: number; to?: never });

/**
 * Count-up number that replays whenever its containing card scrolls in/out of view.
 * Mirrors the IntersectionObserver pattern from docs/brand-drop/mocks/visuals.html.
 *
 * Observes the nearest card (falls back to self) at threshold 0.35, or follows
 * the caller's `play` flag. Respects prefers-reduced-motion in both modes.
 */
export default function CountUp({
  to,
  value,
  decimals = 0,
  duration,
  durationMs,
  play,
  format,
  suffix,
  className,
}: Props) {
  const target = to ?? value ?? 0;
  const animationDuration = durationMs ?? duration ?? (play === undefined ? 720 : 760);
  const spanRef = useRef<HTMLSpanElement | null>(null);
  const rafRef = useRef<number | null>(null);
  const isInViewRef = useRef(false);
  const [text, setText] = useState<string>(() => format ? format(0) : (0).toFixed(decimals));

  useEffect(() => {
    const el = spanRef.current;
    if (!el) return;

    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    const formatValue = format ?? ((v: number) => v.toFixed(decimals));

    const runCount = () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      if (reduce) {
        setText(formatValue(target));
        return;
      }
      const start = performance.now();
      const step = (now: number) => {
        const t = animationDuration <= 0 ? 1 : Math.min(1, (now - start) / animationDuration);
        // ease-out-quart
        const eased = 1 - Math.pow(1 - t, 4);
        const v = target * eased;
        setText(formatValue(format ? Math.round(v) : v));
        if (t < 1) {
          rafRef.current = requestAnimationFrame(step);
        } else {
          rafRef.current = null;
          setText(formatValue(target));
        }
      };
      rafRef.current = requestAnimationFrame(step);
    };

    const reset = () => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      setText(formatValue(0));
    };

    if (play !== undefined) {
      if (play) runCount();
      else reset();
      return () => {
        if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      };
    }

    // Find nearest card-like container so trigger fires when the whole card enters view.
    const card =
      el.closest('.cc-card, .cc-stat-tile, .player-card, .cc-insight, [data-count-card], [data-countup-card]') ||
      el;

    isInViewRef.current = false;
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting && !isInViewRef.current) {
            isInViewRef.current = true;
            runCount();
          } else if (!entry.isIntersecting && isInViewRef.current) {
            isInViewRef.current = false;
            reset();
          }
        });
      },
      { threshold: 0.35 }
    );

    io.observe(card);

    return () => {
      io.disconnect();
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [target, decimals, animationDuration, play, format]);

  return (
    <span ref={spanRef} className={className} data-count-to={target}>
      {text}
      {suffix}
    </span>
  );
}

export { CountUp };
