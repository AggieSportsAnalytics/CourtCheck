'use client';

import { useEffect, useRef, useState } from 'react';

type Props = {
  to: number;
  /** Number of decimal places to render. Defaults to 0. */
  decimals?: number;
  /** Animation duration in ms. */
  duration?: number;
  /** Optional suffix rendered after the number (NOT animated). */
  suffix?: React.ReactNode;
  className?: string;
};

/**
 * Count-up number that replays whenever its containing card scrolls in/out of view.
 * Mirrors the IntersectionObserver pattern from docs/brand-drop/mocks/visuals.html.
 *
 * Observes the nearest ancestor matching `.cc-card, .player-card, .cc-insight, [data-count-card]`
 * (falls back to self) at threshold 0.35.
 */
export default function CountUp({
  to,
  decimals = 0,
  duration = 720,
  suffix,
  className,
}: Props) {
  const spanRef = useRef<HTMLSpanElement | null>(null);
  const rafRef = useRef<number | null>(null);
  const isInViewRef = useRef(false);
  const [text, setText] = useState<string>(() => (0).toFixed(decimals));

  useEffect(() => {
    const el = spanRef.current;
    if (!el) return;

    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    const formatValue = (v: number) => v.toFixed(decimals);

    const runCount = () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      if (reduce) {
        setText(formatValue(to));
        return;
      }
      const start = performance.now();
      const step = (now: number) => {
        const t = Math.min(1, (now - start) / duration);
        // ease-out-quart
        const eased = 1 - Math.pow(1 - t, 4);
        const v = to * eased;
        setText(formatValue(v));
        if (t < 1) {
          rafRef.current = requestAnimationFrame(step);
        } else {
          rafRef.current = null;
          setText(formatValue(to));
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

    // Find nearest card-like container so trigger fires when the whole card enters view.
    const card =
      (el.closest('.cc-card, .player-card, .cc-insight, [data-count-card]') as Element | null) ||
      el;

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
  }, [to, decimals, duration]);

  return (
    <span ref={spanRef} className={className} data-count-to={to}>
      {text}
      {suffix}
    </span>
  );
}
