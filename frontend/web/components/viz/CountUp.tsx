'use client';

import { useEffect, useRef, useState } from 'react';

/**
 * rAF count-up from 0 → `value`, gated by `play`. Ease-out-quart over
 * `durationMs` (default 760ms — matches the canonical buildBars tick in
 * docs/brand-drop/mocks/visuals.html). Honors prefers-reduced-motion by
 * snapping straight to the value.
 *
 * Pair `play` with the same entrance gate that drives the bar width so the
 * number and the bar rise together.
 */
export default function CountUp({
  value,
  play,
  durationMs = 760,
  decimals = 0,
  suffix = '',
}: {
  value: number;
  play: boolean;
  durationMs?: number;
  decimals?: number;
  suffix?: string;
}) {
  const [display, setDisplay] = useState(0);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    if (!play) {
      setDisplay(0);
      return;
    }
    if (
      typeof window !== 'undefined' &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches
    ) {
      setDisplay(value);
      return;
    }
    const start = performance.now();
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / durationMs);
      const eased = 1 - Math.pow(1 - t, 4);
      setDisplay(value * eased);
      if (t < 1) {
        rafRef.current = requestAnimationFrame(tick);
      } else {
        setDisplay(value);
      }
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    };
  }, [value, play, durationMs]);

  const shown =
    decimals > 0 ? display.toFixed(decimals) : Math.round(display).toString();
  return (
    <>
      {shown}
      {suffix}
    </>
  );
}
