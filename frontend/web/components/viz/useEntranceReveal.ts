'use client';

import { useEffect, useRef, useState } from 'react';

/**
 * One-shot entrance gate. Returns false on first render, flips to true after
 * two animation frames so CSS `width` transitions on bar fills actually fire
 * (without this the bar renders directly at its target width and never
 * animates). Pair the returned flag with both the bar width and a <CountUp />
 * `play` prop so the bar and the number rise together.
 *
 * Re-arms when `dep` changes (e.g. data finishes loading).
 */
export function useEntranceReveal(dep?: unknown): boolean {
  const [shown, setShown] = useState(false);
  const raf2Ref = useRef<number | null>(null);

  useEffect(() => {
    setShown(false);
    const raf1 = requestAnimationFrame(() => {
      raf2Ref.current = requestAnimationFrame(() => setShown(true));
    });
    return () => {
      cancelAnimationFrame(raf1);
      if (raf2Ref.current != null) cancelAnimationFrame(raf2Ref.current);
    };
  }, [dep]);

  return shown;
}
