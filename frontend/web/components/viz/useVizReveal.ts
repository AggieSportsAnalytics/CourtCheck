'use client';

import { useEffect, useRef } from 'react';

/**
 * Mount/data-change reveal for SVG viz primitives. Returns a ref to attach
 * to the wrapping SVG.
 *
 * Implementation: toggles a `cc-viz-prereveal` class on the SVG root. CSS
 * rules in globals.css force any `.shot-dot`, `.heatmap-zone`, `.spacing-line`,
 * or `.spacing-endpoint` inside a `.cc-viz-prereveal` parent to opacity 0.
 * On mount AND on `depKey` change, the class is applied, then removed after
 * two animation frames so the CSS opacity transition (480ms) fires.
 *
 * The IO-based variant was tried but unreliable: when the SVG is already in
 * view at observe time, IO fires before the prereveal opacity-0 state gets
 * its own paint frame, and the browser collapses both into one paint —
 * skipping the transition entirely. Mount-trigger is rock-solid because the
 * raf double-buffer guarantees a separate paint for the hidden state.
 *
 * The original "viz only animated above the fold" complaint is moot because
 * viz tiles re-mount with a new depKey when data arrives — by then the user
 * has scrolled, and the animation plays into their viewport.
 *
 * No imperative `el.style.opacity = ...` — React owns the DOM, CSS owns the
 * transition.
 */
export function useVizReveal<T extends SVGElement>(
  _selector: string,
  options: { staggerMs?: number; depKey?: unknown } = {},
) {
  const ref = useRef<T | null>(null);
  const { depKey } = options;

  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    if (typeof window === 'undefined') return;

    node.classList.add('cc-viz-prereveal');
    // Two animation frames so the class commit lands as a separate paint from
    // the removal — without this the browser collapses both into one paint
    // and skips the transition.
    let raf2Id: number | null = null;
    const raf1Id = requestAnimationFrame(() => {
      raf2Id = requestAnimationFrame(() => {
        node.classList.remove('cc-viz-prereveal');
      });
    });

    return () => {
      cancelAnimationFrame(raf1Id);
      if (raf2Id !== null) cancelAnimationFrame(raf2Id);
      node.classList.remove('cc-viz-prereveal');
    };
  }, [depKey]);

  return ref;
}
