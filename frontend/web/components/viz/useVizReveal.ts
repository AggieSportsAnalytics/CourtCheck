'use client';

import { useEffect, useRef } from 'react';

/**
 * Scroll-on-mount reveal for SVG viz primitives. Returns a ref to attach to
 * the wrapping SVG.
 *
 * Implementation: toggles a `cc-viz-prereveal` class on the SVG root. CSS
 * rules in globals.css force any `.shot-dot`, `.heatmap-zone`, `.spacing-line`,
 * or `.spacing-endpoint` inside a `.cc-viz-prereveal` parent to opacity 0.
 * On mount the class is added immediately, then removed after one paint, which
 * triggers the CSS opacity transition (480ms) on every primitive at once.
 *
 * No imperative `el.style.opacity = ...` — React owns the DOM, CSS owns the
 * transition. Previous IO + style-mutation versions raced React's reconcile
 * loop and ended with cells/dots invisible after re-renders or Strict Mode
 * double-effect. This version cannot leak state into the DOM because every
 * "hidden" assertion is a single boolean on the SVG root.
 *
 * `staggerMs` and `depKey` are kept on the signature for callers; staggered
 * entrance is not currently implemented (mock had it; reliability over polish).
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
    // the removal — without this the browser collapses both into one paint and
    // skips the transition. raf IDs are plain numbers (not objects), so we
    // hold them in closure-scoped vars for the cleanup callback.
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
