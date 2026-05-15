'use client';

import { useEffect, useRef } from 'react';

/**
 * Scroll-triggered reveal for SVG viz primitives. Returns a ref to attach
 * to the wrapping SVG (className kept for backwards compat).
 *
 * Implementation: per-child Web Animations API (WAAPI) with
 * target-aware end values. On scroll-into-viewport, every child matching
 * `selector` inside the wrapper animates from opacity 0 to ITS OWN target
 * opacity (read from the SVG `opacity` attribute, default 1). This
 * preserves per-cell intensity on heatmap-zone (where each cell has
 * `opacity={0.04 + intensity * 0.66}`) — animating to a fixed `1` and
 * holding via `fill: 'forwards'` would lock every cell at full opacity
 * and flatten the gradient.
 *
 * After the animation finishes, we remove the inline override AND cancel
 * the animation effect so the SVG attribute opacity wins again — future
 * class toggles (`.dim`, selection, etc.) work normally.
 *
 * Honors prefers-reduced-motion (skips the animation).
 */
export function useVizReveal<T extends SVGElement>(
  selector: string,
  options: { staggerMs?: number; durationMs?: number; depKey?: unknown } = {},
) {
  const ref = useRef<T | null>(null);
  const { staggerMs = 30, durationMs = 480, depKey } = options;

  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    if (typeof window === 'undefined') return;

    const reduceMotion = window.matchMedia(
      '(prefers-reduced-motion: reduce)',
    ).matches;
    if (reduceMotion) {
      console.log('[viz-reveal] reduced-motion ON — skipping animation');
      return;
    }

    const targets = Array.from(node.querySelectorAll<SVGElement>(selector));
    if (!targets.length) {
      console.log(`[viz-reveal] no children matched "${selector}" inside`, node);
      return;
    }

    // Read each child's TARGET opacity from the SVG `opacity` attribute
    // (heatmap cells have per-intensity values; shot-dot has no attr → 1).
    const targetOpacities = targets.map((el) => {
      const attr = el.getAttribute('opacity');
      return attr ? parseFloat(attr) : 1;
    });

    // Pre-hide via inline style (overrides SVG attribute) so children
    // don't flash at full opacity before the IO callback fires.
    targets.forEach((el) => {
      el.style.opacity = '0';
    });
    console.log(
      `[viz-reveal] hidden ${targets.length} "${selector}" children, awaiting scroll-in`,
    );

    let triggered = false;
    const anims: Animation[] = [];
    const reveal = () => {
      if (triggered) return;
      triggered = true;
      console.log(`[viz-reveal] revealing ${targets.length} "${selector}" children`);
      targets.forEach((el, i) => {
        const target = targetOpacities[i];
        const anim = el.animate(
          [{ opacity: 0 }, { opacity: target }],
          {
            duration: durationMs,
            delay: i * staggerMs,
            easing: 'cubic-bezier(0.2, 0.7, 0.2, 1)',
            fill: 'forwards',
          },
        );
        anims.push(anim);
        anim.onfinish = () => {
          // Remove inline override → SVG attribute (target) wins.
          el.style.removeProperty('opacity');
          // Cancel the WAAPI effect so future opacity changes (.dim, etc.)
          // aren't blocked by the held fill:'forwards' value.
          anim.cancel();
        };
      });
    };

    const obs = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            console.log(`[viz-reveal] intersect → revealing "${selector}"`);
            obs.disconnect();
            requestAnimationFrame(reveal);
            return;
          }
        }
      },
      { threshold: 0, rootMargin: '0px' },
    );
    obs.observe(node);

    return () => {
      obs.disconnect();
      anims.forEach((a) => a.cancel());
      targets.forEach((el) => el.style.removeProperty('opacity'));
    };
  }, [selector, staggerMs, durationMs, depKey]);

  return {
    ref,
    style: undefined as React.CSSProperties | undefined,
    className: '',
  };
}
