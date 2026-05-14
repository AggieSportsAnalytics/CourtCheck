'use client';

import { useEffect, useState } from 'react';

/**
 * Horizontal bar chart with staggered fill + count-up animation.
 * Ported from visuals.html buildBars() / match-detail.html.
 *
 * Timing (locked):
 *   - 760ms fill, cubic-bezier(0.165, 0.84, 0.44, 1)
 *   - 100ms stagger per row
 *   - ease-out-quart on the count-up tick so the number lands clean
 *
 * Animates on mount and whenever `data` changes (e.g. poll → done transition).
 * Respects prefers-reduced-motion.
 */
export type BarDatum = {
  label: string;
  pct: number;
  color: string; // CSS var or hex
};

type Props = {
  data: BarDatum[];
  /** ARIA label for the chart's purpose, e.g. "Shot mix". */
  ariaLabel?: string;
};

const BAR_DUR = 760;
const BAR_STAGGER = 100;

export default function ShotBars({ data, ariaLabel }: Props) {
  // Display values that the CSS transition + count-up follow. Initialized at 0
  // so the bars enter from empty regardless of incoming data, then ramp.
  const [widths, setWidths] = useState<number[]>(() => data.map(() => 0));
  const [displayPcts, setDisplayPcts] = useState<number[]>(() => data.map(() => 0));
  const [shown, setShown] = useState<boolean[]>(() => data.map(() => false));

  useEffect(() => {
    // Reset to zero, then ramp on the next frame so CSS transitions catch the
    // change. Without the reset, identical pct values across renders never
    // re-fire the transition.
    setWidths(data.map(() => 0));
    setDisplayPcts(data.map(() => 0));
    setShown(data.map(() => false));

    if (typeof window === 'undefined') return;
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    const rafs: number[] = [];
    const timeouts: ReturnType<typeof setTimeout>[] = [];

    if (reduce) {
      // Skip transitions for users who opted out — show final state on next tick.
      const t = setTimeout(() => {
        setWidths(data.map((d) => d.pct));
        setDisplayPcts(data.map((d) => d.pct));
        setShown(data.map(() => true));
      }, 0);
      timeouts.push(t);
      return () => timeouts.forEach((x) => clearTimeout(x));
    }

    data.forEach((d, i) => {
      const t = setTimeout(() => {
        setShown((prev) => {
          const next = prev.slice();
          next[i] = true;
          return next;
        });
        setWidths((prev) => {
          const next = prev.slice();
          next[i] = d.pct;
          return next;
        });
        const start = performance.now();
        const step = (now: number) => {
          const elapsed = Math.min(1, (now - start) / BAR_DUR);
          const eased = 1 - Math.pow(1 - elapsed, 4);
          setDisplayPcts((prev) => {
            const next = prev.slice();
            next[i] = Math.round(d.pct * eased);
            return next;
          });
          if (elapsed < 1) {
            rafs.push(requestAnimationFrame(step));
          } else {
            setDisplayPcts((prev) => {
              const next = prev.slice();
              next[i] = d.pct;
              return next;
            });
          }
        };
        rafs.push(requestAnimationFrame(step));
      }, i * BAR_STAGGER);
      timeouts.push(t);
    });

    return () => {
      timeouts.forEach((x) => clearTimeout(x));
      rafs.forEach((r) => cancelAnimationFrame(r));
    };
  }, [data]);

  return (
    <div aria-label={ariaLabel}>
      {data.map((d, i) => (
        <div
          key={`${d.label}-${i}`}
          className="grid items-center py-1.5"
          style={{ gridTemplateColumns: '110px 1fr 56px', gap: 14 }}
        >
          <span className="text-[0.9rem] font-medium text-ink-soft">{d.label}</span>
          <div className="h-5 rounded-full overflow-hidden relative bg-shade dark:bg-surface">
            <div
              className="h-full rounded-full"
              style={{
                width: `${widths[i] ?? 0}%`,
                background: d.color,
                transformOrigin: 'left center',
                transition: 'width 760ms cubic-bezier(0.165, 0.84, 0.44, 1)',
              }}
            />
          </div>
          <span
            className="text-right text-[0.98rem] font-display font-medium text-ink"
            style={{
              fontFeatureSettings: '"tnum"',
              opacity: shown[i] ? 1 : 0,
              transition: 'opacity 320ms var(--ease-out)',
            }}
          >
            {displayPcts[i] ?? 0}%
          </span>
        </div>
      ))}
    </div>
  );
}
