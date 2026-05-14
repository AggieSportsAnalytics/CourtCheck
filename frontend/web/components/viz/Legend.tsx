'use client';

import { STROKES, StrokeKey } from './CourtSVG';

/**
 * Interactive stroke legend. Ported from visuals.html renderLegend().
 *
 * Renders one chip per stroke (color swatch + label + count). Clicking a chip
 * toggles the filter; when a filter is active, a "Show all" reset chip appears
 * and non-active chips dim.
 */
type Props = {
  counts: Partial<Record<StrokeKey, number>>;
  /** Bounces detected but not paired with a classified stroke. Shown as a
   *  dimmed, non-interactive chip so coaches see the full bounce volume. */
  unknownCount?: number;
  activeKey: StrokeKey | null;
  onToggle: (key: StrokeKey) => void;
};

export default function Legend({ counts, unknownCount, activeKey, onToggle }: Props) {
  return (
    <div className="flex flex-wrap gap-2">
      {STROKES.map((s) => {
        const isActive = activeKey === s.key;
        const isMuted = activeKey !== null && !isActive;
        return (
          <button
            key={s.key}
            type="button"
            onClick={() => onToggle(s.key)}
            className={`cc-legend-chip inline-flex items-center gap-2 pl-3 pr-3.5 py-1.5 rounded-full border bg-paper text-[0.82rem] font-medium select-none ${
              isActive
                ? 'border-ink text-ink bg-cream dark:bg-surface'
                : 'border-line text-ink-soft'
            } ${isMuted ? 'muted' : ''}`}
          >
            <span
              className="block w-2.5 h-2.5 rounded-full shrink-0"
              style={{ background: s.color }}
            />
            <span>{s.label}</span>
            {counts[s.key] != null && (
              <span className="font-mono text-[0.7rem] text-ink-mute ml-0.5">
                {counts[s.key]}
              </span>
            )}
          </button>
        );
      })}
      {unknownCount != null && unknownCount > 0 && (
        <span
          className="cc-legend-chip inline-flex items-center gap-2 pl-3 pr-3.5 py-1.5 rounded-full border border-line bg-paper text-ink-mute text-[0.82rem] font-medium select-none"
          aria-label={`${unknownCount} unclassified bounces`}
        >
          <span
            className="block w-2.5 h-2.5 rounded-full shrink-0 border border-line"
            style={{ background: 'var(--color-paper)' }}
          />
          <span>Unclassified</span>
          <span className="font-mono text-[0.7rem] text-ink-mute ml-0.5">
            {unknownCount}
          </span>
        </span>
      )}
      {activeKey && (
        <button
          type="button"
          onClick={() => onToggle(activeKey)}
          className="cc-legend-chip inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full border border-line bg-paper text-ink-soft text-[0.82rem] font-medium italic"
        >
          Show all
        </button>
      )}
    </div>
  );
}
