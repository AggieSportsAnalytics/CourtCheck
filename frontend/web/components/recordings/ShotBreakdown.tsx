'use client';

import ShotBars, { BarDatum } from '../viz/ShotBars';

type Props = {
  mix: BarDatum[];
};

/**
 * "Shot mix" card. How often each stroke was hit (frequency %).
 *
 * Per-stroke accuracy used to live here as a second column but moved into
 * the Shot Map's right rail (ShotAccuracyMini in VizPanel) — both surfaces
 * derive from the same `realDots` filter, so they were duplicating the
 * same data shown two different ways.
 */
export default function ShotBreakdown({ mix }: Props) {
  return (
    <div
      className="cc-card bg-paper border border-line rounded-[14px] mb-8"
      style={{ padding: '26px 28px' }}
    >
      <div className="mb-4">
        <span className="inline-flex items-center gap-2 font-mono text-[0.72rem] uppercase tracking-[0.18em] text-court before:content-[''] before:w-1.5 before:h-1.5 before:bg-clay before:rounded-full">
          Shot mix · this recording
        </span>
        <h3 className="font-display font-medium text-[1.25rem] tracking-tight mt-3">
          How often each stroke fires.
        </h3>
        <div className="text-ink-soft text-[0.95rem] mt-1">
          Per-stroke accuracy lives in the Shot Map rail above.
        </div>
      </div>

      <div className="mt-2 max-w-[640px]">
        <ShotBars data={mix} ariaLabel="Shot mix" />
      </div>
    </div>
  );
}
