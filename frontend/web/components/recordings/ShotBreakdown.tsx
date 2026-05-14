'use client';

import ShotBars, { BarDatum } from '../viz/ShotBars';

type Props = {
  mix: BarDatum[];
  accuracy: BarDatum[];
};

/**
 * Two-panel "Shot breakdown" card. Mix (frequency %) on left, Accuracy
 * (in/won %) on right. Stacks vertically below 880px.
 *
 * Bars are identical animation (760ms cubic, 100ms stagger, count-up tick) —
 * built from the canonical buildBars() viz primitive.
 */
export default function ShotBreakdown({ mix, accuracy }: Props) {
  return (
    <div
      className="cc-card bg-paper border border-line rounded-[14px] mb-8"
      style={{ padding: '26px 28px' }}
    >
      <div className="mb-4">
        <span className="inline-flex items-center gap-2 font-mono text-[0.72rem] uppercase tracking-[0.18em] text-court before:content-[''] before:w-1.5 before:h-1.5 before:bg-clay before:rounded-full">
          Shot breakdown · this recording
        </span>
        <h3 className="font-display font-medium text-[1.25rem] tracking-tight mt-3">
          How she played, this recording.
        </h3>
        <div className="text-ink-soft text-[0.95rem] mt-1">
          Left: how often. Right: how well.
        </div>
      </div>

      <div className="cc-bars-grid mt-2">
        <div className="min-w-0">
          <div className="font-mono text-[0.68rem] uppercase tracking-[0.14em] text-ink-mute mb-3.5">
            Mix
          </div>
          <ShotBars data={mix} ariaLabel="Shot mix" />
        </div>
        <div className="min-w-0">
          <div className="font-mono text-[0.68rem] uppercase tracking-[0.14em] text-ink-mute mb-3.5">
            Accuracy
          </div>
          <ShotBars data={accuracy} ariaLabel="Shot accuracy" />
        </div>
      </div>

      <style>{`
        .cc-bars-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 56px;
        }
        @media (max-width: 880px) {
          .cc-bars-grid {
            grid-template-columns: 1fr;
            gap: 32px;
          }
        }
      `}</style>
    </div>
  );
}
