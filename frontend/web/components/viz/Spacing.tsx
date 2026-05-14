'use client';

import { useId, useMemo } from 'react';
import CourtSVG, { STROKES, StrokeKey } from './CourtSVG';
import { useVizReveal } from './useVizReveal';

/**
 * Spacing viz: each shot is drawn as player→ball line, colored by extension
 * quality (ideal=court, squeezed=clay, long=amber). Ball endpoint is lime to
 * carry the "lime = the ball" semantic from the brand laws.
 *
 * Ported from visuals.html buildSpacing(). Bottom-half view (player's side).
 */
export type SpacingQuality = 'ideal' | 'squeezed' | 'long';

export type SpacingShot = {
  stroke: StrokeKey;
  px: number;
  py: number;
  bx: number;
  by: number;
  q: SpacingQuality;
};

const QUALITY_COLOR: Record<SpacingQuality, string> = {
  ideal: 'var(--color-court)',
  squeezed: 'var(--color-clay)',
  long: 'var(--color-amber)',
};

/** Sample contact spacing — M. Lin contacts in the bottom half-court. */
export const SAMPLE_SPACING: SpacingShot[] = [
  // Forehand (deuce/right side)
  { stroke: 'forehand', px: 19, py: 73, bx: 22, by: 71, q: 'ideal' },
  { stroke: 'forehand', px: 20, py: 75, bx: 23, by: 73, q: 'ideal' },
  { stroke: 'forehand', px: 21, py: 71, bx: 25, by: 69, q: 'ideal' },
  { stroke: 'forehand', px: 18, py: 76, bx: 19, by: 75, q: 'squeezed' },
  { stroke: 'forehand', px: 22, py: 72, bx: 23, by: 71, q: 'squeezed' },
  { stroke: 'forehand', px: 16, py: 74, bx: 25, by: 71, q: 'long' },
  { stroke: 'forehand', px: 20, py: 75, bx: 23, by: 72, q: 'ideal' },
  { stroke: 'forehand', px: 19, py: 73, bx: 22, by: 70, q: 'ideal' },
  // Backhand (ad/left side)
  { stroke: 'backhand', px: 5, py: 73, bx: 8, by: 70, q: 'ideal' },
  { stroke: 'backhand', px: 6, py: 75, bx: 7, by: 75, q: 'squeezed' },
  { stroke: 'backhand', px: 7, py: 72, bx: 7, by: 71, q: 'squeezed' },
  { stroke: 'backhand', px: 4, py: 74, bx: 7, by: 72, q: 'ideal' },
  { stroke: 'backhand', px: 5, py: 73, bx: 6, by: 73, q: 'squeezed' },
  { stroke: 'backhand', px: 8, py: 71, bx: 11, by: 68, q: 'ideal' },
  { stroke: 'backhand', px: 3, py: 75, bx: 9, by: 72, q: 'long' },
  // Serves (player at baseline)
  { stroke: 'serve', px: 13, py: 77, bx: 13, by: 75, q: 'ideal' },
  { stroke: 'serve', px: 14, py: 77, bx: 14, by: 75, q: 'ideal' },
  { stroke: 'serve', px: 13, py: 77, bx: 15, by: 75, q: 'ideal' },
  { stroke: 'serve', px: 14, py: 77, bx: 14, by: 75, q: 'ideal' },
];

export function spacingCounts(
  shots: SpacingShot[]
): Partial<Record<StrokeKey, number>> {
  const counts: Partial<Record<StrokeKey, number>> = {};
  STROKES.forEach((s) => {
    counts[s.key] = 0;
  });
  shots.forEach((s) => {
    counts[s.stroke] = (counts[s.stroke] ?? 0) + 1;
  });
  return counts;
}

type Props = {
  shots: SpacingShot[];
  activeFilter: StrokeKey | null;
};

/** Extra court units below the baseline shown in the spacing viz — players
 *  typically stand 1–5 ft behind the baseline during play. */
export const SPACING_EXTEND_BEHIND = 10;

export default function Spacing({ shots, activeFilter }: Props) {
  const rawId = useId();
  const shadowId = useMemo(
    () => `dot-shadow-${rawId.replace(/[^a-zA-Z0-9]/g, '')}`,
    [rawId]
  );

  const svgRef = useVizReveal<SVGSVGElement>(
    '.spacing-line, .spacing-endpoint',
    { staggerMs: 25, depKey: shots.length },
  );

  return (
    <CourtSVG ref={svgRef} half="bottom" shadowId={shadowId} extendBehind={SPACING_EXTEND_BEHIND}>
      {shots.map((s, i) => {
        const color = QUALITY_COLOR[s.q];
        const dim = activeFilter !== null && activeFilter !== s.stroke;
        const dimClass = dim ? 'dim' : '';
        return (
          <g key={i} data-stroke={s.stroke} data-quality={s.q}>
            <line
              className={`spacing-line ${dimClass}`}
              x1={s.px}
              y1={s.py}
              x2={s.bx}
              y2={s.by}
              stroke={color}
              strokeWidth={0.5}
              strokeLinecap="round"
              opacity={1}
            />
            {/* Player end (white, with quality-color outline) */}
            <circle
              className={`spacing-endpoint ${dimClass}`}
              cx={s.px}
              cy={s.py}
              r={0.55}
              fill="white"
              stroke={color}
              strokeWidth={0.18}
            />
            {/* Ball end (lime = the ball) */}
            <circle
              className={`spacing-endpoint ${dimClass}`}
              cx={s.bx}
              cy={s.by}
              r={0.45}
              fill="var(--color-lime)"
              stroke={color}
              strokeWidth={0.18}
            />
          </g>
        );
      })}
    </CourtSVG>
  );
}
