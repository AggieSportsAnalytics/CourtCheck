'use client';

import { useId, useMemo } from 'react';
import CourtSVG, { STROKES, StrokeKey } from './CourtSVG';
import { useVizReveal } from './useVizReveal';

/**
 * Spacing viz: each shot is drawn as player→ball line, colored by extension
 * quality (ideal=court, squeezed=clay, jammed=plum, long=amber). Ball endpoint
 * is lime to carry the "lime = the ball" semantic from the brand laws.
 *
 * Ported from visuals.html buildSpacing(). Bottom-half view (player's side).
 */
export type SpacingQuality = 'jammed' | 'squeezed' | 'ideal' | 'long';

export type SpacingShot = {
  stroke: StrokeKey;
  px: number;
  py: number;
  bx: number;
  by: number;
  q: SpacingQuality;
};

const QUALITY_COLOR: Record<SpacingQuality, string> = {
  jammed: 'var(--color-plum)',
  squeezed: 'var(--color-clay)',
  ideal: 'var(--color-court)',
  long: 'var(--color-amber)',
};

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

  const { ref: svgRef, style: revealStyle } = useVizReveal<SVGSVGElement>(
    '.spacing-line, .spacing-endpoint',
    { staggerMs: 25, depKey: shots.length },
  );

  return (
    <CourtSVG ref={svgRef} style={revealStyle} half="bottom" shadowId={shadowId} extendBehind={SPACING_EXTEND_BEHIND}>
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
