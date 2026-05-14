'use client';

import { useId, useMemo } from 'react';
import CourtSVG, { STROKES, STROKE_COLOR_BY_KEY, StrokeKey } from './CourtSVG';
import { useVizReveal } from './useVizReveal';

/**
 * Shot map viz. Ported from visuals.html buildShotMap().
 *
 * Dots colored by stroke type, layered over a white halo with drop-shadow.
 * Top-half-court view (opponent's side — y=0..39). When the user filters by
 * stroke (legend chip), non-matching dots gain a `.dim` class which globals.css
 * fades to 10% opacity via the `.shot-dot.dim` rule. (No keyframe animation
 * here so JS-set opacity isn't fought.)
 */
export type ShotDot = {
  x: number;
  y: number;
  /** 'unknown' covers bounces detected without a confidently paired stroke. */
  stroke: StrokeKey | 'unknown';
  /** False = bounce landed out. Rendered as an X marker, not a dot. Defaults
   *  to true so callers that don't pass `in` keep the prior behavior. */
  in?: boolean;
};

type Props = {
  dots: ShotDot[];
  activeFilter: StrokeKey | null;
};

/** Court units the shot map extends past the doubles sideline and past the
 *  opponent's baseline so a clearly-out bounce still renders. Tuned so wide /
 *  long bounces within reason fit in the tile without ballooning the view. */
export const SHOT_MAP_EXTEND_BEHIND = 4;
export const SHOT_MAP_EXTEND_SIDE = 2;

/** Sample data — mock M. Lin's shots landing on opponent's side. */
export const SAMPLE_SHOTS: ShotDot[] = [
  // Forehand cross-court winners landing deep on opponent's right
  { x: 21, y: 5, stroke: 'forehand' },
  { x: 23, y: 7, stroke: 'forehand' },
  { x: 19, y: 11, stroke: 'forehand' },
  { x: 24, y: 9, stroke: 'forehand' },
  { x: 22, y: 14, stroke: 'forehand' },
  { x: 25, y: 13, stroke: 'forehand' },
  { x: 20, y: 17, stroke: 'forehand' },
  { x: 23, y: 18, stroke: 'forehand' },
  // Backhand cross-court (opponent's left)
  { x: 5, y: 6, stroke: 'backhand' },
  { x: 7, y: 9, stroke: 'backhand' },
  { x: 4, y: 13, stroke: 'backhand' },
  { x: 6, y: 16, stroke: 'backhand' },
  { x: 8, y: 11, stroke: 'backhand' },
  { x: 3, y: 8, stroke: 'backhand' },
  // Serves into service boxes
  { x: 18, y: 25, stroke: 'serve' },
  { x: 21, y: 28, stroke: 'serve' },
  { x: 24, y: 32, stroke: 'serve' },
  { x: 7, y: 24, stroke: 'serve' },
  { x: 5, y: 30, stroke: 'serve' },
  { x: 9, y: 34, stroke: 'serve' },
];

export function shotMapCounts(dots: ShotDot[]): Partial<Record<StrokeKey, number>> {
  const counts: Partial<Record<StrokeKey, number>> = {};
  STROKES.forEach((s) => {
    counts[s.key] = 0;
  });
  dots.forEach((d) => {
    if (d.stroke === 'unknown') return; // unknown not represented in stroke legend
    counts[d.stroke] = (counts[d.stroke] ?? 0) + 1;
  });
  return counts;
}

/** Count of bounces that landed but weren't paired with a classified stroke. */
export function shotMapUnknownCount(dots: ShotDot[]): number {
  return dots.reduce((n, d) => (d.stroke === 'unknown' ? n + 1 : n), 0);
}

// Neutral color for bounces detected without a confidently paired stroke event.
const UNKNOWN_STROKE_COLOR = 'var(--color-ink-mute)';

export default function ShotMap({ dots, activeFilter }: Props) {
  const rawId = useId();
  const shadowId = useMemo(
    () => `dot-shadow-${rawId.replace(/[^a-zA-Z0-9]/g, '')}`,
    [rawId]
  );

  // Order dots by stroke so the rendered layering matches the legend order.
  // 'unknown' dots render first (under) so the colored stroke dots sit on top.
  const ordered = useMemo(() => {
    const byStroke: ShotDot[] = [];
    dots.filter((d) => d.stroke === 'unknown').forEach((d) => byStroke.push(d));
    STROKES.forEach((s) => {
      dots.filter((d) => d.stroke === s.key).forEach((d) => byStroke.push(d));
    });
    return byStroke;
  }, [dots]);

  const svgRef = useVizReveal<SVGSVGElement>('.shot-dot', {
    staggerMs: 30,
    depKey: ordered.length,
  });

  return (
    <CourtSVG
      ref={svgRef}
      half="top"
      shadowId={shadowId}
      extendBehind={SHOT_MAP_EXTEND_BEHIND}
      extendSide={SHOT_MAP_EXTEND_SIDE}
    >
      {ordered.map((d, i) => {
        const isUnknown = d.stroke === 'unknown';
        const isOut = d.in === false;
        // Filter chips only narrow named strokes — unknowns always dim under any active filter.
        const dim = activeFilter !== null && (isUnknown || activeFilter !== d.stroke);
        const color = isUnknown
          ? UNKNOWN_STROKE_COLOR
          : STROKE_COLOR_BY_KEY[d.stroke as StrokeKey];
        if (isOut) {
          // OOB: render an X. Stroke color matches the stroke type so the
          // legend still narrows by stroke; an outer white halo keeps it
          // legible on the green court.
          const arm = 0.95;
          return (
            <g
              key={i}
              className={`shot-dot shot-dot-out ${dim ? 'dim' : ''}`}
              data-stroke={d.stroke}
              data-in="false"
            >
              <line
                x1={d.x - arm}
                y1={d.y - arm}
                x2={d.x + arm}
                y2={d.y + arm}
                stroke="white"
                strokeWidth={0.55}
                strokeLinecap="round"
                opacity={0.85}
              />
              <line
                x1={d.x - arm}
                y1={d.y + arm}
                x2={d.x + arm}
                y2={d.y - arm}
                stroke="white"
                strokeWidth={0.55}
                strokeLinecap="round"
                opacity={0.85}
              />
              <line
                x1={d.x - arm}
                y1={d.y - arm}
                x2={d.x + arm}
                y2={d.y + arm}
                stroke={color}
                strokeWidth={0.32}
                strokeLinecap="round"
                opacity={0.95}
              />
              <line
                x1={d.x - arm}
                y1={d.y + arm}
                x2={d.x + arm}
                y2={d.y - arm}
                stroke={color}
                strokeWidth={0.32}
                strokeLinecap="round"
                opacity={0.95}
              />
            </g>
          );
        }
        return (
          <g
            key={i}
            className={`shot-dot ${dim ? 'dim' : ''}`}
            data-stroke={d.stroke}
            data-in="true"
          >
            <circle
              cx={d.x}
              cy={d.y}
              r={1.15}
              fill="white"
              filter={`url(#${shadowId})`}
              opacity={isUnknown ? 0.5 : 0.85}
            />
            <circle
              cx={d.x}
              cy={d.y}
              r={isUnknown ? 0.7 : 0.85}
              fill={color}
              opacity={isUnknown ? 0.65 : 0.95}
            />
          </g>
        );
      })}
    </CourtSVG>
  );
}
