'use client';

import { forwardRef, ReactNode } from 'react';

/**
 * Vertical half-court SVG primitive. Ported verbatim from
 * docs/brand-drop/mocks/visuals.html buildCourtSVG().
 *
 * Coords: x=0..27 (width, real ft), y=0..78 (length, real ft).
 * Net at y=39. Top half shows opponent's side, bottom half shows player's.
 * The viewBox is cropped to the active half so a 27/39 tile renders to scale.
 *
 * The returned <svg> exposes a `data-shadow-id` filter that callers (shot map)
 * can use for the dot drop shadow.
 */
export type CourtHalf = 'top' | 'bottom';

type Props = {
  half: CourtHalf;
  shadowId: string;
  children?: ReactNode;
  className?: string;
  /** Optional inline style overrides — merged on top of the default SVG
   *  layout style. Used by useVizReveal to control entrance opacity/transform. */
  style?: React.CSSProperties;
  /**
   * Extra court units to extend the viewBox past the half boundary. For
   * half='bottom', adds room below the baseline so players standing 1–5 ft
   * behind the line still render inside the tile. For half='top', adds room
   * above the opponent baseline for long out-of-bounds bounces. Default 0.
   */
  extendBehind?: number;
  /**
   * Extra court units to extend the viewBox past each sideline so wide
   * out-of-bounds bounces still render. Default 0.
   */
  extendSide?: number;
};

/** Court tile aspect ratio (width / height) for a given half + extension. */
export function courtTileAspect(half: CourtHalf, extendBehind = 0, extendSide = 0): number {
  return (27 + extendSide * 2) / (39 + extendBehind);
}

const CourtSVG = forwardRef<SVGSVGElement, Props>(function CourtSVG(
  { half, shadowId, children, className, style, extendBehind = 0, extendSide = 0 },
  ref,
) {
  const yMin = half === 'bottom' ? 39 : 0 - (half === 'top' ? extendBehind : 0);
  const height = 39 + extendBehind;
  const xMin = 0 - extendSide;
  const width = 27 + extendSide * 2;

  return (
    <svg
      ref={ref}
      viewBox={`${xMin} ${yMin} ${width} ${height}`}
      preserveAspectRatio="none"
      data-shadow-id={shadowId}
      className={className}
      style={{ display: 'block', width: '100%', height: '100%', ...style }}
    >
      <defs>
        <filter id={shadowId} x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur in="SourceAlpha" stdDeviation="0.25" />
          <feOffset dy="0.2" result="off" />
          <feComponentTransfer>
            <feFuncA type="linear" slope="0.45" />
          </feComponentTransfer>
          <feMerge>
            <feMergeNode />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* Court boundary (full length, drawn so net edge is visible) */}
      <rect
        x={1}
        y={1}
        width={25}
        height={76}
        fill="none"
        stroke="white"
        strokeWidth={0.3}
      />
      {/* Net at y=39 */}
      <line x1={0} y1={39} x2={27} y2={39} stroke="white" strokeWidth={0.5} />
      {/* Service lines */}
      <line x1={1} y1={18} x2={26} y2={18} stroke="white" strokeWidth={0.25} />
      <line x1={1} y1={60} x2={26} y2={60} stroke="white" strokeWidth={0.25} />
      {/* Center service line */}
      <line x1={13.5} y1={18} x2={13.5} y2={60} stroke="white" strokeWidth={0.25} />
      {/* Center marks on baselines */}
      <line x1={13.5} y1={0.5} x2={13.5} y2={1} stroke="white" strokeWidth={0.3} />
      <line x1={13.5} y1={77} x2={13.5} y2={77.5} stroke="white" strokeWidth={0.3} />

      {children}
    </svg>
  );
});

export default CourtSVG;

/**
 * Canonical stroke vocabulary. 3-class to match the live stroke classifier
 * (`backend/models/stroke_classifier_tcn.py STROKE_LABELS`).
 * Uses brand CSS vars so dark mode lights up automatically.
 */
// Stroke palette — pinned to the dark-mode brand hexes so the dashboard
// colors stay identical in both light and dark mode AND match the colors
// the backend bakes into the annotated video (STROKE_COLORS_BGR in
// backend/vision/drawing.py). Single source of truth — every stroke
// surface (ShotMap dots, legend chips, mix bars, BouncePanel chip,
// minimap bounces) reads from STROKE_COLOR_BY_KEY below or this array.
//   forehand → #6FA88B  brand sage green (matches BGR(139,168,111))
//   backhand → #B584A6  brand plum       (matches BGR(166,132,181))
//   serve    → #DDB166  brand amber      (matches BGR(102,177,221))
export const STROKES = [
  { key: 'forehand', label: 'Forehand', color: '#6FA88B' },
  { key: 'backhand', label: 'Backhand', color: '#B584A6' },
  { key: 'serve', label: 'Serve/Overhead', color: '#DDB166' },
] as const;

export type StrokeKey = (typeof STROKES)[number]['key'];

export const STROKE_COLOR_BY_KEY: Record<StrokeKey, string> = Object.fromEntries(
  STROKES.map((s) => [s.key, s.color])
) as Record<StrokeKey, string>;
