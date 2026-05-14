'use client';

import { useId, useMemo } from 'react';
import CourtSVG from './CourtSVG';
import { useVizReveal } from './useVizReveal';

/**
 * Court coverage heatmap. 8 cols × 12 rows on the bottom half-court.
 * Ported from visuals.html buildCoverage() / generateCoverage().
 *
 * Cell intensity is derived from a coaching heuristic (most time near baseline,
 * slight deuce-side bias). Final opacity = 0.04 + intensity * 0.66 on a lime
 * fill so the heatmap stays inside the "lime = ball / hot signal" semantic.
 */
const ROWS = 12;
const COLS = 8;

function generateCoverage(rows: number, cols: number): number[][] {
  const grid: number[][] = [];
  for (let r = 0; r < rows; r++) {
    const row: number[] = [];
    for (let c = 0; c < cols; c++) {
      const lengthFrac = r / (rows - 1); // 0 = net, 1 = baseline
      const widthFrac = (c + 0.5) / cols; // 0 = left, 1 = right
      const lengthIntensity = Math.pow(lengthFrac, 1.7);
      const widthDist = Math.abs(widthFrac - 0.55);
      const widthIntensity = Math.max(0, 1 - Math.pow(widthDist * 1.8, 1.5));
      let intensity = lengthIntensity * widthIntensity;
      const seed = ((r * 17 + c * 23) % 100) / 100;
      intensity *= 0.82 + seed * 0.32;
      row.push(Math.min(1, Math.max(0, intensity)));
    }
    grid.push(row);
  }
  return grid;
}

export const SAMPLE_COVERAGE = generateCoverage(ROWS, COLS);

type Props = {
  grid?: number[][];
};

/** Extra court units below the baseline shown in the coverage viz — players
 *  often spend most of the recording 1–5 ft behind the baseline. */
export const COVERAGE_EXTEND_BEHIND = 10;

export default function Coverage({ grid = SAMPLE_COVERAGE }: Props) {
  const rawId = useId();
  const shadowId = useMemo(
    () => `dot-shadow-${rawId.replace(/[^a-zA-Z0-9]/g, '')}`,
    [rawId]
  );

  // Grid covers y=39 (net) to y=85 (~7 court units past the baseline).
  // Backend mirrors this in build_coverage_grid. Without the extra behind-
  // baseline rows, every behind-baseline position clamped to the last row and
  // the heatmap read as a single hot stripe even when the player moved.
  const halfH = 46;
  const halfW = 25;
  const cellH = halfH / ROWS;
  const cellW = halfW / COLS;
  const yOffset = 39;

  const svgRef = useVizReveal<SVGSVGElement>('.heatmap-zone', {
    staggerMs: 8,
    depKey: grid.length,
  });

  return (
    <CourtSVG ref={svgRef} half="bottom" shadowId={shadowId} extendBehind={COVERAGE_EXTEND_BEHIND}>
      {grid.map((row, r) =>
        row.map((intensity, c) => (
          <rect
            key={`${r}-${c}`}
            className="heatmap-zone"
            x={1 + c * cellW}
            y={yOffset + r * cellH}
            width={cellW}
            height={cellH}
            fill="var(--color-lime)"
            opacity={0.04 + intensity * 0.66}
          />
        ))
      )}
    </CourtSVG>
  );
}
