'use client';

import { useMemo, useState } from 'react';
import ShotMap, { SAMPLE_SHOTS, shotMapCounts, shotMapUnknownCount, type ShotDot } from '../viz/ShotMap';
import Spacing, { SAMPLE_SPACING, spacingCounts, type SpacingShot } from '../viz/Spacing';
import Coverage from '../viz/Coverage';
import Legend from '../viz/Legend';
import { StrokeKey } from '../viz/CourtSVG';
import { PositionTile, type PositionSummary } from './CoachInsights';

type VizMode = 'shotMap' | 'spacing' | 'coverage';

/** API shape (from /api/recordings/[id].shots). */
export type ApiShot = {
  frame: number;
  time_s: number | null;
  stroke: 'forehand' | 'backhand' | 'serve' | 'unknown';
  player: 1 | 2;
  court_x: number;
  court_y: number;
  in: boolean;
  ball_court_x: number | null;
  ball_court_y: number | null;
  player_court_x: number | null;
  player_court_y: number | null;
};

type Props = {
  shots?: ApiShot[];
  /** 12x8 occupancy grid normalized to peak=1, from the pipeline. */
  coverageGrid?: number[][];
  /** 4-zone court-position summary. Rendered as a sub-tile in the Coverage
   *  mode's right rail so coaches see "where she stood" + "how often she
   *  stood at the baseline" side-by-side. */
  positionSummary?: PositionSummary | null;
  /** Recording status — drives the Coverage empty state vs sample fallback. */
  recordingStatus?: string;
};

const MODES: { key: VizMode; label: string }[] = [
  { key: 'shotMap', label: 'Shot map' },
  { key: 'spacing', label: 'Spacing' },
  { key: 'coverage', label: 'Coverage' },
];

const HEAD: Record<VizMode, { eyebrow: string; title: string; sub: string; halfLabel: string }> = {
  shotMap: {
    eyebrow: 'Shot map · this recording',
    title: 'Every bounce, by stroke.',
    sub: 'Tap a stroke to isolate. Hover for detail.',
    halfLabel: "Opponent's half · where shots land",
  },
  spacing: {
    eyebrow: 'Spacing · this recording',
    title: 'Contact spacing, by quality.',
    sub: 'Lines connect player to ball. Color encodes extension.',
    halfLabel: "Player's half · contact spacing",
  },
  coverage: {
    eyebrow: 'Coverage · this recording',
    title: 'Where she stood.',
    sub: 'Density of time-at-position over the recording.',
    halfLabel: "Player's half · time at position",
  },
};

/** Project API shots onto the top-half (opponent's side) court for ShotMap.
 *
 * Mirror by **bounce location**, not by who hit. Previously the code mirrored
 * only `s.player === 2` bounces — which assumed P1 always hits to opponent's
 * side. In practice that's not always true: contact-frame refinement can
 * attribute a bounce on the near half to a P1 swing event, and the pipeline
 * also emits bounces with `stroke='unknown'` that don't have a paired swing.
 * Those bounces ended up with y in [39, 78], failed the `y > 45` filter, and
 * disappeared from the shotmap even though they were visible on the minimap.
 *
 * Mirror anything past the net so every detected bounce shows on the
 * opponent-half view, regardless of who hit. The shot map is the canonical
 * "every bounce" surface. */
function buildShotMapDots(shots: ApiShot[]): ShotDot[] {
  const dots: ShotDot[] = [];
  for (const s of shots) {
    if (s.court_x == null || s.court_y == null) continue;
    let y = s.court_y;
    if (y > 39) {
      // Near-half bounce — mirror to top half so it shows up on this view.
      y = 78 - y;
    }
    // Allow a few units past the top baseline (OOB long) and a few units past
    // each sideline (OOB wide). Drop anything farther — likely tracking noise.
    if (y < -6 || y > 45) continue;
    if (s.court_x < -4 || s.court_x > 31) continue;
    dots.push({
      x: s.court_x,
      y,
      stroke: (s.stroke === 'unknown' ? 'unknown' : (s.stroke as StrokeKey)),
      in: s.in,
    });
  }
  return dots;
}

/** Derive spacing shots for the bottom-half spacing viz.
 *  Only includes shots with both player + ball at contact. Quality classified by
 *  distance: short = squeezed, mid = ideal, far = long. */
function buildSpacingShots(shots: ApiShot[]): SpacingShot[] {
  const out: SpacingShot[] = [];
  for (const s of shots) {
    if (s.stroke === 'unknown') continue; // spacing needs a stroke type for color
    if (
      s.player_court_x == null ||
      s.player_court_y == null ||
      s.ball_court_x == null ||
      s.ball_court_y == null
    )
      continue;
    let px = s.player_court_x;
    let py = s.player_court_y;
    let bx = s.ball_court_x;
    let by = s.ball_court_y;
    if (s.player === 2) {
      py = 78 - py;
      by = 78 - by;
    }
    if (py < 39 || by < 39) continue; // contact must be on player half
    const d = Math.hypot(bx - px, by - py);
    // 4-bin per coach-insights spec. 1 court unit ≈ 1 ft.
    //   jammed:    < 1 ft   — ball is on top of the player, no swing room
    //   squeezed:  1–2 ft   — close, but room to extend
    //   ideal:     2–3.5 ft — proper extension
    //   long:      ≥ 3.5 ft — reaching, lost balance
    const q: SpacingShot['q'] =
      d < 1.0 ? 'jammed' : d < 2.0 ? 'squeezed' : d < 3.5 ? 'ideal' : 'long';
    out.push({ stroke: s.stroke as StrokeKey, px, py, bx, by, q });
  }
  return out;
}

export default function VizPanel({ shots = [], coverageGrid, positionSummary, recordingStatus }: Props) {
  const [mode, setMode] = useState<VizMode>('shotMap');
  const [shotFilter, setShotFilter] = useState<StrokeKey | null>(null);
  const [spacingFilter, setSpacingFilter] = useState<StrokeKey | null>(null);

  const head = HEAD[mode];

  // Derive viz-shaped data from real shots; fall back to sample only when empty.
  const realDots = useMemo(() => buildShotMapDots(shots), [shots]);
  const realSpacing = useMemo(() => buildSpacingShots(shots), [shots]);
  const shotDots: ShotDot[] = realDots.length > 0 ? realDots : SAMPLE_SHOTS;
  const spacingShots: SpacingShot[] =
    realSpacing.length > 0 ? realSpacing : SAMPLE_SPACING;
  const usingReal = realDots.length > 0;

  // Coverage: real grid if backend gave us a 12x8 with any non-zero cell.
  const realCoverageGrid = useMemo(() => {
    if (!coverageGrid || coverageGrid.length === 0) return null;
    const hasSignal = coverageGrid.some((row) => row.some((v) => v > 0));
    return hasSignal ? coverageGrid : null;
  }, [coverageGrid]);
  const usingRealCoverage = realCoverageGrid !== null;

  // Real-data coverage insight: deepest hot zone + lateral bias
  const coverageInsight = useMemo(() => {
    if (!realCoverageGrid) return null;
    const rows = realCoverageGrid.length;
    const cols = realCoverageGrid[0]?.length ?? 0;
    if (rows === 0 || cols === 0) return null;
    let total = 0;
    let baselineHeavy = 0; // bottom third (last 4 rows)
    let leftHalf = 0;
    let rightHalf = 0;
    const baselineFromRow = Math.floor(rows * 0.66);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = realCoverageGrid[r][c];
        total += v;
        if (r >= baselineFromRow) baselineHeavy += v;
        if (c < cols / 2) leftHalf += v;
        else rightHalf += v;
      }
    }
    if (total <= 0) return null;
    const baselinePct = Math.round((baselineHeavy / total) * 100);
    const sideBias =
      Math.abs(leftHalf - rightHalf) / total > 0.1
        ? leftHalf > rightHalf
          ? 'ad side'
          : 'deuce side'
        : null;
    const sidePct = sideBias
      ? Math.round(
          (Math.max(leftHalf, rightHalf) / total) * 100,
        )
      : null;
    return { baselinePct, sideBias, sidePct };
  }, [realCoverageGrid]);

  // Insight numbers — real when we have data, else fall back to the mock copy.
  const shotInsight = useMemo(() => {
    if (!usingReal) return null;
    const inN = shots.filter((s) => s.in).length;
    const outN = shots.filter((s) => !s.in).length;
    const total = inN + outN;
    const accuracy = total > 0 ? Math.round((inN / total) * 100) : 0;
    return { inN, outN, accuracy };
  }, [shots, usingReal]);

  // Read from the SAME array that feeds the map + legend so the three never
  // disagree. Previous version read realSpacing only, so when zero real shots
  // survived filtering and the map fell back to sample data, the insight kept
  // showing "0 / 0 / 0" while the map + chips showed mock numbers.
  const spacingInsight = useMemo(() => {
    const jammed = spacingShots.filter((s) => s.q === 'jammed').length;
    const squeezed = spacingShots.filter((s) => s.q === 'squeezed').length;
    const ideal = spacingShots.filter((s) => s.q === 'ideal').length;
    const long = spacingShots.filter((s) => s.q === 'long').length;
    return { jammed, squeezed, ideal, long };
  }, [spacingShots]);

  return (
    <div
      className="cc-card bg-paper border border-line rounded-[14px] mb-8"
      style={{ padding: '26px 30px' }}
    >
      <div className="mb-3.5">
        <span className="inline-flex items-center gap-2 font-mono text-[0.72rem] uppercase tracking-[0.18em] text-court before:content-[''] before:w-1.5 before:h-1.5 before:bg-clay before:rounded-full">
          {head.eyebrow}
        </span>
        <h3 className="font-display font-medium text-[1.25rem] tracking-tight mt-3">
          {head.title}
        </h3>
        <div className="text-ink-soft text-[0.95rem] mt-1">{head.sub}</div>

        <div
          className="inline-flex gap-1 p-1 bg-shade dark:bg-surface border border-line-soft rounded-[10px] mt-3.5"
          role="tablist"
          aria-label="Court visualization mode"
        >
          {MODES.map(({ key, label }) => {
            const active = mode === key;
            return (
              <button
                key={key}
                type="button"
                role="tab"
                aria-selected={active}
                onClick={() => setMode(key)}
                className={`appearance-none border font-mono text-[0.72rem] uppercase tracking-[0.1em] px-3.5 py-2 rounded-[7px] cursor-pointer ${
                  active
                    ? 'bg-ink text-cream border-ink dark:bg-court dark:border-court'
                    : 'bg-transparent text-ink-soft border-transparent hover:text-ink hover:border-line'
                }`}
                style={{ transition: 'background var(--duration-base) var(--ease-out), color var(--duration-base) var(--ease-out), border-color var(--duration-base) var(--ease-out)' }}
              >
                {label}
              </button>
            );
          })}
        </div>
      </div>

      <div className="cc-viz-grid mt-2">
        <div className="min-w-0">
          <div
            className="cc-court-tile mx-auto"
            style={{
              maxHeight: '56vh',
              maxWidth: '100%',
              width: 'auto',
              // ShotMap extends ~4 units past the opponent baseline + ~2 units
              // past each sideline so OOB bounces render — viewBox grows to
              // 31/43. Spacing + Coverage extend ~10 units below the baseline
              // so players standing behind the line render — viewBox 27/49.
              aspectRatio: mode === 'shotMap' ? '31 / 43' : '27 / 49',
              marginTop: 4,
            }}
          >
            {mode === 'shotMap' && (
              <ShotMap dots={shotDots} activeFilter={shotFilter} />
            )}
            {mode === 'spacing' && (
              <Spacing shots={spacingShots} activeFilter={spacingFilter} />
            )}
            {mode === 'coverage' &&
              (recordingStatus === 'done' && !usingRealCoverage ? (
                <CoverageEmpty />
              ) : (
                <Coverage grid={realCoverageGrid ?? undefined} />
              ))}
          </div>
          <div className="text-center font-mono text-[0.66rem] uppercase tracking-[0.14em] text-ink-mute mt-2">
            {head.halfLabel}
            {(() => {
              // What "is sample" means per mode:
              //   shotMap: no real bounces produced any dots
              //   spacing: no real bounces had valid player+ball contact coords
              //   coverage: backend didn't produce a non-empty coverage_grid
              const isSample =
                mode === 'shotMap'
                  ? !usingReal
                  : mode === 'spacing'
                    ? realSpacing.length === 0
                    : !usingRealCoverage;
              if (mode === 'coverage' && recordingStatus === 'done' && !usingRealCoverage) {
                return (
                  <span className="ml-2 normal-case tracking-normal text-clay">
                    · <em>no coverage data</em>
                  </span>
                );
              }
              if (isSample) {
                return (
                  <span className="ml-2 normal-case tracking-normal text-clay">
                    · <em>sample data</em>
                  </span>
                );
              }
              return null;
            })()}
          </div>
        </div>

        <div className="cc-viz-side min-w-0 flex flex-col gap-5">
          {mode === 'shotMap' && (
            <>
              <Legend
                counts={shotMapCounts(shotDots)}
                unknownCount={usingReal ? shotMapUnknownCount(shotDots) : 0}
                activeKey={shotFilter}
                onToggle={(k) => setShotFilter((prev) => (prev === k ? null : k))}
              />
              <ShotInOutKey />
              <CoachingInsight>
                {shotInsight ? (
                  <>
                    <strong>{shotInsight.inN}</strong> in,{' '}
                    <strong>{shotInsight.outN}</strong> out.{' '}
                    <strong>{shotInsight.accuracy}%</strong> accuracy. Slice the
                    count by stroke to see the bias.
                  </>
                ) : (
                  <>
                    <strong>24</strong> in, <strong>4</strong> near-miss.{' '}
                    <strong>87%</strong> accuracy. Slice the count by stroke to
                    see the bias.
                  </>
                )}
              </CoachingInsight>
            </>
          )}

          {mode === 'spacing' && (
            <>
              <Legend
                counts={spacingCounts(spacingShots)}
                activeKey={spacingFilter}
                onToggle={(k) => setSpacingFilter((prev) => (prev === k ? null : k))}
              />
              <QualityLegend />
              <MarkerLegend />
              <CoachingInsight>
                {spacingInsight ? (
                  <>
                    <strong>{spacingInsight.ideal}</strong> shots at ideal
                    extension. <strong>{spacingInsight.squeezed}</strong>{' '}
                    squeezed,{' '}
                    <strong>{spacingInsight.jammed}</strong> jammed (ball on
                    top), <strong>{spacingInsight.long}</strong> reaching.
                  </>
                ) : (
                  <>
                    <strong>23</strong> shots at ideal extension.{' '}
                    <strong>8</strong> squeezed, <strong>2</strong> jammed
                    (ball on top), <strong>4</strong> reaching. Step back a
                    half-step on returns to clean up the squeezed shots.
                  </>
                )}
              </CoachingInsight>
            </>
          )}

          {mode === 'coverage' && (
            <>
              <div className="text-[0.88rem] text-ink-soft leading-relaxed">
                Density of where she stood over the recording. Brighter zones =
                more time spent at that position.
              </div>
              {positionSummary && positionSummary.n_frames > 0 && (
                <PositionTile data={positionSummary} />
              )}
              <CoachingInsight>
                {coverageInsight ? (
                  <>
                    <strong>{coverageInsight.baselinePct}%</strong> of the
                    recording was spent at the baseline third of the court.
                    {coverageInsight.sideBias && coverageInsight.sidePct && (
                      <>
                        {' '}
                        Lateral bias toward the{' '}
                        <strong>{coverageInsight.sideBias}</strong> (
                        <strong>{coverageInsight.sidePct}%</strong> of time).
                      </>
                    )}
                  </>
                ) : (
                  <>
                    <strong>82%</strong> of your recording was within 2 feet of
                    the baseline. Only <strong>9%</strong> inside the service
                    boxes. Stepping in on second-serve returns could shorten
                    points.
                  </>
                )}
              </CoachingInsight>
            </>
          )}
        </div>
      </div>

      <style>{`
        .cc-viz-grid {
          display: grid;
          gap: 28px;
          grid-template-columns: minmax(0, 1.4fr) minmax(0, 1fr);
        }
        .cc-viz-side {
          padding-left: 24px;
          border-left: 1px solid var(--color-line-soft);
        }
        @media (max-width: 900px) {
          .cc-viz-grid {
            grid-template-columns: 1fr;
            gap: 22px;
          }
          .cc-viz-side {
            padding-left: 0;
            border-left: none;
            border-top: 1px solid var(--color-line-soft);
            padding-top: 22px;
          }
        }
      `}</style>
    </div>
  );
}

function CoachingInsight({ children }: { children: React.ReactNode }) {
  return (
    <div
      className="rounded-lg flex gap-3 items-start text-[0.92rem] leading-relaxed text-ink-soft"
      style={{
        padding: '14px 16px',
        background: 'color-mix(in srgb, var(--color-court) 6%, transparent)',
      }}
    >
      <span className="text-clay font-display italic font-semibold shrink-0">→</span>
      <span className="[&_strong]:text-ink [&_strong]:font-display [&_strong]:font-medium [&_strong]:tracking-[-0.005em]" style={{ fontFeatureSettings: '"tnum"' }}>
        {children}
      </span>
    </div>
  );
}

function QualityLegend() {
  return (
    <div
      className="flex flex-wrap gap-x-6 gap-y-2.5 pt-3.5 text-[0.82rem] text-ink-soft"
      style={{ borderTop: '1px solid var(--color-line-soft)' }}
    >
      <QualityKey color="var(--color-plum)" label="Jammed (< 1 ft)" width={10} />
      <QualityKey color="var(--color-clay)" label="Squeezed" width={16} />
      <QualityKey color="var(--color-court)" label="Ideal" width={28} />
      <QualityKey color="var(--color-amber)" label="Long (reaching)" width={42} />
    </div>
  );
}

function QualityKey({ color, label, width }: { color: string; label: string; width: number }) {
  return (
    <span className="inline-flex items-center gap-2.5 font-medium">
      <span className="inline-flex items-center">
        <svg width={width} height={8} viewBox={`0 0 ${width} 8`} xmlns="http://www.w3.org/2000/svg">
          <circle cx={2} cy={4} r={1.6} fill="white" stroke={color} strokeWidth={0.7} />
          <line x1={3} y1={4} x2={width - 3} y2={4} stroke={color} strokeWidth={1.6} strokeLinecap="round" />
          <circle cx={width - 2} cy={4} r={1.4} fill="var(--color-lime)" stroke={color} strokeWidth={0.7} />
        </svg>
      </span>
      {label}
    </span>
  );
}

function ShotInOutKey() {
  return (
    <div
      className="flex flex-wrap gap-x-5 font-mono text-[0.72rem] uppercase tracking-[0.18em] text-ink-soft"
      aria-label="Bounce in/out key"
    >
      <span className="inline-flex items-center gap-2">
        <span
          className="inline-block w-2.5 h-2.5 rounded-full shrink-0"
          style={{ background: 'var(--color-court)' }}
          aria-hidden
        />
        In
      </span>
      <span className="inline-flex items-center gap-2">
        <svg width={12} height={12} viewBox="0 0 12 12" aria-hidden>
          <line x1={2} y1={2} x2={10} y2={10} stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" />
          <line x1={2} y1={10} x2={10} y2={2} stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" />
        </svg>
        Out
      </span>
    </div>
  );
}

function MarkerLegend() {
  return (
    <div
      className="flex flex-wrap gap-x-5 font-mono text-[0.72rem] uppercase tracking-[0.18em] text-ink-soft"
      aria-label="Marker key"
    >
      <span className="inline-flex items-center gap-2">
        <span
          className="inline-block w-2.5 h-2.5 rounded-full shrink-0"
          style={{ background: 'var(--color-paper)', border: '1.2px solid var(--color-ink-soft)' }}
          aria-hidden
        />
        Player
      </span>
      <span className="inline-flex items-center gap-2">
        <span
          className="inline-block w-2.5 h-2.5 rounded-full shrink-0"
          style={{ background: 'var(--color-lime)', border: '1.2px solid var(--color-ink-soft)' }}
          aria-hidden
        />
        Ball
      </span>
    </div>
  );
}

/**
 * Empty state shown when a recording is `done` but coverage_grid is empty —
 * means the pipeline ran before the coverage_grid migration was applied (or
 * before the build_coverage_grid helper was deployed). Tells coach to reprocess.
 */
function CoverageEmpty() {
  return (
    <div className="flex h-full w-full flex-col items-center justify-center gap-3 px-6 py-10 text-center">
      <svg
        viewBox="0 0 24 24"
        width={36}
        height={36}
        fill="none"
        stroke="currentColor"
        strokeWidth={1.4}
        strokeLinecap="round"
        strokeLinejoin="round"
        className="text-ink-mute"
        aria-hidden
      >
        <rect x="3" y="5" width="18" height="14" rx="2" />
        <line x1="3" y1="12" x2="21" y2="12" />
        <line x1="12" y1="5" x2="12" y2="19" />
      </svg>
      <p className="font-display text-[1.1rem] text-ink leading-snug">
        No coverage <em>data</em> for this recording.
      </p>
      <p className="max-w-[300px] text-[0.85rem] text-ink-soft leading-relaxed">
        Coverage came online after this recording was processed. Reprocess it
        to generate the heatmap.
      </p>
    </div>
  );
}
