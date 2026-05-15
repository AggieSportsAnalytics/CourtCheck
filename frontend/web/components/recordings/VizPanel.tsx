'use client';

import { useEffect, useMemo, useState, type RefObject } from 'react';
import ShotMap, { SAMPLE_SHOTS, shotMapCounts, shotMapUnknownCount, type ShotDot } from '../viz/ShotMap';
import Spacing, { SAMPLE_SPACING, spacingCounts, type SpacingShot } from '../viz/Spacing';
import Coverage from '../viz/Coverage';
import Legend from '../viz/Legend';
import { StrokeKey, STROKE_COLOR_BY_KEY } from '../viz/CourtSVG';
import { PositionTile, type PositionSummary } from './CoachInsights';
import CountUp from '../viz/CountUp';
import { useEntranceReveal } from '../viz/useEntranceReveal';

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
  /** Optional. When provided, clicking a bounce on the shot map opens a
   *  side-panel and "Play from here" seeks the video. */
  videoRef?: RefObject<HTMLVideoElement | null>;
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
      time_s: s.time_s,
      frame: s.frame,
      player: s.player,
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

export default function VizPanel({ shots = [], coverageGrid, positionSummary, recordingStatus, videoRef }: Props) {
  const [mode, setMode] = useState<VizMode>('shotMap');
  const [shotFilter, setShotFilter] = useState<StrokeKey | null>(null);
  const [spacingFilter, setSpacingFilter] = useState<StrokeKey | null>(null);
  const [selectedShot, setSelectedShot] = useState<ShotDot | null>(null);

  // Click-anywhere-off close: when a bounce is selected, any click whose
  // target isn't a `.shot-dot` (the markers themselves) and isn't inside
  // a `[data-bounce-panel]` (the inline detail card) clears the selection.
  // Keeps the interaction tight without an explicit close button.
  useEffect(() => {
    if (!selectedShot) return;
    const onDocClick = (e: MouseEvent) => {
      const target = e.target as HTMLElement | null;
      if (!target) return;
      if (target.closest('.shot-dot')) return;
      if (target.closest('[data-bounce-panel]')) return;
      setSelectedShot(null);
    };
    // Listen on capture so we run before any in-component handlers can
    // stop propagation.
    document.addEventListener('click', onDocClick, true);
    return () => document.removeEventListener('click', onDocClick, true);
  }, [selectedShot]);

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

  // Insight numbers — count from the SAME filtered set the visual renders
  // (realDots), not the raw `shots` array. Previously the stat said "7 in 6
  // out" while the map only showed 1 out, because buildShotMapDots drops
  // shots without court coords + shots that landed past the wide-OOB cutoff.
  const shotInsight = useMemo(() => {
    if (!usingReal) return null;
    const inN = realDots.filter((d) => d.in).length;
    const outN = realDots.filter((d) => !d.in).length;
    const total = inN + outN;
    const accuracy = total > 0 ? Math.round((inN / total) * 100) : 0;
    return { inN, outN, accuracy };
  }, [realDots, usingReal]);

  // Per-stroke mix + accuracy — REAL numbers from realDots, computed once
  // and shown together in the rail. Replaces both the redundant
  // in/out/accuracy text AND the standalone ShotBreakdown card; the rail
  // is now the single source of truth for stroke-distribution stats.
  const shotByStroke = useMemo(() => {
    if (!usingReal) return null;
    const total = realDots.length;
    const compute = (key: StrokeKey) => {
      const subset = realDots.filter((d) => d.stroke === key);
      const inN = subset.filter((d) => d.in).length;
      const n = subset.length;
      return {
        key,
        n,
        mixPct: total > 0 ? Math.round((n / total) * 100) : 0,
        accPct: n > 0 ? Math.round((inN / n) * 100) : null,
      };
    };
    return [compute('forehand'), compute('backhand'), compute('serve')];
  }, [realDots, usingReal]);

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
              <ShotMap
                dots={shotDots}
                activeFilter={shotFilter}
                onSelect={usingReal ? (dot) => setSelectedShot(dot) : undefined}
                selectedFrame={selectedShot?.frame ?? null}
              />
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
              <StrokeBarsMini
                eyebrow="Mix"
                metricKey="mixPct"
                byStroke={shotByStroke}
                overallText={
                  shotInsight
                    ? `${shotInsight.inN + shotInsight.outN} shots tracked`
                    : null
                }
              />
              <StrokeBarsMini
                eyebrow="Accuracy"
                metricKey="accPct"
                byStroke={shotByStroke}
                overallText={
                  shotInsight
                    ? `${shotInsight.accuracy}% overall · ${shotInsight.inN} in, ${shotInsight.outN} out`
                    : null
                }
              />
              <BouncePanel
                shot={selectedShot}
                onClose={() => setSelectedShot(null)}
                videoRef={videoRef}
              />
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

/** Inline panel rendered in the right-side rail of the shot map view, under
 *  the existing in/out/accuracy insight. Shows the selected bounce's
 *  metadata + a "Play from here" button that seeks the parent <video>.
 *  Renders nothing when no shot is selected. */
function BouncePanel({
  shot,
  onClose,
  videoRef,
}: {
  shot: ShotDot | null;
  onClose: () => void;
  videoRef?: RefObject<HTMLVideoElement | null>;
}) {
  useEffect(() => {
    if (!shot) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [shot, onClose]);

  if (!shot) return null;

  const isUnknown = shot.stroke === 'unknown';
  const strokeLabel = isUnknown
    ? 'Unclassified'
    : shot.stroke === 'forehand'
      ? 'Forehand'
      : shot.stroke === 'backhand'
        ? 'Backhand'
        : 'Serve / Overhead';
  const strokeColor = isUnknown
    ? 'var(--color-ink-mute)'
    : STROKE_COLOR_BY_KEY[shot.stroke as StrokeKey];
  const isOut = shot.in === false;
  const sec = Math.max(0, shot.time_s ?? 0);
  const mm = Math.floor(sec / 60).toString().padStart(2, '0');
  const ss = Math.floor(sec % 60).toString().padStart(2, '0');
  const ts = `${mm}:${ss}`;

  const playFromHere = () => {
    const v = videoRef?.current;
    if (!v) return;
    v.currentTime = sec;
    void v.play();
    // Scroll the video into view so the user sees the playback they just
    // jumped to — they're scrolled down at the shot map otherwise. center
    // the video vertically; smooth so it doesn't feel jarring.
    v.scrollIntoView({ behavior: 'smooth', block: 'center' });
    onClose();
  };

  return (
    <div
      role="region"
      aria-label="Selected bounce detail"
      data-bounce-panel
      className="rounded-lg border border-line bg-paper"
      style={{
        padding: '14px 16px',
        display: 'flex',
        flexDirection: 'column',
        gap: 12,
        animation: 'cc-bounce-panel-in 200ms cubic-bezier(.2,.7,.2,1)',
      }}
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <span className="font-mono text-[0.66rem] uppercase tracking-[0.18em] text-ink-mute">
            Selected bounce
          </span>
          <p
            className="font-display font-medium leading-none mt-1"
            style={{ fontFeatureSettings: '"tnum"', fontSize: '1.4rem' }}
          >
            {ts}
          </p>
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="Clear selection"
          className="text-ink-mute hover:text-ink transition-colors"
          style={{ padding: 2, lineHeight: 0 }}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </div>

      <div className="flex flex-wrap gap-1.5">
        <span
          className="inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[0.7rem] font-mono uppercase tracking-[0.12em]"
          style={{
            background: `color-mix(in srgb, ${strokeColor} 14%, var(--color-paper))`,
            color: strokeColor,
            border: `1px solid color-mix(in srgb, ${strokeColor} 35%, var(--color-line))`,
          }}
        >
          <span style={{ width: 5, height: 5, borderRadius: '50%', background: strokeColor }} aria-hidden />
          {strokeLabel}
        </span>
        <span
          className="inline-flex items-center rounded-full px-2 py-0.5 text-[0.7rem] font-mono uppercase tracking-[0.12em]"
          style={{
            background: isOut
              ? 'color-mix(in srgb, var(--color-clay) 12%, var(--color-paper))'
              : 'color-mix(in srgb, var(--color-court) 10%, var(--color-paper))',
            color: isOut ? 'var(--color-clay)' : 'var(--color-court)',
            border: `1px solid ${isOut ? 'color-mix(in srgb, var(--color-clay) 35%, var(--color-line))' : 'color-mix(in srgb, var(--color-court) 35%, var(--color-line))'}`,
          }}
        >
          {isOut ? 'Out' : 'In'}
        </span>
      </div>

      <button
        type="button"
        onClick={playFromHere}
        disabled={!videoRef?.current}
        className="inline-flex items-center justify-center gap-2 rounded-full bg-ink text-cream px-4 py-2 text-[0.85rem] font-medium transition-transform duration-150 ease-out hover:-translate-y-px disabled:opacity-50 disabled:cursor-not-allowed dark:bg-court-deep"
      >
        <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
          <path d="M5 4l14 8-14 8V4z" />
        </svg>
        Play from here
      </button>

      <style>{`
        @keyframes cc-bounce-panel-in {
          from { transform: translateY(-4px); opacity: 0; }
          to   { transform: translateY(0);    opacity: 1; }
        }
        @media (prefers-reduced-motion: reduce) {
          @keyframes cc-bounce-panel-in {
            from, to { transform: translateY(0); opacity: 1; }
          }
        }
      `}</style>
    </div>
  );
}

/** One-metric mini-chart for the Shot Map right rail. Renders 3 rows
 *  (Forehand / Backhand / Serve), each with a stroke-colored bar + the
 *  metric value + the per-stroke count. Used twice — once for Mix, once
 *  for Accuracy — so the two charts stay visually parallel without
 *  cramming both metrics into the same row. */
function StrokeBarsMini({
  eyebrow,
  metricKey,
  byStroke,
  overallText,
}: {
  eyebrow: string;
  metricKey: 'mixPct' | 'accPct';
  byStroke:
    | {
        key: StrokeKey;
        n: number;
        mixPct: number;
        accPct: number | null;
      }[]
    | null;
  overallText: string | null;
}) {
  // Hook before any early-return so it runs on every render path.
  const shown = useEntranceReveal(byStroke);
  if (!byStroke || overallText === null) {
    return (
      <div
        className="rounded-lg text-[0.85rem] leading-snug text-ink-soft"
        style={{
          padding: '12px 14px',
          background: 'color-mix(in srgb, var(--color-court) 6%, transparent)',
        }}
      >
        <span className="font-mono text-[0.66rem] uppercase tracking-[0.18em] text-ink-mute mr-2">
          {eyebrow}
        </span>
        Unlocks once the recording finishes processing.
      </div>
    );
  }

  return (
    <div
      className="cc-strokebars rounded-lg"
      style={{
        padding: '12px 14px',
        background: 'color-mix(in srgb, var(--color-court) 6%, transparent)',
        display: 'flex',
        flexDirection: 'column',
        gap: 10,
      }}
    >
      <div className="flex items-baseline justify-between gap-3">
        <span className="font-mono text-[0.66rem] uppercase tracking-[0.18em] text-ink-mute">
          {eyebrow}
        </span>
        <span className="text-[0.74rem] text-ink-soft">{overallText}</span>
      </div>

      <div className="space-y-1.5">
        {byStroke.map((row) => {
          const color = STROKE_COLOR_BY_KEY[row.key];
          const label =
            row.key === 'forehand'
              ? 'Forehand'
              : row.key === 'backhand'
                ? 'Backhand'
                : 'Serve';
          const value = row[metricKey];
          return (
            <div
              key={row.key}
              className="grid items-center"
              style={{ gridTemplateColumns: '76px 1fr 52px', gap: 8 }}
            >
              <span className="text-[0.78rem] text-ink-soft truncate">{label}</span>
              <div className="h-2 rounded-full overflow-hidden bg-shade dark:bg-surface">
                {value !== null && (
                  <div
                    className="h-full rounded-full"
                    style={{
                      width: shown ? `${value}%` : '0%',
                      background: color,
                      transition: 'width 720ms cubic-bezier(0.165, 0.84, 0.44, 1)',
                    }}
                  />
                )}
              </div>
              <span
                className="text-right text-[0.78rem] font-display font-medium text-ink"
                style={{ fontFeatureSettings: '"tnum"' }}
              >
                {value !== null ? (
                  <CountUp value={value} play={shown} suffix="%" />
                ) : (
                  '—'
                )}
                <span className="text-ink-mute font-normal text-[0.7rem] ml-1">
                  ({row.n})
                </span>
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
