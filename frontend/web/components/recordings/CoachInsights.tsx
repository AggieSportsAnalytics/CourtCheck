'use client';

import { Prose } from '@/components/ui/display';

import { RefObject } from 'react';
import { useEntranceReveal } from '../viz/useEntranceReveal';
import CountUp from '@/components/ui/CountUp';

/**
 * Coach Insights — three metrics surfaced per `coach-insights-spec.md`:
 *   1. Court Position (4-zone baseline distance — Coach Jackson's #1 ask)
 *   2. Net Game (approach count + heuristic win rate)
 *   3. Errors (direction breakdown: long / wide / net)
 *
 * Per-player split is intentionally not rendered (Brian's call). FH/BH overlay
 * on errors deferred until stroke classifier consistently labels rallies.
 *
 * Every metric exposes a "show me" timestamp list that seeks the video via
 * the shared videoRef.
 */

export type PositionSummary = {
  inside_pct: number;
  on_pct: number;
  behind_5_10_pct: number;
  behind_10_plus_pct: number;
  n_frames: number;
};

export type NetApproachEvent = {
  frame: number;
  time_s: number;
  outcome: 'won' | 'lost';
};

export type NetApproachSummary = {
  approaches: number;
  wins: number;
  win_pct: number;
  events: NetApproachEvent[];
};

export type ErrorEvent = {
  frame: number;
  time_s: number;
  /** 'missed return' = opponent's ball landed in-bounds and P1 didn't swing
   *  at it. The other three are P1's own shots that landed out. */
  direction: 'long' | 'wide' | 'net' | 'missed return';
  court_x: number;
  court_y: number;
  /** 1 = near (P1) — only one we attribute today. Future-proofed for a P2 split. */
  player?: 1 | 2;
};

export type ErrorSummary = {
  total: number;
  long: number;
  wide: number;
  net_err: number;
  /** Bounces where opponent's ball landed in your half and you didn't swing
   *  within ~1.5s. Distinct from the 3 OOB buckets above (those are your
   *  shots that missed). */
  missed_return: number;
  events: ErrorEvent[];
};

type Props = {
  netApproach: NetApproachSummary | null;
  errors: ErrorSummary | null;
  videoRef: RefObject<HTMLVideoElement | null>;
};

export default function CoachInsights({ netApproach, errors, videoRef }: Props) {
  // Court Position now lives inside the Coverage tab in VizPanel (a "where she
  // stood" pair with the heatmap), so this panel surfaces just the two
  // actionable timestamps coaches click into: errors and net game.
  const hasAny = Boolean(netApproach || errors);
  if (!hasAny) return null;

  return (
    <div
      className="cc-card bg-paper border border-line rounded-[14px] mb-8"
      style={{ padding: '26px 30px' }}
    >
      <div className="mb-3.5">
        <span className="inline-flex items-center gap-2 font-mono text-[0.82rem] uppercase tracking-[0.12em] text-court before:content-[''] before:w-1.5 before:h-1.5 before:bg-clay before:rounded-full">
          Coaching notes
        </span>
        <h3 className="font-display font-medium text-[1.25rem] tracking-tight mt-3">
          Errors and net play
        </h3>
        <div className="text-ink-soft text-[1.02rem] mt-1">
          Tap a timestamp to jump to it in the video.
        </div>
      </div>

      <div className="cc-coach-grid mt-5">
        <ErrorTile data={errors} videoRef={videoRef} />
        <NetApproachTile data={netApproach} videoRef={videoRef} />
      </div>

      <style>{`
        .cc-coach-grid {
          display: grid;
          gap: 24px;
          grid-template-columns: 1fr 1fr;
        }
        @media (max-width: 980px) {
          .cc-coach-grid { grid-template-columns: 1fr; gap: 28px; }
        }
      `}</style>
    </div>
  );
}

function TileHead({ eyebrow, headline }: { eyebrow: string; headline: string }) {
  return (
    <div className="mb-3">
      <div className="font-mono text-[0.82rem] uppercase tracking-[0.12em] text-ink-mute">
        {eyebrow}
      </div>
      <div className="font-display font-medium text-[1.05rem] tracking-tight mt-1.5">
        {headline}
      </div>
    </div>
  );
}

export function PositionTile({ data }: { data: PositionSummary | null }) {
  // useEntranceReveal MUST be called before any early return so it runs on
  // every render path (React rules of hooks).
  const shown = useEntranceReveal(data?.n_frames ?? 0);
  if (!data || data.n_frames === 0) {
    return (
      <div className="cc-coach-tile">
        <TileHead eyebrow="Court position" headline="–" />
        <Prose className="text-ink-mute">No position data.</Prose>
      </div>
    );
  }
  const zones: { label: string; pct: number; color: string }[] = [
    { label: 'Inside baseline', pct: data.inside_pct, color: 'var(--color-court)' },
    { label: 'On baseline', pct: data.on_pct, color: 'var(--color-court-light, var(--color-court))' },
    { label: '5–10 ft behind', pct: data.behind_5_10_pct, color: 'var(--color-amber)' },
    { label: '10+ ft behind', pct: data.behind_10_plus_pct, color: 'var(--color-clay)' },
  ];
  const headline = (() => {
    const dominant = [...zones].sort((a, b) => b.pct - a.pct)[0];
    return `${Math.round(dominant.pct)}% ${dominant.label.toLowerCase()}.`;
  })();
  return (
    <div className="cc-coach-tile">
      <TileHead eyebrow="Court position" headline={headline} />
      <div className="space-y-1.5">
        {zones.map((z, i) => (
          <div
            key={z.label}
            className="grid items-center"
            style={{ gridTemplateColumns: '128px 1fr 42px', gap: 10 }}
          >
            <span className="text-[0.88rem] text-ink-soft">{z.label}</span>
            <div className="h-2 rounded-full overflow-hidden bg-shade">
              <div
                className="h-full rounded-full"
                style={{
                  width: shown ? `${Math.min(100, Math.max(0, z.pct))}%` : '0%',
                  background: z.color,
                  // 100ms-per-row stagger keeps the entrance lively without
                  // feeling like a queue.
                  transition: `width 720ms cubic-bezier(0.165, 0.84, 0.44, 1) ${i * 100}ms`,
                }}
              />
            </div>
            <span
              className="text-right text-[0.88rem] font-display font-medium text-ink"
              style={{ fontFeatureSettings: '"tnum"' }}
            >
              <CountUp value={z.pct} play={shown} suffix="%" />
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function NetApproachTile({
  data,
  videoRef,
}: {
  data: NetApproachSummary | null;
  videoRef: RefObject<HTMLVideoElement | null>;
}) {
  if (!data || data.approaches === 0) {
    return (
      <div className="cc-coach-tile">
        <TileHead eyebrow="Net game" headline={data ? 'No net approaches detected.' : 'No net approach data.'} />
        <Prose className="text-ink-mute mt-1">
          {data
            ? 'The tracker did not identify a net approach in this recording.'
            : 'Reprocess this recording to generate net approach data.'}
        </Prose>
      </div>
    );
  }
  return (
    <div className="cc-coach-tile">
      <TileHead
        eyebrow="Net game"
        headline={`${data.approaches} approached · ${data.wins} won (${Math.round(data.win_pct)}%).`}
      />
      <div className="text-[0.88rem] text-ink-mute mb-2">
        Win rate is a heuristic: the rally ended within 3 seconds with the opponent's ball out.
      </div>
      <TimestampList
        items={data.events.map((e) => ({
          frame: e.frame,
          time_s: e.time_s,
          label: e.outcome === 'won' ? 'Won' : 'Lost',
          tone: e.outcome === 'won' ? 'win' : 'loss',
        }))}
        videoRef={videoRef}
        emptyLabel="No events"
      />
    </div>
  );
}

function ErrorTile({
  data,
  videoRef,
}: {
  data: ErrorSummary | null;
  videoRef: RefObject<HTMLVideoElement | null>;
}) {
  // Hook before any early-return so it runs on every render.
  const shown = useEntranceReveal(data?.total ?? 0);
  if (!data || data.total === 0) {
    return (
      <div className="cc-coach-tile">
        <TileHead eyebrow="Your errors" headline="No out-of-bounds bounces." />
      </div>
    );
  }
  const missed = data.missed_return ?? 0;
  const oob = data.long + data.wide + data.net_err;
  const rows: { label: string; n: number; color: string }[] = [
    { label: 'Unreturned balls', n: missed, color: 'var(--color-court)' },
    { label: 'Long', n: data.long, color: 'var(--color-amber)' },
    { label: 'Wide', n: data.wide, color: 'var(--color-plum)' },
    { label: 'Net', n: data.net_err, color: 'var(--color-clay)' },
  ];
  const total = Math.max(1, data.total);
  return (
    <div className="cc-coach-tile">
      <TileHead
        eyebrow="Your errors"
        headline={`${data.total} error${data.total === 1 ? '' : 's'}: ${missed} unreturned, ${oob} out.`}
      />
      <Prose className="text-ink-soft mb-3 -mt-1.5">
        An unreturned ball landed in the player's half without a swing. This includes the opponent's placements as well as the player's errors.
      </Prose>
      <div className="space-y-1.5 mb-3">
        {rows.map((r, i) => (
          <div
            key={r.label}
            className="grid items-center"
            style={{ gridTemplateColumns: '112px 1fr 28px', gap: 10 }}
          >
            <span className="text-[0.88rem] text-ink-soft">{r.label}</span>
            <div className="h-2 rounded-full overflow-hidden bg-shade">
              <div
                className="h-full rounded-full"
                style={{
                  width: shown ? `${(r.n / total) * 100}%` : '0%',
                  background: r.color,
                  transition: `width 720ms cubic-bezier(0.165, 0.84, 0.44, 1) ${i * 100}ms`,
                }}
              />
            </div>
            <span
              className="text-right text-[0.88rem] font-display font-medium text-ink"
              style={{ fontFeatureSettings: '"tnum"' }}
            >
              <CountUp value={r.n} play={shown} />
            </span>
          </div>
        ))}
      </div>
      <TimestampList
        items={data.events.map((e) => ({
          frame: e.frame,
          time_s: e.time_s,
          label: e.direction === 'missed return'
            ? 'Unreturned balls'
            : e.direction[0].toUpperCase() + e.direction.slice(1),
          tone: 'neutral',
        }))}
        videoRef={videoRef}
        emptyLabel="No errors"
      />
    </div>
  );
}

function TimestampList({
  items,
  videoRef,
  emptyLabel,
}: {
  items: { frame: number; time_s: number; label: string; tone: 'win' | 'loss' | 'neutral' }[];
  videoRef: RefObject<HTMLVideoElement | null>;
  emptyLabel: string;
}) {
  if (items.length === 0) {
    return <Prose className="text-ink-mute">{emptyLabel}</Prose>;
  }
  return (
    <div className="flex flex-wrap gap-1.5">
      {items.slice(0, 14).map((it, i) => {
        const m = Math.floor(it.time_s / 60);
        const s = Math.floor(it.time_s % 60);
        const ts = `${m}:${s.toString().padStart(2, '0')}`;
        const toneClass =
          it.tone === 'win'
            ? 'cc-ts-win'
            : it.tone === 'loss'
              ? 'cc-ts-loss'
              : 'cc-ts-neutral';
        return (
          <button
            key={`${it.frame}-${i}`}
            type="button"
            onClick={() => {
              const v = videoRef.current;
              if (!v) return;
              v.currentTime = it.time_s;
              if (v.paused) void v.play();
            }}
            className={`cc-ts-chip ${toneClass}`}
            title={`${it.label} · jump to ${ts}`}
          >
            <span
              className="font-mono text-[0.82rem] tabular-nums"
              style={{ fontFeatureSettings: '"tnum"' }}
            >
              {ts}
            </span>
            <span className="text-[0.82rem]">{it.label}</span>
          </button>
        );
      })}
      {/* .cc-coach-tile + .cc-ts-chip rules live in globals.css so PositionTile
          can be reused outside this card (Coverage rail in VizPanel). */}
    </div>
  );
}
