'use client';

import { RefObject } from 'react';

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
  direction: 'long' | 'wide' | 'net';
  court_x: number;
  court_y: number;
};

export type ErrorSummary = {
  total: number;
  long: number;
  wide: number;
  net_err: number;
  events: ErrorEvent[];
};

type Props = {
  position: PositionSummary | null;
  netApproach: NetApproachSummary | null;
  errors: ErrorSummary | null;
  videoRef: RefObject<HTMLVideoElement | null>;
};

export default function CoachInsights({ position, netApproach, errors, videoRef }: Props) {
  const hasAny = Boolean(position || netApproach || errors);
  if (!hasAny) return null;

  return (
    <div
      className="cc-card bg-paper border border-line rounded-[14px] mb-8"
      style={{ padding: '26px 30px' }}
    >
      <div className="mb-3.5">
        <span className="inline-flex items-center gap-2 font-mono text-[0.72rem] uppercase tracking-[0.18em] text-court before:content-[''] before:w-1.5 before:h-1.5 before:bg-clay before:rounded-full">
          Coach insights · this recording
        </span>
        <h3 className="font-display font-medium text-[1.25rem] tracking-tight mt-3">
          What stood out.
        </h3>
        <div className="text-ink-soft text-[0.95rem] mt-1">
          Position, net game, errors. Tap a timestamp to jump.
        </div>
      </div>

      <div className="cc-coach-grid mt-5">
        <PositionTile data={position} />
        <NetApproachTile data={netApproach} videoRef={videoRef} />
        <ErrorTile data={errors} videoRef={videoRef} />
      </div>

      <style>{`
        .cc-coach-grid {
          display: grid;
          gap: 24px;
          grid-template-columns: 1.1fr 1fr 1fr;
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
      <div className="font-mono text-[0.66rem] uppercase tracking-[0.16em] text-ink-mute">
        {eyebrow}
      </div>
      <div className="font-display font-medium text-[1.05rem] tracking-tight mt-1.5">
        {headline}
      </div>
    </div>
  );
}

function PositionTile({ data }: { data: PositionSummary | null }) {
  if (!data || data.n_frames === 0) {
    return (
      <div className="cc-coach-tile">
        <TileHead eyebrow="Court position" headline="—" />
        <p className="text-[0.88rem] text-ink-mute italic">No position data.</p>
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
        {zones.map((z) => (
          <div
            key={z.label}
            className="grid items-center"
            style={{ gridTemplateColumns: '128px 1fr 42px', gap: 10 }}
          >
            <span className="text-[0.82rem] text-ink-soft">{z.label}</span>
            <div className="h-2 rounded-full overflow-hidden bg-shade dark:bg-surface">
              <div
                className="h-full rounded-full"
                style={{
                  width: `${Math.min(100, Math.max(0, z.pct))}%`,
                  background: z.color,
                  transition: 'width 600ms var(--ease-out)',
                }}
              />
            </div>
            <span
              className="text-right text-[0.82rem] font-display font-medium text-ink"
              style={{ fontFeatureSettings: '"tnum"' }}
            >
              {Math.round(z.pct)}%
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
        <TileHead eyebrow="Net game" headline="No net approaches detected." />
        <p className="text-[0.88rem] text-ink-mute italic mt-1">
          Player stayed behind the service line for the whole recording.
        </p>
      </div>
    );
  }
  return (
    <div className="cc-coach-tile">
      <TileHead
        eyebrow="Net game"
        headline={`${data.approaches} approached · ${data.wins} won (${Math.round(data.win_pct)}%).`}
      />
      <div className="text-[0.78rem] text-ink-mute mb-2">
        Win rate heuristic (rally end &lt; 3s · opponent OOB last). Refines post-pilot.
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
  if (!data || data.total === 0) {
    return (
      <div className="cc-coach-tile">
        <TileHead eyebrow="Errors" headline="No out-of-bounds bounces." />
      </div>
    );
  }
  const rows: { label: string; n: number; color: string }[] = [
    { label: 'Long', n: data.long, color: 'var(--color-amber)' },
    { label: 'Wide', n: data.wide, color: 'var(--color-plum)' },
    { label: 'Net', n: data.net_err, color: 'var(--color-clay)' },
  ];
  const total = Math.max(1, data.total);
  return (
    <div className="cc-coach-tile">
      <TileHead
        eyebrow="Errors"
        headline={`${data.total} OOB · ${data.long} long, ${data.wide} wide, ${data.net_err} net.`}
      />
      <div className="space-y-1.5 mb-3">
        {rows.map((r) => (
          <div
            key={r.label}
            className="grid items-center"
            style={{ gridTemplateColumns: '60px 1fr 28px', gap: 10 }}
          >
            <span className="text-[0.82rem] text-ink-soft">{r.label}</span>
            <div className="h-2 rounded-full overflow-hidden bg-shade dark:bg-surface">
              <div
                className="h-full rounded-full"
                style={{
                  width: `${(r.n / total) * 100}%`,
                  background: r.color,
                  transition: 'width 600ms var(--ease-out)',
                }}
              />
            </div>
            <span
              className="text-right text-[0.82rem] font-display font-medium text-ink"
              style={{ fontFeatureSettings: '"tnum"' }}
            >
              {r.n}
            </span>
          </div>
        ))}
      </div>
      <TimestampList
        items={data.events.map((e) => ({
          frame: e.frame,
          time_s: e.time_s,
          label: e.direction[0].toUpperCase() + e.direction.slice(1),
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
    return <p className="text-[0.82rem] text-ink-mute italic">{emptyLabel}</p>;
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
              className="font-mono text-[0.7rem] tabular-nums"
              style={{ fontFeatureSettings: '"tnum"' }}
            >
              {ts}
            </span>
            <span className="text-[0.7rem]">{it.label}</span>
          </button>
        );
      })}
      <style>{`
        .cc-coach-tile {
          border: 1px solid var(--color-line);
          border-radius: 12px;
          padding: 18px 18px 16px;
          background: color-mix(in srgb, var(--color-court) 4%, var(--color-paper));
          min-width: 0;
        }
        .cc-ts-chip {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          padding: 4px 8px 4px 7px;
          border-radius: 100px;
          border: 1px solid var(--color-line);
          background: var(--color-paper);
          color: var(--color-ink-soft);
          cursor: pointer;
          transition: border-color var(--duration-base) var(--ease-out),
            background var(--duration-base) var(--ease-out),
            color var(--duration-base) var(--ease-out);
        }
        .cc-ts-chip:hover {
          border-color: var(--color-ink);
          color: var(--color-ink);
        }
        .cc-ts-win {
          border-color: color-mix(in srgb, var(--color-court) 30%, var(--color-line));
          color: var(--color-court);
        }
        .cc-ts-loss {
          border-color: color-mix(in srgb, var(--color-clay) 30%, var(--color-line));
          color: var(--color-clay);
        }
      `}</style>
    </div>
  );
}
