'use client';

import { RefObject, useState } from 'react';

export type RallyShot = {
  frame: number;
  time_s: number;
  player: 1 | 2;
  stroke: 'serve' | 'forehand' | 'backhand' | 'unknown';
  bounce_frame: number | null;
  bounce_x: number | null;
  bounce_y: number | null;
  in: boolean | null;
};

export type Rally = {
  rally_idx: number;
  start_frame: number;
  end_frame: number;
  duration_s: number;
  shot_count: number;
  server: 1 | 2 | null;
  winner: 1 | 2 | null;
  end_reason: string;
  truncated: boolean;
  shots: RallyShot[];
};

export type RallySummary = {
  total: number;
  avg_length: number;
  median_length: number;
  p1_wins: number;
  p2_wins: number;
  p1_win_rate: number;
  end_reasons: Record<string, number>;
};

type Props = {
  rallies: Rally[];
  videoRef: RefObject<HTMLVideoElement | null>;
  fps: number | null;
};

function outcomeLabel(rally: Rally): { text: string; tone: 'win' | 'loss' | 'neutral' } {
  if (rally.winner === 1) return { text: 'You won', tone: 'win' };
  if (rally.winner === 2) return { text: 'Opponent won', tone: 'loss' };
  // winner is null when the rally end couldn't be classified (clipped rally,
  // no swing before the last bounce, projection gap). The Errors tile is an
  // independent codepath (build_error_summary) so it can still count an error
  // on a bounce whose rally we couldn't resolve — surface "Not classified" here rather than
  // a jarring "Unknown" that reads as a bug next to that count.
  return { text: 'Not classified', tone: 'neutral' };
}

const REASON_LABELS = new Map([
  ['p1_winner', 'Your winner'],
  ['p2_winner', 'Opponent winner'],
  ['p1_long', 'You: long'],
  ['p1_wide', 'You: wide'],
  ['p1_net', 'You: net'],
  ['p2_long', 'Opponent: long'],
  ['p2_wide', 'Opponent: wide'],
  ['p2_net', 'Opponent: net'],
  ['p1_missed_return', 'Unreturned (opponent placement)'],
  ['p2_missed_return', 'Unreturned by opponent'],
  ['unknown', 'Not classified'],
]);

function reasonLabel(endReason?: string | null): string {
  return REASON_LABELS.get(endReason ?? 'unknown') ?? 'Not classified';
}

function fmtTs(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

export default function RallyTable({ rallies, videoRef, fps }: Props) {
  const [openIdx, setOpenIdx] = useState<number | null>(null);
  // Show truncated rallies only when there's nothing else to show — otherwise
  // hide them so the table doesn't surface clipped half-rallies.
  const playable = rallies.filter((r) => !r.truncated);
  const visible = playable.length > 0 ? playable : rallies;

  if (visible.length === 0) {
    return null;
  }

  const fpsSafe = fps && fps > 0 ? fps : 30;

  const seekToFrame = (frame: number) => {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = frame / fpsSafe;
    if (v.paused) void v.play();
  };

  return (
    <div
      className="cc-card bg-paper border border-line rounded-[14px] mb-8"
      style={{ padding: '26px 30px' }}
    >
      <div className="mb-3.5">
        <span className="inline-flex items-center gap-2 font-mono text-[0.82rem] uppercase tracking-[0.12em] text-court before:content-[''] before:w-1.5 before:h-1.5 before:bg-clay before:rounded-full">
          Rallies
        </span>
        <h3 className="font-display font-medium text-[1.25rem] tracking-tight mt-3">
          Rally by rally
        </h3>
        <div className="text-ink-soft text-[1.02rem] mt-1">
          Open a row for the shot sequence. Timestamps jump to the video.
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-[0.95rem]" style={{ borderCollapse: 'collapse' }}>
          <thead>
            <tr className="text-ink-mute font-mono text-[0.82rem] uppercase tracking-[0.12em]">
              <th className="text-left py-2 pr-3 font-normal">#</th>
              <th className="text-left py-2 pr-3 font-normal">Server</th>
              <th className="text-left py-2 pr-3 font-normal">Length</th>
              <th className="text-left py-2 pr-3 font-normal">Outcome</th>
              <th className="text-left py-2 pr-3 font-normal">Reason</th>
              <th className="text-left py-2 font-normal">Time</th>
            </tr>
          </thead>
          <tbody>
            {visible.map((r) => {
              const outcome = outcomeLabel(r);
              const isOpen = openIdx === r.rally_idx;
              return (
                <tr
                  key={r.rally_idx}
                  tabIndex={0}
                  role="button"
                  aria-expanded={isOpen}
                  className={`border-t border-line cursor-pointer transition-colors focus-visible:bg-shade ${
                    isOpen ? 'bg-shade/60' : 'hover:bg-shade/40'
                  }`}
                  onClick={() => setOpenIdx(isOpen ? null : r.rally_idx)}
                  onKeyDown={(e) => {
                    if (e.target !== e.currentTarget) return;
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      setOpenIdx(isOpen ? null : r.rally_idx);
                    }
                  }}
                >
                  <td className="py-2 pr-3 text-ink-mute font-mono tabular-nums">
                    {r.rally_idx + 1}
                  </td>
                  <td className="py-2 pr-3">
                    {r.server === 1 ? 'You' : r.server === 2 ? 'Opponent' : 'Not classified'}
                  </td>
                  <td className="py-2 pr-3 font-display tabular-nums">{r.shot_count}</td>
                  <td className="py-2 pr-3">
                    <span
                      className={
                        outcome.tone === 'win'
                          ? 'text-court'
                          : outcome.tone === 'loss'
                            ? 'text-clay'
                            : 'text-ink-mute'
                      }
                    >
                      {outcome.text}
                    </span>
                  </td>
                  <td className="py-2 pr-3 text-ink-soft text-[0.88rem]">
                    {reasonLabel(r.end_reason)}
                  </td>
                  <td className="py-2">
                    <button
                      type="button"
                      className="cc-ts-chip cc-ts-neutral"
                      onClick={(e) => {
                        e.stopPropagation();
                        seekToFrame(r.start_frame);
                      }}
                      title={`Jump to ${fmtTs(r.start_frame / fpsSafe)}`}
                    >
                      <span
                        className="font-mono text-[0.82rem] tabular-nums"
                        style={{ fontFeatureSettings: '"tnum"' }}
                      >
                        {fmtTs(r.start_frame / fpsSafe)}
                      </span>
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {openIdx !== null &&
        (() => {
          const rally = visible.find((r) => r.rally_idx === openIdx);
          if (!rally) return null;
          return (
            <div className="mt-4 pt-4 border-t border-line">
              <div className="font-mono text-[0.82rem] uppercase tracking-[0.12em] text-ink-mute mb-2">
                Rally {rally.rally_idx + 1} · {rally.shot_count} shots ·{' '}
                {rally.duration_s.toFixed(1)}s
              </div>
              <ol className="space-y-1.5">
                {rally.shots.map((s, i) => (
                  <li
                    key={`${rally.rally_idx}-${i}`}
                    className="flex items-center gap-3 text-[0.95rem]"
                  >
                    <span className="text-ink-mute font-mono tabular-nums w-6">
                      {i + 1}.
                    </span>
                    <span className="w-20 shrink-0 text-ink-soft">
                      {s.player === 1 ? 'You' : 'Opponent'}
                    </span>
                    <span className="w-20 capitalize">{s.stroke}</span>
                    <button
                      type="button"
                      className="cc-ts-chip cc-ts-neutral"
                      onClick={() => seekToFrame(s.frame)}
                    >
                      <span
                        className="font-mono text-[0.82rem] tabular-nums"
                        style={{ fontFeatureSettings: '"tnum"' }}
                      >
                        {fmtTs(s.time_s)}
                      </span>
                    </button>
                    {s.bounce_x !== null && s.bounce_y !== null && (
                      <span className="text-ink-mute text-[0.88rem]">
                        bounce {s.in ? 'in' : 'out'}
                      </span>
                    )}
                  </li>
                ))}
              </ol>
            </div>
          );
        })()}
    </div>
  );
}
