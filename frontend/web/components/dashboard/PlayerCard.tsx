'use client';

import Link from 'next/link';
import { useState } from 'react';
import CountUp from './CountUp';
import { playerPhotoProxyUrl } from '@/lib/utils';

export type PlayerMetric = {
  key: string;
  label: string;
  value: number;
  /** Optional delta in absolute pts (signed). Card lead = metric with biggest |delta|. */
  delta?: number;
  decimals?: number;
  /** Suffix unit (e.g. "%"). Rendered small inside the number. */
  unitSuffix?: string;
};

export type PlayerCardData = {
  id: string;
  name: string;
  firstName: string;
  lastName: string;
  meta: string;
  initials: string;
  avatarGradient: string;
  photoUrl: string | null;
  lastClipDate: string | null;
  metrics: PlayerMetric[];
};

type Props = {
  player: PlayerCardData;
};

/**
 * Player card with 5-metric row + dim-siblings hover.
 *
 * The metric with the biggest |delta| is the "lead" metric: it stays at full opacity,
 * shows the signed delta, and non-lead metrics dim to 0.5 at rest.
 *
 * On hovering any single metric, that metric goes to full opacity and ALL siblings
 * (in the same card) drop to 0.35. Implemented with a peer/group pattern via group
 * scoped CSS in styled spans (no global CSS leakage).
 */
export default function PlayerCard({ player }: Props) {
  const leadIdx = findLeadIndex(player.metrics);
  const [photoFailed, setPhotoFailed] = useState(false);
  const photoSrc = photoFailed ? null : playerPhotoProxyUrl(player.photoUrl);

  return (
    <Link
      href={`/players/${player.id}`}
      className="player-card cc-card flex flex-col gap-4 p-7 cursor-pointer group/card"
      style={{ display: 'flex' }}
    >
      {/* Head: avatar + name + meta */}
      <div className="flex items-center gap-4">
        <div
          className="w-14 h-14 rounded-full flex-shrink-0 overflow-hidden flex items-center justify-center text-cream"
          style={{
            background: player.avatarGradient,
            fontFamily: 'var(--font-display)',
            fontWeight: 500,
            fontSize: '1.45rem',
            letterSpacing: '-0.012em',
          }}
          aria-hidden
        >
          {photoSrc ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={photoSrc}
              alt=""
              className="w-full h-full object-cover"
              onError={() => setPhotoFailed(true)}
            />
          ) : (
            player.initials
          )}
        </div>
        <div className="flex flex-col gap-0.5 min-w-0">
          <span
            className="text-ink"
            style={{
              fontFamily: 'var(--font-display)',
              fontWeight: 500,
              fontSize: '1.4rem',
              letterSpacing: '-0.014em',
              lineHeight: 1.05,
            }}
          >
            {player.firstName} <em>{player.lastName}</em>
          </span>
          <span className="text-sm text-ink-soft">{player.meta}</span>
        </div>
      </div>

      {/* 5-metric row (uses peer/sibling hover via group/metric-row) */}
      <div
        className="grid grid-cols-5 gap-2 p-3.5 rounded-[10px] bg-shade dark:bg-surface group/metricrow"
      >
        {player.metrics.map((m, i) => {
          const isLead = i === leadIdx;
          return (
            <div
              key={m.key}
              className={[
                'flex flex-col gap-px transition-opacity duration-150 ease-out',
                // Rest state: non-lead metrics dim to 0.5
                isLead ? 'opacity-100' : 'opacity-50',
                // Group hover: when any metric in the row is hovered, ALL siblings drop to 0.35.
                // The hovered one itself overrides back to 1 via hover: below.
                'group-hover/metricrow:opacity-[0.35]',
                'hover:!opacity-100',
              ].join(' ')}
            >
              <span
                className="text-ink"
                style={{
                  fontFamily: 'var(--font-display)',
                  fontWeight: 500,
                  fontSize: '1.3rem',
                  letterSpacing: '-0.012em',
                  fontFeatureSettings: "'tnum'",
                  lineHeight: 1.05,
                }}
              >
                <CountUp
                  to={m.value}
                  decimals={m.decimals ?? 0}
                  suffix={
                    m.unitSuffix ? (
                      <span
                        className="text-ink-mute"
                        style={{ fontSize: '0.62em', marginLeft: 2 }}
                      >
                        {m.unitSuffix}
                      </span>
                    ) : undefined
                  }
                />
              </span>
              <span className="font-mono uppercase tracking-[0.1em] text-[0.6rem] text-ink-mute">
                {m.label}
              </span>
              {isLead && typeof m.delta === 'number' && (
                <span
                  className={[
                    'font-mono uppercase tracking-[0.1em] text-[0.6rem] mt-0.5',
                    m.delta >= 0
                      ? 'text-court dark:text-court-light'
                      : 'text-clay dark:text-clay-soft',
                  ].join(' ')}
                >
                  {m.delta >= 0 ? '▲' : '▼'} {Math.abs(m.delta).toFixed(0)} pts
                </span>
              )}
            </div>
          );
        })}
      </div>

      {/* Footer: last clip + arrow */}
      <div className="flex justify-between items-center mt-auto pt-3.5 border-t border-line-soft">
        <span className="font-mono uppercase tracking-[0.1em] text-[0.7rem] text-ink-mute">
          {player.lastClipDate ? `Last clip · ${player.lastClipDate}` : 'No clips yet'}
        </span>
        <span
          className="text-ink-mute group-hover/card:translate-x-1 group-hover/card:text-ink transition-all duration-200"
          aria-hidden
        >
          →
        </span>
      </div>
    </Link>
  );
}

function findLeadIndex(metrics: PlayerMetric[]): number {
  let bestIdx = 0;
  let bestAbs = -Infinity;
  metrics.forEach((m, i) => {
    if (typeof m.delta === 'number' && Math.abs(m.delta) > bestAbs) {
      bestAbs = Math.abs(m.delta);
      bestIdx = i;
    }
  });
  return bestIdx;
}
