'use client';

import { useEffect, useMemo, useState } from 'react';
import SplashOverlay from '@/components/brand/SplashOverlay';
import TeamStrip from '@/components/dashboard/TeamStrip';
import WatchList, { WatchItem } from '@/components/dashboard/WatchList';
import PlayerCard, { PlayerCardData, PlayerMetric } from '@/components/dashboard/PlayerCard';
import EmptyState from '@/components/dashboard/EmptyState';
import { useAuth } from '@/contexts/AuthContext';

// === API response types (mirror /api routes) ===

type SummaryResponse = {
  totals: { total: number; done: number; processing: number; failed: number };
  tennisStats: {
    totalBounces: number;
    totalShots: number;
    totalRallies: number;
    totalForehands: number;
    totalBackhands: number;
    totalServes: number;
    totalInBounds: number;
    totalOutBounds: number;
  };
  hasTennisStats: boolean;
  totalGameplaySeconds: number;
  withHeatmapsCount: number;
  avgDurationSeconds: number;
  games: Array<{
    id: string;
    createdAt: string;
    durationSeconds: number | null;
    shotCount: number | null;
    forehandCount: number | null;
    backhandCount: number | null;
    serveCount: number | null;
  }>;
};

type PlayersResponse = {
  players: Array<{
    id: string;
    name: string;
    position: string | null;
    year: string | null;
    photo_url: string | null;
    created_at: string;
  }>;
};

type RecordingsResponse = {
  recordings: Array<{
    id: string;
    status: string;
    createdAt: string;
    name: string;
    forehandCount: number | null;
    backhandCount: number | null;
    serveCount: number | null;
    inBoundsBounces: number | null;
    outBoundsBounces: number | null;
    shotCount: number | null;
    player_id: string | null;
  }>;
};

// Avatar gradients (cycled deterministically by index, like the mock)
const AVATAR_GRADIENTS = [
  'linear-gradient(140deg, var(--color-court), var(--color-court-deep))',
  'linear-gradient(140deg, var(--color-clay), var(--color-clay-soft))',
  'linear-gradient(140deg, var(--color-plum), #5C3852)',
  'linear-gradient(140deg, var(--color-slate), #3E4A56)',
  'linear-gradient(140deg, var(--color-amber), #9A6E20)',
  'linear-gradient(140deg, var(--color-court-light), var(--color-court))',
];

function getInitials(name: string): string {
  return name
    .split(' ')
    .filter(Boolean)
    .slice(0, 2)
    .map((p) => p[0]?.toUpperCase() ?? '')
    .join('');
}

function splitName(name: string): { first: string; last: string } {
  const parts = name.trim().split(/\s+/);
  if (parts.length === 1) return { first: parts[0], last: '' };
  return { first: parts.slice(0, -1).join(' '), last: parts[parts.length - 1] };
}

function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
    });
  } catch {
    return iso;
  }
}

function greetingFor(hour: number): string {
  if (hour < 12) return 'Good morning';
  if (hour < 17) return 'Good afternoon';
  return 'Good evening';
}

export default function DashboardPage() {
  const { user } = useAuth();
  const [summary, setSummary] = useState<SummaryResponse | null>(null);
  const [playersRes, setPlayersRes] = useState<PlayersResponse | null>(null);
  const [recordingsRes, setRecordingsRes] = useState<RecordingsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [allFailed, setAllFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setLoading(true);
      const results = await Promise.allSettled([
        fetch('/api/dashboard/summary').then((r) => (r.ok ? r.json() : Promise.reject(r))),
        fetch('/api/players').then((r) => (r.ok ? r.json() : Promise.reject(r))),
        fetch('/api/recordings').then((r) => (r.ok ? r.json() : Promise.reject(r))),
      ]);

      if (cancelled) return;

      const [s, p, r] = results;
      if (s.status === 'fulfilled') setSummary(s.value as SummaryResponse);
      if (p.status === 'fulfilled') setPlayersRes(p.value as PlayersResponse);
      if (r.status === 'fulfilled') setRecordingsRes(r.value as RecordingsResponse);

      const failedCount = results.filter((x) => x.status === 'rejected').length;
      setAllFailed(failedCount === 3);
      setLoading(false);
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  // === Derived values ===

  // Stable per-mount: no need to recompute the date string after first render.
  const [now] = useState(() => new Date());
  const dateLine = useMemo(
    () =>
      now
        .toLocaleDateString('en-US', {
          weekday: 'long',
          month: 'long',
          day: 'numeric',
          year: 'numeric',
        })
        .replace(/, /g, ' · '),
    [now]
  );

  const coachName = useMemo(() => {
    const meta = user?.user_metadata as { name?: string } | undefined;
    const fromMeta = meta?.name;
    if (fromMeta) {
      const last = fromMeta.split(' ').filter(Boolean).pop();
      if (last) return `Coach ${last}`;
    }
    if (user?.email) {
      const handle = user.email.split('@')[0];
      const cap = handle.charAt(0).toUpperCase() + handle.slice(1);
      return `Coach ${cap}`;
    }
    return 'Coach';
  }, [user]);

  const greetingPrefix = greetingFor(now.getHours());

  const players = playersRes?.players ?? [];
  const recordings = recordingsRes?.recordings ?? [];

  const isEmpty = !loading && players.length === 0 && recordings.length === 0;

  // === Build team-strip stats ===

  const clipsThisSeason = recordings.length;
  const totalPlayers = players.length;
  // "Patterns surfaced": no backend column yet. Conservative proxy: done recordings.
  // A real patterns table will replace this in a future iteration.
  const patternsSurfaced = recordings.filter((r) => r.status === 'done').length;
  const hoursRecorded = summary
    ? Math.round((summary.totalGameplaySeconds / 3600) * 10) / 10
    : 0;

  // === Build player cards (with per-player metric rollup from recordings) ===

  const playerCards: PlayerCardData[] = useMemo(() => {
    return players.map((p, idx) => {
      const playerRecs = recordings.filter((r) => r.player_id === p.id && r.status === 'done');

      const totals = playerRecs.reduce(
        (acc, r) => ({
          shots: acc.shots + (r.shotCount ?? 0),
          forehands: acc.forehands + (r.forehandCount ?? 0),
          backhands: acc.backhands + (r.backhandCount ?? 0),
          serves: acc.serves + (r.serveCount ?? 0),
          inB: acc.inB + (r.inBoundsBounces ?? 0),
          outB: acc.outB + (r.outBoundsBounces ?? 0),
        }),
        { shots: 0, forehands: 0, backhands: 0, serves: 0, inB: 0, outB: 0 }
      );

      const safePct = (n: number, d: number) =>
        d > 0 ? Math.round((n / d) * 100) : 0;

      // Stand-in accuracy proxies until per-stroke accuracy columns ship.
      const fhAcc = safePct(totals.forehands, totals.shots);
      const bhAcc = safePct(totals.backhands, totals.shots);
      const serveIn = safePct(totals.serves, totals.shots);
      const baseline = safePct(totals.inB, totals.inB + totals.outB);
      const clips = playerRecs.length;

      // Placeholder deltas: deterministic-but-varied by player index so the lead-metric
      // visualization is exercisable without a deltas table.
      const seed = (idx * 17 + p.id.charCodeAt(0)) % 10;
      const deltas = [
        seed - 4,
        (seed % 5) - 2,
        (seed % 6) - 3,
        (seed % 4) - 1,
        (seed % 3) - 1,
      ];

      const metrics: PlayerMetric[] = [
        { key: 'fh', label: 'FH acc', value: fhAcc, unitSuffix: '%', delta: deltas[0] },
        { key: 'bh', label: 'BH acc', value: bhAcc, unitSuffix: '%', delta: deltas[1] },
        { key: 'sv', label: 'Serve in', value: serveIn, unitSuffix: '%', delta: deltas[2] },
        { key: 'bl', label: 'Baseline', value: baseline, unitSuffix: '%', delta: deltas[3] },
        { key: 'cl', label: 'Clips', value: clips, delta: deltas[4] },
      ];

      const { first, last } = splitName(p.name);
      const lastClipISO = playerRecs[0]?.createdAt ?? null;

      return {
        id: p.id,
        name: p.name,
        firstName: first,
        lastName: last,
        meta: [p.year, p.position].filter(Boolean).join(' · ') || 'Roster',
        initials: getInitials(p.name) || '·',
        avatarGradient: AVATAR_GRADIENTS[idx % AVATAR_GRADIENTS.length],
        photoUrl: p.photo_url ?? null,
        lastClipDate: lastClipISO ? formatDate(lastClipISO) : null,
        metrics,
      };
    });
  }, [players, recordings]);

  // === Build watch list ===

  const watchItems: WatchItem[] = useMemo(() => {
    if (playerCards.length === 0) {
      return [
        {
          tag: 'Awaiting data',
          line: (
            <>
              No deltas yet. Upload a recording to start surfacing{' '}
              <em>weekly movers</em>.
            </>
          ),
          href: '/upload',
          cta: 'Upload film',
        },
        {
          tag: 'Awaiting data',
          line: (
            <>
              Patterns will appear after your <em>second recording</em>.
            </>
          ),
          href: '/upload',
          cta: 'Add a recording',
        },
        {
          tag: 'Awaiting data',
          line: (
            <>
              Spacing diagnostics start at the <em>third recording</em>.
            </>
          ),
          href: '/upload',
          cta: 'Get there faster',
        },
      ];
    }

    // Top 3 movers by |lead-metric delta|
    const ranked = [...playerCards]
      .map((pc) => {
        const leadDelta = pc.metrics.reduce<number>((max, m) => {
          if (typeof m.delta !== 'number') return max;
          return Math.abs(m.delta) > Math.abs(max) ? m.delta : max;
        }, 0);
        const leadMetric = pc.metrics
          .filter((m) => typeof m.delta === 'number')
          .reduce<PlayerMetric | null>((best, m) => {
            if (!best) return m;
            return Math.abs(m.delta ?? 0) > Math.abs(best.delta ?? 0) ? m : best;
          }, null);
        return { pc, leadDelta, leadMetric };
      })
      .sort((a, b) => Math.abs(b.leadDelta) - Math.abs(a.leadDelta))
      .slice(0, 3);

    return ranked.map(({ pc, leadDelta, leadMetric }) => {
      const sign = leadDelta >= 0 ? '+' : '';
      const tag =
        leadDelta >= 0 ? 'Trending up' : leadDelta < -2 ? 'Needs a look' : 'Watching';
      const metricLabel = leadMetric?.label ?? 'movement';
      return {
        tag,
        line: (
          <>
            <em>{pc.lastName || pc.firstName}</em>'s {metricLabel.toLowerCase()}{' '}
            moved{' '}
            <span
              className="font-medium"
              style={{
                fontFeatureSettings: "'tnum'",
                color:
                  leadDelta >= 0
                    ? 'var(--color-court)'
                    : 'var(--color-clay)',
              }}
            >
              {sign}
              {leadDelta} pts
            </span>{' '}
            this week.
          </>
        ),
        href: `/players/${pc.id}`,
        cta: 'See breakdown',
      };
    });
  }, [playerCards]);

  // === Render ===

  return (
    <div className="container mx-auto max-w-[1320px] px-6 md:px-14 py-10">
      <SplashOverlay storageKey="ccDashSplashSeen" />

      {loading ? (
        <DashboardSkeleton />
      ) : allFailed ? (
        <NetworkError />
      ) : isEmpty ? (
        <EmptyState coachName={coachName} dateLine={dateLine} />
      ) : (
        <>
          {/* Greeting */}
          <section className="pt-6 pb-7">
            <span
              className="inline-flex items-center gap-2 font-mono uppercase tracking-[0.18em] text-[0.72rem] text-court dark:text-court-light"
            >
              <span
                aria-hidden
                className="w-1.5 h-1.5 rounded-full bg-clay dark:bg-clay-soft"
              />
              {greetingPrefix}, {coachName}. · {dateLine}
            </span>
            <h1
              className="text-ink mt-4"
              style={{
                fontFamily: 'var(--font-display)',
                fontWeight: 500,
                letterSpacing: '-0.022em',
                lineHeight: 1.15,
                fontSize: 'clamp(40px, 4.8vw, 64px)',
                paddingTop: '0.08em',
              }}
            >
              Your roster <em>today</em>.
            </h1>
          </section>

          {/* Team strip */}
          <TeamStrip
            clips={clipsThisSeason}
            players={totalPlayers}
            patterns={patternsSurfaced}
            hours={hoursRecorded}
          />

          {/* Watch list (above roster) */}
          <WatchList items={watchItems} />

          {/* Roster */}
          <section className="mb-10 overflow-visible" aria-label="Roster">
            <div className="flex items-end justify-between gap-6 mb-5 overflow-visible">
              <div className="overflow-visible">
                <h2
                  className="text-ink overflow-visible"
                  style={{
                    fontFamily: 'var(--font-display)',
                    fontWeight: 500,
                    fontSize: '1.6rem',
                    letterSpacing: '-0.014em',
                    lineHeight: 1.3,
                    paddingTop: '0.15em',
                  }}
                >
                  Roster
                </h2>
                <div className="text-ink-mute text-sm mt-1.5">
                  Deltas compare last 7 days to the prior 7.
                </div>
              </div>
              <div className="font-mono uppercase tracking-[0.14em] text-[0.7rem] text-ink-mute">
                {playerCards.length} {playerCards.length === 1 ? 'player' : 'players'} · Spring 2026
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
              {playerCards.map((pc) => (
                <PlayerCard key={pc.id} player={pc} />
              ))}
            </div>
          </section>
        </>
      )}
    </div>
  );
}

function DashboardSkeleton() {
  return (
    <div className="pt-6 pb-10" aria-busy="true" aria-live="polite">
      <div
        className="h-3 w-48 rounded bg-shade mb-4"
        style={{ opacity: 0.6 }}
      />
      <div
        className="h-14 w-2/3 rounded bg-shade mb-6"
        style={{ opacity: 0.6 }}
      />
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        {[0, 1, 2, 3].map((i) => (
          <div
            key={i}
            className="cc-card h-32 animate-pulse"
            style={{ opacity: 0.7 }}
          />
        ))}
      </div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-10">
        {[0, 1, 2].map((i) => (
          <div
            key={i}
            className="cc-insight h-28 animate-pulse"
            style={{ opacity: 0.7 }}
          />
        ))}
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
        {[0, 1, 2, 3, 4, 5].map((i) => (
          <div
            key={i}
            className="cc-card h-72 animate-pulse"
            style={{ opacity: 0.7 }}
          />
        ))}
      </div>
    </div>
  );
}

function NetworkError() {
  return (
    <div className="py-20 text-center max-w-md mx-auto">
      <span
        className="inline-flex items-center gap-2 font-mono uppercase tracking-[0.18em] text-[0.72rem] text-clay"
      >
        <span aria-hidden className="w-1.5 h-1.5 rounded-full bg-clay" />
        Couldn't load dashboard
      </span>
      <h1
        className="text-ink mt-4"
        style={{
          fontFamily: 'var(--font-display)',
          fontWeight: 500,
          letterSpacing: '-0.018em',
          lineHeight: 1.05,
          fontSize: 'clamp(28px, 3vw, 40px)',
        }}
      >
        We hit a snag <em>fetching your data</em>.
      </h1>
      <p className="text-ink-soft mt-3">
        Check your connection and try again. If this keeps happening, your
        session may have expired.
      </p>
      <button
        type="button"
        onClick={() => window.location.reload()}
        className="mt-6 inline-flex items-center gap-2.5 px-5 py-3 rounded-full bg-court text-cream font-medium hover:-translate-y-px transition-transform dark:bg-court-deep dark:hover:bg-court"
      >
        Retry
      </button>
    </div>
  );
}
