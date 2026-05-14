'use client';

import CountUp from './CountUp';

type Stat = {
  label: string;
  value: number;
  decimals?: number;
  suffix?: React.ReactNode;
  caption?: string;
};

type Props = {
  clips: number;
  players: number;
  patterns: number;
  hours: number;
};

export default function TeamStrip({ clips, players, patterns, hours }: Props) {
  const stats: Stat[] = [
    { label: 'Clips this season', value: clips },
    { label: 'Players', value: players },
    { label: 'Patterns surfaced', value: patterns },
    {
      label: 'Hours recorded',
      value: hours,
      decimals: 1,
      suffix: <span className="ml-1 text-[0.42em] text-ink-mute">hrs</span>,
    },
  ];

  return (
    <section
      className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-7"
      aria-label="Team metrics"
    >
      {stats.map((s) => (
        <div
          key={s.label}
          className="cc-card p-6"
          data-count-card
        >
          <div
            className="font-mono uppercase tracking-[0.14em] text-[0.7rem] text-ink-mute mb-3"
          >
            {s.label}
          </div>
          <div
            className="text-ink"
            style={{
              fontFamily: 'var(--font-display)',
              fontWeight: 500,
              fontFeatureSettings: "'tnum'",
              fontSize: 'clamp(40px, 4.4vw, 56px)',
              lineHeight: 1.0,
              letterSpacing: '-0.022em',
            }}
          >
            <CountUp
              to={s.value}
              decimals={s.decimals ?? 0}
              suffix={s.suffix}
            />
          </div>
        </div>
      ))}
    </section>
  );
}
