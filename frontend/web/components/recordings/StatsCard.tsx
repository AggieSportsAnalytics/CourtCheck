'use client';

export type StatTileData = {
  label: string;
  value: string | number;
  unit?: string;
};

type Props = {
  shotsTracked?: number;
  tiles: StatTileData[];
};

/**
 * 4-tile stats card. 4-col grid, collapses to 2x2 below 900px.
 *
 * Per-tile hover: translateY(-2px) + shadow + border-ink-mute. Value swaps
 * to court color on hover. No dim-siblings (intentionally removed last
 * session).
 */
export default function StatsCard({ shotsTracked, tiles }: Props) {
  return (
    <div
      className="cc-card bg-paper border border-line rounded-[14px] mb-8"
      style={{ padding: '32px 36px' }}
    >
      <div
        className="flex justify-between items-end mb-[18px] pb-4"
        style={{ borderBottom: '1px solid var(--color-line-soft)' }}
      >
        <div>
          <span className="inline-flex items-center gap-2 font-mono text-[0.72rem] uppercase tracking-[0.18em] text-court before:content-[''] before:w-1.5 before:h-1.5 before:bg-clay before:rounded-full">
            Stats · this recording
          </span>
          <h2
            className="font-display font-medium mt-2"
            style={{ fontSize: '1.75rem', letterSpacing: '-0.015em' }}
          >
            The numbers from this recording.
          </h2>
          <div className="text-ink-soft text-[0.92rem] mt-1">
            Auto-tracked. Player only. Opponent data shown where comparable.
          </div>
        </div>
        {typeof shotsTracked === 'number' && (
          <span className="font-mono text-[0.7rem] uppercase tracking-[0.14em] text-ink-mute shrink-0">
            {shotsTracked.toLocaleString()} shots
          </span>
        )}
      </div>

      <div className="grid gap-[18px] grid-cols-4 max-[900px]:grid-cols-2">
        {tiles.map((t) => (
          <Tile key={t.label} label={t.label} value={t.value} unit={t.unit} />
        ))}
      </div>
    </div>
  );
}

function Tile({ label, value, unit }: StatTileData) {
  return (
    <div
      className="cc-stat-tile group cursor-default bg-shade dark:bg-surface border border-transparent"
      style={{ padding: '26px 24px' }}
    >
      <div
        className="font-mono uppercase mb-3 text-ink-mute group-hover:text-ink-soft"
        style={{
          fontSize: '0.7rem',
          letterSpacing: '0.14em',
          transition: 'color var(--duration-base) var(--ease-out)',
        }}
      >
        {label}
      </div>
      <div
        className="font-display font-medium leading-[1.05] group-hover:text-court dark:group-hover:text-court-light"
        style={{
          fontSize: '2.6rem',
          letterSpacing: '-0.014em',
          fontFeatureSettings: '"tnum"',
          transition: 'color var(--duration-base) var(--ease-out)',
        }}
      >
        {value}
        {unit && (
          <span
            className="text-ink-mute font-normal"
            style={{ fontSize: '0.55em', marginLeft: 4 }}
          >
            {unit}
          </span>
        )}
      </div>
    </div>
  );
}
