'use client';

import Link from 'next/link';

export type WatchItem = {
  tag: string;
  line: React.ReactNode;
  href?: string;
  cta?: string;
};

type Props = {
  items: WatchItem[];
};

export default function WatchList({ items }: Props) {
  return (
    <section className="mb-10 overflow-visible" aria-label="Watch list">
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
            Watch list
          </h2>
          <div className="text-ink-mute text-sm mt-1.5">
            Three players who moved the most this week.
          </div>
        </div>
        <div className="font-mono uppercase tracking-[0.14em] text-[0.7rem] text-ink-mute hidden md:block">
          Last 7 days
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {items.map((item, idx) => (
          <article
            key={idx}
            className="cc-insight flex flex-col gap-3.5"
            style={{ padding: '20px 22px' }}
          >
            <div className="font-mono uppercase tracking-[0.16em] text-[0.66rem] text-ink-mute">
              {item.tag}
            </div>
            <div
              className="text-ink"
              style={{
                fontFamily: 'var(--font-display)',
                fontWeight: 500,
                fontSize: '1.04rem',
                letterSpacing: '-0.012em',
                lineHeight: 1.35,
              }}
            >
              {item.line}
            </div>
            {item.href ? (
              <Link
                href={item.href}
                className="mt-auto inline-flex items-center gap-1.5 text-court text-[0.82rem] font-medium dark:text-court-light transition-[gap] duration-150"
              >
                <span>{item.cta ?? 'See breakdown'}</span>
                <span aria-hidden>→</span>
              </Link>
            ) : (
              <span className="mt-auto inline-flex items-center gap-1.5 text-court text-[0.82rem] font-medium dark:text-court-light">
                <span>{item.cta ?? 'See breakdown'}</span>
                <span aria-hidden>→</span>
              </span>
            )}
          </article>
        ))}
      </div>
    </section>
  );
}
