'use client';

import Link from 'next/link';
import { Prose } from '@/components/ui/display';

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
          <div className="text-ink-mute text-[0.95rem] mt-1.5">
            Players with the newest recordings.
          </div>
        </div>
        <div className="font-mono uppercase tracking-[0.12em] text-[0.82rem] text-ink-mute hidden md:block">
          Latest recordings
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {items.map((item, idx) => (
          <article
            key={idx}
            className="cc-insight flex flex-col gap-3.5"
            style={{ padding: '20px 22px' }}
          >
            <div className="font-mono uppercase tracking-[0.12em] text-[0.82rem] text-ink-mute">
              {item.tag}
            </div>
            <Prose className="font-medium tracking-[-0.012em]">
              {item.line}
            </Prose>
            {item.href ? (
              <Link
                href={item.href}
                className="mt-auto inline-flex items-center gap-1.5 text-court text-[0.88rem] font-medium transition-[gap] duration-150"
              >
                <span>{item.cta ?? 'See breakdown'}</span>
                <span aria-hidden>→</span>
              </Link>
            ) : (
              <span className="mt-auto inline-flex items-center gap-1.5 text-court text-[0.88rem] font-medium">
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
