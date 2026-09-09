'use client';

import { Fragment, ReactNode } from 'react';
import { Prose } from '@/components/ui/display';

export type ScoutingSections = {
  matchSnapshot: string;
  positioningTendencies: string;
  errorPatterns: string;
  strengths: string;
  areasToImprove: string;
  oneLineAdjustment: string;
};

/**
 * Wrap numeric stats in <strong> so they pop visually in the prose. Catches:
 *   - percentages (e.g. "62%", "8.3%")
 *   - bare numbers (e.g. "12 unforced errors", "3 of 7")
 *   - score-style pairs (e.g. "6-4", "2/3")
 * Excludes years (1900–2099 standalone) and ordinals (1st, 2nd) to avoid
 * highlighting noise tokens. Splits the string into ReactNode chunks.
 */
const STAT_RE =
  /(\b\d+(?:\.\d+)?\s*%|\b\d+(?:\.\d+)?\s*(?:of|\/|-)\s*\d+(?:\.\d+)?|\b\d+(?:\.\d+)?\b)/g;
function highlightStats(text: string): ReactNode[] {
  if (!text) return [text];
  const out: ReactNode[] = [];
  let lastIndex = 0;
  let i = 0;
  for (const m of text.matchAll(STAT_RE)) {
    const token = m[0];
    // Filter ordinals and 4-digit years that aren't stats
    if (/^(19|20)\d{2}$/.test(token.trim())) continue;
    const start = m.index ?? 0;
    if (start > lastIndex) out.push(text.slice(lastIndex, start));
    out.push(
      <strong
        key={`s-${i++}`}
        className="font-display font-medium text-ink"
        style={{ fontFeatureSettings: '"tnum"' }}
      >
        {token}
      </strong>,
    );
    lastIndex = start + token.length;
  }
  if (lastIndex < text.length) out.push(text.slice(lastIndex));
  return out.length > 0 ? out : [text];
}

type Props = {
  headline?: string;
  sections: ScoutingSections;
  readMinutes?: number;
};

/**
 * Scouting report card — six section headers (coach Jackson's preferred
 * format), tight prose.
 *
 * Header style: JetBrains Mono UPPERCASE, court color.
 * Final section is a court-tinted left-rail callout.
 */
export default function ScoutingReport({ headline, sections, readMinutes }: Props) {
  return (
    <article
      className="bg-paper border border-line rounded-[14px] mb-8 p-6 sm:py-9 sm:px-11"
      style={{ boxShadow: 'var(--shadow-card)' }}
      aria-label="Scouting report"
    >
      <div
        className="flex flex-wrap items-end justify-between gap-4 mb-6 pb-[18px]"
        style={{ borderBottom: '1px solid var(--color-line-soft)' }}
      >
        <div>
          <span className="inline-flex items-center gap-2 font-mono text-[0.82rem] uppercase tracking-[0.12em] text-court before:content-[''] before:w-1.5 before:h-1.5 before:bg-clay before:rounded-full">
            Scouting report
          </span>
          <h2
            className="font-display font-medium mt-2 leading-[1.1]"
            style={{
              fontVariationSettings: "'opsz' 72",
              fontSize: '1.95rem',
              letterSpacing: '-0.016em',
            }}
          >
            {headline ?? (
              <>
                Report
              </>
            )}
          </h2>
        </div>
        {typeof readMinutes === 'number' && (
          <span className="font-mono text-[0.82rem] uppercase tracking-[0.12em] text-ink-mute shrink-0">
            ~ {readMinutes} min read
          </span>
        )}
      </div>

      <Section heading="Snapshot" text={sections.matchSnapshot} />
      <Section heading="Positioning" text={sections.positioningTendencies} />
      <Section heading="Errors" text={sections.errorPatterns} />
      <Section heading="Strengths" text={sections.strengths} />
      <Section heading="Work on" text={sections.areasToImprove} />

      <section
        className="rounded-lg mt-2"
        style={{
          padding: '22px 28px',
          background: 'color-mix(in srgb, var(--color-court) 6%, transparent)',
          borderLeft: '3px solid var(--color-court)',
        }}
      >
        <h3 className="font-mono text-[0.82rem] uppercase tracking-[0.12em] font-semibold text-court mb-2">
          Coaching cue
        </h3>
        <Prose
          className="font-display font-medium text-ink m-0 text-[1.18rem] leading-[1.65]"
        >
          {sections.oneLineAdjustment}
        </Prose>
      </section>
    </article>
  );
}

function Section({ heading, text }: { heading: string; text: string }) {
  // Pre-split the text so numeric stats render in bold-court. Wrapped in a
  // Fragment to keep React's reconciler happy with mixed string+element kids.
  const parts = highlightStats(text);
  return (
    <section className="mb-[30px] last:mb-0">
      <h3
        className="font-display font-semibold text-ink mb-2"
        style={{
          fontSize: '1.18rem',
          letterSpacing: '-0.012em',
          lineHeight: 1.2,
        }}
      >
        {heading}
      </h3>
      <Prose
        className="font-normal tracking-[-0.005em]"
      >
        {parts.map((p, i) => (
          <Fragment key={i}>{p}</Fragment>
        ))}
      </Prose>
    </section>
  );
}
