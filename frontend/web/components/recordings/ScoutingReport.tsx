'use client';

export type ScoutingSections = {
  matchSnapshot: string;
  positioningTendencies: string;
  errorPatterns: string;
  strengths: string;
  areasToImprove: string;
  oneLineAdjustment: string;
};

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
      className="bg-paper border border-line rounded-[14px] mb-8"
      style={{ padding: '36px 44px', boxShadow: 'var(--shadow-card)' }}
      aria-label="Scouting report"
    >
      <div
        className="flex items-end justify-between gap-4 mb-6 pb-[18px]"
        style={{ borderBottom: '1px solid var(--color-line-soft)' }}
      >
        <div>
          <span className="inline-flex items-center gap-2 font-mono text-[0.72rem] uppercase tracking-[0.18em] text-court before:content-[''] before:w-1.5 before:h-1.5 before:bg-clay before:rounded-full">
            Scouting report · this recording
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
                What stood out in <em>this recording.</em>
              </>
            )}
          </h2>
        </div>
        {typeof readMinutes === 'number' && (
          <span className="font-mono text-[0.7rem] uppercase tracking-[0.14em] text-ink-mute shrink-0">
            ~ {readMinutes} min read
          </span>
        )}
      </div>

      <Section heading="Match Snapshot" text={sections.matchSnapshot} />
      <Section heading="Positioning Tendencies" text={sections.positioningTendencies} />
      <Section heading="Error Patterns" text={sections.errorPatterns} />
      <Section heading="Strengths" text={sections.strengths} />
      <Section heading="Areas to Improve" text={sections.areasToImprove} />

      <section
        className="rounded-lg mt-2"
        style={{
          padding: '22px 28px',
          background: 'color-mix(in srgb, var(--color-court) 6%, transparent)',
          borderLeft: '3px solid var(--color-court)',
        }}
      >
        <h3 className="font-mono text-[0.72rem] uppercase tracking-[0.18em] font-semibold text-court mb-2">
          One-Line Coaching Adjustment
        </h3>
        <p
          className="font-display italic font-medium text-ink m-0"
          style={{ fontSize: '1.18rem', lineHeight: 1.65 }}
        >
          {sections.oneLineAdjustment}
        </p>
      </section>
    </article>
  );
}

function Section({ heading, text }: { heading: string; text: string }) {
  return (
    <section className="mb-[26px] last:mb-0">
      <h3 className="font-mono text-[0.72rem] uppercase tracking-[0.18em] font-semibold text-court mb-2.5">
        {heading}
      </h3>
      <p
        className="font-display font-normal text-ink"
        style={{
          fontVariationSettings: "'opsz' 18",
          fontSize: '1.15rem',
          lineHeight: 1.65,
          letterSpacing: '-0.005em',
        }}
      >
        {text}
      </p>
    </section>
  );
}
