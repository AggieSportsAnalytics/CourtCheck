'use client';

import Link from 'next/link';

type Props = {
  coachName: string;
  dateLine: string;
};

/**
 * First-session empty state.
 * Renders when the workspace has no players AND no recordings.
 */
export default function EmptyState({ coachName, dateLine }: Props) {
  return (
    <>
      {/* Hero */}
      <section className="pt-14 pb-7">
        <span
          className="inline-flex items-center gap-2 font-mono uppercase tracking-[0.18em] text-[0.72rem] text-court dark:text-court-light"
        >
          <span
            aria-hidden
            className="w-1.5 h-1.5 rounded-full bg-clay dark:bg-clay-soft"
          />
          {dateLine}
        </span>
        <h1
          className="text-ink mt-4"
          style={{
            fontFamily: 'var(--font-display)',
            fontWeight: 500,
            letterSpacing: '-0.022em',
            lineHeight: 1.0,
            fontSize: 'clamp(48px, 6.4vw, 84px)',
          }}
        >
          Welcome, <em>{coachName}.</em>
        </h1>
        <p
          className="text-ink-soft mt-4 max-w-[60ch]"
          style={{ fontSize: '1.2rem', lineHeight: 1.5 }}
        >
          CourtCheck reads your recordings the way an experienced assistant coach
          watches them back. Every shot in context, every pattern in plain
          English. Upload your first recording below and we'll surface insights
          by morning.
        </p>
        <div className="flex gap-3.5 items-center flex-wrap mt-7">
          <Link
            href="/upload"
            className="inline-flex items-center gap-2.5 px-6 py-3.5 rounded-full bg-court text-cream font-medium text-base transition-transform hover:-translate-y-px dark:bg-court-deep dark:hover:bg-court"
            style={{ transition: 'transform 160ms cubic-bezier(0.34, 1.56, 0.64, 1), background 240ms cubic-bezier(0.2, 0.8, 0.2, 1)' }}
          >
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.75"
              strokeLinecap="round"
              strokeLinejoin="round"
              width="18"
              height="18"
            >
              <path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242" />
              <path d="M12 12v9" />
              <path d="m16 16-4-4-4 4" />
            </svg>
            Upload your first recording
          </Link>
          <Link
            href="/landing"
            className="inline-flex items-center gap-2.5 px-5 py-3 rounded-full border border-line text-ink text-base font-medium hover:border-ink transition-colors"
          >
            See a sample analysis →
          </Link>
        </div>
      </section>

      {/* Upload spotlight */}
      <Link
        href="/upload"
        className="block my-6 bg-paper rounded-[20px] border-[1.5px] border-dashed border-line hover:border-ink-mute transition-colors"
      >
        <div className="grid grid-cols-1 md:grid-cols-2 gap-10 items-center p-12 md:p-16">
          <div>
            <span
              className="inline-flex items-center gap-2 font-mono uppercase tracking-[0.18em] text-[0.72rem] text-court dark:text-court-light"
            >
              <span
                aria-hidden
                className="w-1.5 h-1.5 rounded-full bg-clay dark:bg-clay-soft"
              />
              Drop a recording
            </span>
            <h2
              className="text-ink mt-4 mb-3.5"
              style={{
                fontFamily: 'var(--font-display)',
                fontWeight: 500,
                letterSpacing: '-0.018em',
                lineHeight: 1.05,
                fontSize: 'clamp(32px, 3.6vw, 48px)',
              }}
            >
              Five minutes to your <em>first insight</em>.
            </h2>
            <p className="text-ink-soft mb-6 max-w-[36ch]" style={{ fontSize: '1.05rem', lineHeight: 1.55 }}>
              Upload an MP4 from this morning's recording. Within about 15 minutes
              you'll have shot patterns, court coverage, spacing analysis, and a
              coaching report. Ready to read between practices.
            </p>
            <div className="flex gap-6 flex-wrap text-sm text-ink-mute">
              <span>
                <span
                  className="text-ink font-medium mr-1 inline-block"
                  style={{ fontSize: '1.15rem', fontFamily: 'var(--font-display)', fontFeatureSettings: "'tnum'" }}
                >
                  60
                </span>
                sec to upload
              </span>
              <span>
                <span
                  className="text-ink font-medium mr-1 inline-block"
                  style={{ fontSize: '1.15rem', fontFamily: 'var(--font-display)', fontFeatureSettings: "'tnum'" }}
                >
                  15
                </span>
                min to analyze
              </span>
              <span>
                <span
                  className="text-ink font-medium mr-1 inline-block"
                  style={{ fontSize: '1.15rem', fontFamily: 'var(--font-display)', fontFeatureSettings: "'tnum'" }}
                >
                  5
                </span>
                min to your first pattern
              </span>
            </div>
          </div>

          <div
            className="w-full max-w-[400px] mx-auto bg-shade dark:bg-surface rounded-[18px] border-[1.5px] border-dashed border-line p-12 flex flex-col items-center justify-center gap-4 hover:border-ink-mute transition-colors"
            role="button"
            tabIndex={0}
            aria-label="Upload a recording"
          >
            <div
              className="w-15 h-15 rounded-2xl flex items-center justify-center text-court dark:text-court-light"
              style={{
                width: 60,
                height: 60,
                background: 'color-mix(in srgb, var(--color-court) 9%, transparent)',
              }}
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" width="26" height="26">
                <path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242" />
                <path d="M12 12v9" />
                <path d="m16 16-4-4-4 4" />
              </svg>
            </div>
            <div className="text-center">
              <div
                className="text-ink mb-1"
                style={{
                  fontFamily: 'var(--font-display)',
                  fontWeight: 500,
                  fontSize: '1.2rem',
                  letterSpacing: '-0.012em',
                }}
              >
                Drop your recording
              </div>
              <div className="text-ink-soft text-[0.92rem]">
                or{' '}
                <span className="text-court dark:text-court-light font-medium border-b border-current">
                  browse files
                </span>
              </div>
            </div>
            <div className="font-mono uppercase tracking-[0.12em] text-[0.68rem] text-ink-mute">
              MP4 · MOV · AVI · max 500 MB
            </div>
          </div>
        </div>
      </Link>

      {/* Empty preview cards */}
      <section className="mb-14">
        <div className="flex justify-between items-end mb-6 gap-6 flex-wrap">
          <h3
            className="text-ink"
            style={{
              fontFamily: 'var(--font-display)',
              fontWeight: 500,
              fontSize: 'clamp(28px, 3vw, 40px)',
              letterSpacing: '-0.014em',
            }}
          >
            What you'll see <em>after the first recording</em>.
          </h3>
          <p className="text-ink-mute text-[0.95rem] max-w-[38ch]">
            A glimpse of the dashboard. Upload to make it yours.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          <EmptyPreviewCard
            iconPath={
              <path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.5.5 0 0 1-.96 0L9.24 2.18a.5.5 0 0 0-.96 0l-2.35 8.36A2 2 0 0 1 4 12H2" />
            }
            title="Patterns, in plain English."
            body={`"Lin wins 71% of service games when she opens with a slice." We extract patterns from at least 2 recordings before surfacing them.`}
            postLabel="Earned after first recording"
          />
          <EmptyPreviewCard
            iconPath={
              <>
                <path d="M14.106 5.553a2 2 0 0 0 1.788 0l3.659-1.83A1 1 0 0 1 21 4.619v12.764a1 1 0 0 1-.553.894l-4.553 2.277a2 2 0 0 1-1.788 0l-4.212-2.106a2 2 0 0 0-1.788 0l-3.659 1.83A1 1 0 0 1 3 19.381V6.618a1 1 0 0 1 .553-.894l4.553-2.277a2 2 0 0 1 1.788 0Z" />
                <path d="M15 5.764v15" />
                <path d="M9 3.236v15" />
              </>
            }
            title="Shots, mapped to the court."
            body="Every bounce, every contact, every stroke type. Overlaid on a court tile so you can see what film won't tell you in 90 minutes."
            courtPreview
          />
          <EmptyPreviewCard
            iconPath={
              <>
                <path d="M3 3v18h18" />
                <path d="M18 17V9" />
                <path d="M13 17V5" />
                <path d="M8 17v-3" />
              </>
            }
            title="Spacing, called out."
            body={`"Squeezed" or "long." Was contact tight or reaching? We flag the technical patterns that matter for Wednesday's drills.`}
            postLabel="Earned after first recording"
          />
        </div>
      </section>

      {/* Closing */}
      <section className="text-center py-14">
        <h2
          className="text-ink mb-4 mx-auto"
          style={{
            fontFamily: 'var(--font-display)',
            fontWeight: 500,
            fontSize: 'clamp(32px, 4vw, 56px)',
            letterSpacing: '-0.018em',
            maxWidth: '28ch',
          }}
        >
          Drop your first recording.
        </h2>
        <p
          className="text-ink-soft mb-7 mx-auto"
          style={{ fontSize: '1.05rem', maxWidth: '50ch' }}
        >
          The dashboard fills in with every recording you upload. Patterns get
          sharper after the second. Spacing diagnostics get useful after the
          third.
        </p>
        <Link
          href="/upload"
          className="inline-flex items-center gap-2.5 px-6 py-3.5 rounded-full bg-court text-cream font-medium text-base hover:-translate-y-px transition-transform dark:bg-court-deep dark:hover:bg-court"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" width="18" height="18">
            <path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242" />
            <path d="M12 12v9" />
            <path d="m16 16-4-4-4 4" />
          </svg>
          Upload a match
        </Link>
      </section>
    </>
  );
}

function EmptyPreviewCard({
  iconPath,
  title,
  body,
  postLabel,
  courtPreview,
}: {
  iconPath: React.ReactNode;
  title: string;
  body: string;
  postLabel?: string;
  courtPreview?: boolean;
}) {
  return (
    <div className="cc-card flex flex-col gap-3.5 p-7 min-h-[280px] overflow-hidden relative">
      <div
        className="w-9 h-9 rounded-[10px] flex items-center justify-center text-court dark:text-court-light flex-shrink-0"
        style={{
          background: 'color-mix(in srgb, var(--color-court) 8%, transparent)',
        }}
      >
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.75"
          strokeLinecap="round"
          strokeLinejoin="round"
          width="18"
          height="18"
        >
          {iconPath}
        </svg>
      </div>
      <h4
        className="text-ink"
        style={{
          fontFamily: 'var(--font-display)',
          fontWeight: 500,
          fontSize: '1.25rem',
          letterSpacing: '-0.012em',
        }}
      >
        {title}
      </h4>
      <p className="text-ink-soft text-[0.93rem] leading-[1.5]">{body}</p>

      {courtPreview ? (
        <div
          className="w-full max-w-[200px] mx-auto mt-1 cc-court-tile opacity-50"
          aria-hidden
        >
          <svg viewBox="0 0 27 39" preserveAspectRatio="none" className="w-full h-full block">
            <rect x="1" y="1" width="25" height="37" fill="none" stroke="white" strokeWidth="0.3" opacity="0.6" />
            <line x1="1" y1="18" x2="26" y2="18" stroke="white" strokeWidth="0.25" opacity="0.5" />
            <line x1="13.5" y1="18" x2="13.5" y2="38" stroke="white" strokeWidth="0.25" opacity="0.5" />
          </svg>
        </div>
      ) : (
        <div className="mt-auto pt-3.5 border-t border-line-soft font-mono uppercase tracking-[0.1em] text-[0.7rem] text-ink-mute">
          {postLabel}
        </div>
      )}
    </div>
  );
}
