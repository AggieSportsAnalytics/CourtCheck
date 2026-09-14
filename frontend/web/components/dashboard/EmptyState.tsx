'use client';

import { Prose } from '@/components/ui/display';

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
          className="inline-flex items-center gap-2 font-mono uppercase tracking-[0.12em] text-[0.82rem] text-court"
        >
          <span
            aria-hidden
            className="w-1.5 h-1.5 rounded-full bg-clay"
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
          Welcome, {coachName}.
        </h1>
        <Prose
          className="text-ink-soft mt-4 max-w-[60ch] text-[1.2rem]"
        >
          Upload a recording and CourtCheck tracks every shot, bounce, and stroke, then writes up what a coach would notice. Processing takes about 15 minutes.
        </Prose>
        <div className="flex gap-3.5 items-center flex-wrap mt-7">
          <Link
            href="/upload"
            className="inline-flex items-center gap-2.5 px-6 py-3.5 rounded-full bg-court text-cream font-medium text-base transition-transform hover:-translate-y-px"
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
              className="inline-flex items-center gap-2 font-mono uppercase tracking-[0.12em] text-[0.82rem] text-court"
            >
              <span
                aria-hidden
                className="w-1.5 h-1.5 rounded-full bg-clay"
              />
              Upload a recording
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
              Your first recording.
            </h2>
            <Prose className="text-ink-soft mb-6 max-w-[36ch]">
              Upload an MP4. In about 15 minutes you get the shot map, court coverage, contact spacing, and a written report.
            </Prose>
            <div className="flex gap-6 flex-wrap text-[0.95rem] text-ink-mute">
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
                min to read the report
              </span>
            </div>
          </div>

          <div
            className="w-full max-w-[400px] mx-auto bg-shade rounded-[18px] border-[1.5px] border-dashed border-line p-12 flex flex-col items-center justify-center gap-4 hover:border-ink-mute transition-colors"
            role="button"
            tabIndex={0}
            aria-label="Upload a recording"
          >
            <div
              className="w-15 h-15 rounded-2xl flex items-center justify-center text-court"
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
              Open upload
              </div>
              <div className="text-ink-soft text-[1.02rem]">
                or{' '}
                <span className="text-court font-medium border-b border-current">
                  choose a recording
                </span>
              </div>
            </div>
            <div className="font-mono uppercase tracking-[0.12em] text-[0.82rem] text-ink-mute">
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
            What a processed recording shows.
          </h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          <EmptyPreviewCard
            iconPath={
              <path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.5.5 0 0 1-.96 0L9.24 2.18a.5.5 0 0 0-.96 0l-2.35 8.36A2 2 0 0 1 4 12H2" />
            }
            title="Patterns in writing"
            body="A written summary of shot placement, positioning, and errors in the recording, with a suggested practice adjustment."
            postLabel="Available after processing"
          />
          <EmptyPreviewCard
            iconPath={
              <>
                <path d="M14.106 5.553a2 2 0 0 0 1.788 0l3.659-1.83A1 1 0 0 1 21 4.619v12.764a1 1 0 0 1-.553.894l-4.553 2.277a2 2 0 0 1-1.788 0l-4.212-2.106a2 2 0 0 0-1.788 0l-3.659 1.83A1 1 0 0 1 3 19.381V6.618a1 1 0 0 1 .553-.894l4.553-2.277a2 2 0 0 1 1.788 0Z" />
                <path d="M15 5.764v15" />
                <path d="M9 3.236v15" />
              </>
            }
            title="Shot map"
            body="Each bounce and contact, colored by stroke, drawn on a court."
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
            title="Contact spacing"
            body={`How far the player was from the ball at contact: jammed, squeezed, ideal, or reaching.`}
            postLabel="Available after processing"
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
          Upload your first recording.
        </h2>
        <Prose
          className="text-ink-soft mb-7 mx-auto"
          style={{ fontVariationSettings: '"opsz" 16', maxWidth: '50ch' }}
        >
          Each processed recording adds court maps, stroke counts, and a written report to your dashboard.
        </Prose>
        <Link
          href="/upload"
          className="inline-flex items-center gap-2.5 px-6 py-3.5 rounded-full bg-court text-cream font-medium text-base hover:-translate-y-px transition-transform"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" width="18" height="18">
            <path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242" />
            <path d="M12 12v9" />
            <path d="m16 16-4-4-4 4" />
          </svg>
          Upload a recording
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
        className="w-9 h-9 rounded-[10px] flex items-center justify-center text-court flex-shrink-0"
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
        <div className="mt-auto pt-3.5 border-t border-line-soft font-mono uppercase tracking-[0.1em] text-[0.82rem] text-ink-mute">
          {postLabel}
        </div>
      )}
    </div>
  );
}
