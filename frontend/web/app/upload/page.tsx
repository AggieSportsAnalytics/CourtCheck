'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useDropzone } from 'react-dropzone';
import { useVideoUpload } from '@/hooks';
import { APP_CONFIG } from '@/constants';
import BounceLoader from '@/components/upload/BounceLoader';
import CheckmarkBurst from '@/components/upload/CheckmarkBurst';

interface Player {
  id: string;
  name: string;
  year: string | null;
}

const MAX_MB = APP_CONFIG.MAX_VIDEO_SIZE / (1024 * 1024);

function formatMB(bytes: number): string {
  return (bytes / (1024 * 1024)).toFixed(1);
}

export default function UploadPage() {
  const router = useRouter();
  const [players, setPlayers] = useState<Player[]>([]);
  const [selectedPlayerId, setSelectedPlayerId] = useState<string | null>(null);
  const [recordingTitle, setRecordingTitle] = useState('');
  const [matchDate, setMatchDate] = useState('');

  const upload = useVideoUpload(undefined, selectedPlayerId);
  const {
    status,
    match_id,
    uploadProgress,
    bytesUploaded,
    bytesTotal,
    progress,
    stage,
    error,
    filename,
    handleFile,
    reset,
  } = upload;

  // Fetch player list once
  useEffect(() => {
    fetch('/api/players')
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (data?.players) setPlayers(data.players);
      })
      .catch(() => {});
  }, []);

  // Dropzone — only active in the idle state
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      const file = acceptedFiles[0];
      if (!file) return;
      void handleFile(file, { name: recordingTitle, matchDate });
    },
    [handleFile, recordingTitle, matchDate],
  );

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop,
    accept: { 'video/*': APP_CONFIG.SUPPORTED_VIDEO_FORMATS },
    maxFiles: 1,
    maxSize: APP_CONFIG.MAX_VIDEO_SIZE,
    disabled: status !== 'idle' && status !== 'failed',
    noClick: true,
    noKeyboard: true,
  });

  // Resolve which state pane to show. `creating upload`, `uploading` → uploading.
  // `pending`, `processing` → processing. `done`, `failed`, `idle` are 1:1.
  const pane: 'idle' | 'uploading' | 'processing' | 'done' | 'failed' = useMemo(() => {
    if (status === 'idle') return 'idle';
    if (status === 'creating upload' || status === 'uploading') return 'uploading';
    if (status === 'pending' || status === 'processing') return 'processing';
    if (status === 'done') return 'done';
    if (status === 'failed') return 'failed';
    return 'idle';
  }, [status]);

  const uploadPct = Math.round(uploadProgress * 100);
  const procPct = Math.round(progress * 100);

  // Diagnostic: log every time the processing-state inputs change so we can
  // tell from devtools whether Modal -> Supabase -> /api/status -> hook is
  // actually flowing. Look for `[Upload]` lines; procPct should rise + stage
  // should change every few seconds while the recording is processing.
  const lastLogRef = useRef<string>('');
  useEffect(() => {
    if (status !== 'processing' && status !== 'pending') return;
    const key = `${status}|${procPct}|${stage ?? 'null'}`;
    if (key === lastLogRef.current) return;
    lastLogRef.current = key;
    // eslint-disable-next-line no-console
    console.log(`[Upload] status=${status} progress=${procPct}% stage=${stage ?? 'null'}`);
  }, [status, procPct, stage]);

  // Processing phase headline — italics on the verb, clay via the global em rule.
  // The backend-reported `stage` is authoritative when present; the percent
  // buckets are only a fallback for the brief window before the pipeline has
  // written its first stage label.
  const phase = useMemo(() => {
    const derived =
      procPct < 5
        ? { verb: 'Loading', rest: ' the recording.', stage: 'Calibrating the court' }
        : procPct < 45
          ? { verb: 'Tracking', rest: ' every shot, frame by frame.', stage: 'Following the ball and players' }
          : procPct < 50
            ? { verb: 'Identifying', rest: ' bounces and strokes.', stage: 'Detecting bounce points and stroke types' }
            : procPct < 95
              ? { verb: 'Drawing', rest: ' your recording overlay.', stage: 'Rendering your annotated recording' }
              : { verb: 'Reading', rest: ' your tendencies.', stage: 'Generating heatmaps and scouting report' };
    return stage ? { ...derived, stage } : derived;
  }, [procPct, stage]);

  const features = [
    { title: 'Every shot', desc: 'Tracked and labeled. Forehand, backhand, serve, volley.' },
    { title: 'Every bounce', desc: 'Exact landing point on the court. In and out calls included.' },
    { title: 'Every movement', desc: 'Where your player covered the court. And where they didn’t.' },
    { title: 'Repeating patterns', desc: 'Sequences that show up more than once. Called out for you.' },
  ];

  // Once an upload starts (processing / done / failed) the page transforms
  // into a status view — drop the marketing header + features grid so the
  // whole flow fits in one viewport without scrolling.
  const compact = pane !== 'idle';

  return (
    <div className={`mx-auto max-w-[760px] px-6 ${compact ? 'py-6' : 'py-10'}`}>
      {/* Page head — compact during processing */}
      <div className={compact ? 'pb-4' : 'pb-8'}>
        <span className="inline-flex items-center gap-2 font-mono text-[0.72rem] uppercase tracking-[0.18em] text-court before:size-[6px] before:rounded-full before:bg-clay before:content-['']">
          Analyze
        </span>
        <h1
          className={`font-display font-medium leading-none tracking-[-0.022em] ${
            compact ? 'mt-2 mb-1 text-[clamp(24px,3vw,36px)]' : 'mt-4 mb-3 text-[clamp(40px,5.2vw,64px)]'
          }`}
          style={{ fontVariationSettings: '"opsz" 72' }}
        >
          Upload a <em>recording</em>.
        </h1>
        {!compact && (
          <p className="max-w-[56ch] text-[1.1rem] text-ink-soft">
            Drop in your recording. We read every shot, every bounce, every pattern. You stop rewinding.
          </p>
        )}
      </div>

      {/* Optional metadata — only visible in idle/failed states */}
      {(pane === 'idle' || pane === 'failed') && (
        <form className="mb-7 grid gap-4" onSubmit={(e) => e.preventDefault()}>
          <div className="grid gap-1.5">
            <label
              htmlFor="player"
              className="font-mono text-[0.7rem] uppercase tracking-[0.14em] text-ink-mute"
            >
              Near-side player
            </label>
            <select
              id="player"
              value={selectedPlayerId ?? ''}
              onChange={(e) => setSelectedPlayerId(e.target.value || null)}
              className="w-full rounded-[10px] border border-line bg-paper px-3.5 py-3 text-[0.95rem] text-ink outline-none transition-colors duration-150 focus:border-ink focus:bg-surface"
            >
              <option value="">Unknown / not assigned</option>
              {players.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                  {p.year ? ` · ${p.year}` : ''}
                </option>
              ))}
            </select>
          </div>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div className="grid gap-1.5">
              <label
                htmlFor="clip-title"
                className="font-mono text-[0.7rem] uppercase tracking-[0.14em] text-ink-mute"
              >
                Recording title
              </label>
              <input
                id="clip-title"
                type="text"
                value={recordingTitle}
                onChange={(e) => setRecordingTitle(e.target.value)}
                placeholder="e.g. Lin vs Stanford · Set 1"
                className="w-full rounded-[10px] border border-line bg-paper px-3.5 py-3 text-[0.95rem] text-ink outline-none transition-colors duration-150 placeholder:text-ink-mute focus:border-ink focus:bg-surface"
              />
            </div>
            <div className="grid gap-1.5">
              <label
                htmlFor="match-date"
                className="font-mono text-[0.7rem] uppercase tracking-[0.14em] text-ink-mute"
              >
                Match date
              </label>
              <input
                id="match-date"
                type="date"
                value={matchDate}
                onChange={(e) => setMatchDate(e.target.value)}
                className="w-full rounded-[10px] border border-line bg-paper px-3.5 py-3 text-[0.95rem] text-ink outline-none transition-colors duration-150 focus:border-ink focus:bg-surface"
              />
            </div>
          </div>
        </form>
      )}

      {/* Upload card — single container, swaps panes by state.
          min-height keeps the idle/uploading/processing/done panes at parity
          height-wise so the layout doesn't jump as the state machine
          progresses (idle dropzone was visibly taller than processing). */}
      <div
        {...(pane === 'idle' ? getRootProps() : {})}
        className={`overflow-hidden rounded-[14px] border bg-paper transition-colors duration-200 flex flex-col justify-center ${
          pane === 'idle'
            ? `border-dashed ${isDragActive ? 'border-court bg-[color-mix(in_srgb,var(--color-court)_4%,var(--color-paper))]' : 'border-line'}`
            : pane === 'failed'
              ? 'border-clay'
              : 'border-line'
        }`}
        style={{ minHeight: 520 }}
        data-state={pane}
      >
        {pane === 'idle' && <input {...getInputProps()} aria-label="Recording file input" />}

        {pane === 'idle' && (
          <div className="px-8 py-14 text-center">
            <div className="mx-auto mb-6 flex size-14 items-center justify-center rounded-2xl bg-shade text-court dark:bg-surface dark:text-court-light">
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="size-6"
              >
                <path d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9" />
                <path d="M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
            </div>
            <h3
              className="mb-2 font-display text-[1.6rem] font-medium tracking-[-0.014em]"
              style={{ fontVariationSettings: '"opsz" 60' }}
            >
              Drop in your recording.
            </h3>
            <p className="mb-4 text-[0.95rem] text-ink-soft">
              or{' '}
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  open();
                }}
                className="border-b border-court font-medium text-court dark:border-court-light dark:text-court-light"
              >
                browse files
              </button>
            </p>
            <p className="font-mono text-[0.72rem] uppercase tracking-[0.14em] text-ink-mute">
              MP4 · MOV · AVI · max {MAX_MB} MB
            </p>
          </div>
        )}

        {pane === 'uploading' && (
          <div className="px-8 py-10 text-center">
            <BounceLoader size={320} />
            <p className="mb-4 font-mono text-[0.72rem] uppercase tracking-[0.14em] text-ink-mute">
              Uploading
            </p>
            <h3
              className="mx-auto mb-6 max-w-[28ch] font-display text-[1.4rem] font-medium tracking-[-0.014em]"
              style={{ fontVariationSettings: '"opsz" 60' }}
            >
              Uploading <em>your recording</em>.
            </h3>
            {filename && (
              <div className="mx-auto mb-7 inline-flex max-w-full items-center gap-2.5 rounded-full bg-shade px-3.5 py-2 text-[0.85rem] dark:bg-surface">
                <span className="max-w-[280px] truncate font-medium">{filename}</span>
                {bytesTotal > 0 && (
                  <span className="font-mono text-[0.74rem] text-ink-mute">
                    {formatMB(bytesTotal)} MB
                  </span>
                )}
              </div>
            )}
            <div className="mx-auto h-[4px] max-w-[360px] overflow-hidden rounded-full bg-shade dark:bg-surface">
              <div
                className="h-full rounded-full bg-court transition-[width] duration-200"
                style={{ width: `${Math.max(0, Math.min(100, uploadPct))}%` }}
              />
            </div>
            <div className="mx-auto mt-2.5 flex max-w-[360px] justify-between text-[0.78rem] text-ink-mute">
              <span>
                {bytesTotal > 0
                  ? `${formatMB(bytesUploaded)} MB / ${formatMB(bytesTotal)} MB`
                  : 'Preparing upload.'}
              </span>
              <span className="font-mono font-medium">{uploadPct}%</span>
            </div>
          </div>
        )}

        {pane === 'processing' && (
          <div className="px-8 py-10 text-center">
            <BounceLoader size={320} />
            <h3
              className="mt-3 mb-2 min-h-[1.5em] font-display text-[1.4rem] font-medium tracking-[-0.014em] transition-opacity duration-200"
              style={{ fontVariationSettings: '"opsz" 72' }}
              key={phase.verb}
            >
              <em>{phase.verb}</em>
              {phase.rest}
            </h3>

            {/* Progress bar — always visible. Indeterminate sweep when progress = 0. */}
            <div className="mx-auto mt-5 h-[4px] max-w-[360px] overflow-hidden rounded-full bg-shade dark:bg-surface">
              {progress > 0 ? (
                <div
                  className="h-full rounded-full bg-court transition-[width] duration-300 ease-out"
                  style={{ width: `${Math.max(2, Math.min(100, procPct))}%` }}
                />
              ) : (
                <div
                  className="h-full w-1/3 rounded-full bg-court"
                  style={{ animation: 'cc-indeterminate 1.4s ease-in-out infinite' }}
                />
              )}
            </div>
            <div className="mx-auto mt-2 flex max-w-[360px] items-center justify-between font-mono text-[0.72rem] uppercase tracking-[0.14em] text-ink-mute">
              <span>{phase.stage}</span>
              <span className="font-medium">
                {progress > 0 ? `${procPct}%` : 'STARTING'}
              </span>
            </div>

            {match_id && (
              <p className="mt-5 font-mono text-[0.7rem] uppercase tracking-[0.1em] text-ink-mute">
                Recording · {match_id.slice(0, 8).toUpperCase()}
              </p>
            )}

            <style>{`
              @keyframes cc-indeterminate {
                0%   { transform: translateX(-100%); }
                50%  { transform: translateX(220%); }
                100% { transform: translateX(420%); }
              }
              @media (prefers-reduced-motion: reduce) {
                @keyframes cc-indeterminate {
                  0%, 100% { transform: translateX(120%); }
                }
              }
            `}</style>
          </div>
        )}

        {pane === 'done' && (
          <div className="px-8 py-6 text-center">
            <CheckmarkBurst />
            <h3
              className="mt-3 mb-2 font-display text-[1.6rem] font-medium tracking-[-0.018em]"
              style={{ fontVariationSettings: '"opsz" 72' }}
            >
              Recording <em>analyzed</em>.
            </h3>
            <p className="mb-5 text-[0.95rem] text-ink-soft">
              Open the recording to see the read.
            </p>
            <div className="flex flex-wrap items-center justify-center gap-3">
              <button
                type="button"
                onClick={() => match_id && router.push(`/recordings/${match_id}`)}
                className="inline-flex items-center gap-2.5 rounded-full bg-ink px-[22px] py-3 text-[0.95rem] font-medium text-cream transition-transform duration-150 ease-out hover:-translate-y-px dark:bg-court-deep"
              >
                Open recording
                <span aria-hidden>→</span>
              </button>
              <button
                type="button"
                onClick={reset}
                className="inline-flex items-center gap-2.5 rounded-full border border-line bg-paper px-[22px] py-3 text-[0.95rem] font-medium text-ink transition-colors duration-200 ease-out hover:border-ink"
              >
                Upload another
              </button>
            </div>
          </div>
        )}

        {pane === 'failed' && (
          <div className="px-8 py-14 text-center">
            <div className="mx-auto mb-6 flex size-14 items-center justify-center rounded-2xl bg-shade text-clay dark:bg-surface">
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="size-6"
              >
                <circle cx="12" cy="12" r="9" />
                <line x1="12" y1="8" x2="12" y2="13" />
                <line x1="12" y1="16" x2="12.01" y2="16" />
              </svg>
            </div>
            <h3
              className="mb-2 font-display text-[1.5rem] font-medium tracking-[-0.014em] text-clay"
              style={{ fontVariationSettings: '"opsz" 60' }}
            >
              Upload didn{'’'}t finish.
            </h3>
            <p className="mx-auto mb-6 max-w-[42ch] text-[0.95rem] text-ink-soft">
              {error || 'Something went wrong on our end. Try the upload again.'}
            </p>
            <button
              type="button"
              onClick={reset}
              className="inline-flex items-center gap-2.5 rounded-full bg-ink px-[22px] py-3 text-[0.95rem] font-medium text-cream transition-transform duration-150 ease-out hover:-translate-y-px dark:bg-court-deep"
            >
              Try again
            </button>
          </div>
        )}
      </div>

      {/* What we analyze — always shown. Tighter padding + smaller icons in
          compact mode so both the processing card and this section fit in one
          viewport without scrolling. */}
      <div className={`rounded-[14px] border border-line bg-paper ${compact ? 'mt-4 p-4' : 'mt-9 p-7'}`}>
        <p className={`font-mono uppercase tracking-[0.14em] text-ink-mute ${compact ? 'mb-3 text-[0.66rem]' : 'mb-5 text-[0.72rem]'}`}>
          What we analyze
        </p>
        <div className={`grid grid-cols-1 sm:grid-cols-2 ${compact ? 'gap-2.5' : 'gap-5'}`}>
          {features.map((f) => (
            <div key={f.title} className={`flex items-start ${compact ? 'gap-2.5' : 'gap-3.5'}`}>
              <div
                className={`shrink-0 flex items-center justify-center rounded-[8px] bg-[color-mix(in_srgb,var(--color-court)_8%,transparent)] text-court dark:bg-[color-mix(in_srgb,var(--color-court-light)_14%,transparent)] dark:text-court-light ${compact ? 'size-7' : 'size-9'}`}
              >
                <svg
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className={compact ? 'size-3.5' : 'size-4'}
                >
                  <circle cx="12" cy="12" r="9" />
                </svg>
              </div>
              <div>
                <h5
                  className={`font-display font-medium tracking-[-0.01em] ${compact ? 'text-[0.92rem] leading-tight' : 'text-[1.05rem]'}`}
                  style={{ fontVariationSettings: '"opsz" 60' }}
                >
                  {f.title}
                </h5>
                <p className={`text-ink-soft ${compact ? 'text-[0.78rem] leading-snug mt-0.5' : 'text-[0.88rem] leading-[1.5]'}`}>{f.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
