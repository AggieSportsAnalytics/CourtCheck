'use client';

import { RefObject, useEffect, useState } from 'react';

export type TimedNote = {
  timestamp_sec: number;
  text: string;
};

function fmtTs(sec: number): string {
  if (!Number.isFinite(sec) || sec < 0) sec = 0;
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

type Props = {
  videoRef: RefObject<HTMLVideoElement | null>;
  notes: TimedNote[];
  onAdd: (note: TimedNote) => void;
  onDelete: (originalIdx: number) => void;
  saving: boolean;
};

/**
 * Right-rail sticky notes card. Mirrors match-detail.html .notes-card:
 *   - sticky top: 92px so the coach can scroll while watching
 *   - input row: live-updating timestamp chip + text input + Add button
 *   - note list: lime-tinted ts chip, Newsreader body text, click-to-seek
 */
export default function NotesPanel({
  videoRef,
  notes,
  onAdd,
  onDelete,
  saving,
}: Props) {
  const [draft, setDraft] = useState('');
  const [nowSec, setNowSec] = useState(0);

  // Track current video time for the "now" timestamp chip. The <video>
  // element may mount AFTER this panel (parent renders both, but signed URLs
  // can lag), so we poll the ref on an interval until it's available.
  useEffect(() => {
    let detach: (() => void) | null = null;
    const attach = () => {
      const v = videoRef.current;
      if (!v) return false;
      const onTime = () => setNowSec(v.currentTime || 0);
      v.addEventListener('timeupdate', onTime);
      v.addEventListener('seeked', onTime);
      v.addEventListener('loadedmetadata', onTime);
      detach = () => {
        v.removeEventListener('timeupdate', onTime);
        v.removeEventListener('seeked', onTime);
        v.removeEventListener('loadedmetadata', onTime);
      };
      return true;
    };
    if (attach()) {
      return () => detach?.();
    }
    const tick = setInterval(() => {
      if (attach()) clearInterval(tick);
    }, 250);
    return () => {
      clearInterval(tick);
      detach?.();
    };
  }, [videoRef]);

  const sorted = [...notes]
    .map((n, i) => ({ ...n, originalIdx: i }))
    .sort((a, b) => a.timestamp_sec - b.timestamp_sec);

  const handleAdd = () => {
    const text = draft.trim();
    if (!text || !videoRef.current) return;
    onAdd({
      timestamp_sec: Math.round(videoRef.current.currentTime * 10) / 10,
      text,
    });
    setDraft('');
  };

  const handleSeek = (sec: number) => {
    if (videoRef.current) videoRef.current.currentTime = sec;
  };

  return (
    <div className="cc-notes-card bg-paper border border-line rounded-[14px] overflow-hidden flex flex-col min-w-0">
      <div className="flex-1 min-h-0 flex flex-col gap-4" style={{ padding: '22px 22px 24px' }}>
        <div className="flex justify-between items-center">
          <h3 className="font-display font-medium text-[1.15rem] leading-tight tracking-tight">
            Timed notes
          </h3>
          <span className="font-mono text-[0.66rem] uppercase tracking-[0.12em] text-ink-mute">
            {saving ? 'Saving' : 'Saved'} ·{' '}
            <span
              className="font-display font-medium text-[0.85rem] text-ink mx-0.5"
              style={{ letterSpacing: 0, textTransform: 'none', fontFeatureSettings: '"tnum"' }}
            >
              {notes.length}
            </span>{' '}
            notes
          </span>
        </div>

        {/* Input row: now-ts | input | add button */}
        <div className="grid items-center gap-2.5" style={{ gridTemplateColumns: 'auto 1fr auto' }}>
          <span
            className="inline-flex items-center px-3 py-1.5 rounded-full font-mono text-[0.74rem] font-medium"
            style={{
              background: '#E1E7A6',
              color: '#6E7522',
              letterSpacing: '0.06em',
              fontFeatureSettings: '"tnum"',
            }}
          >
            {fmtTs(nowSec)}
          </span>
          <input
            type="text"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleAdd();
            }}
            placeholder="Add a note at the current timestamp"
            aria-label="Add a note"
            className="w-full px-3.5 py-2.5 rounded-full bg-shade dark:bg-surface text-ink border border-transparent text-[0.92rem] outline-none focus:bg-paper focus:border-ink-mute placeholder:text-ink-mute"
            style={{ transition: 'border-color var(--duration-quick) var(--ease-out), background var(--duration-quick) var(--ease-out)' }}
          />
          <button
            type="button"
            onClick={handleAdd}
            disabled={!draft.trim() || !videoRef.current}
            className="inline-flex items-center gap-1.5 px-4 py-2 rounded-full bg-ink text-cream dark:bg-court-deep font-medium text-[0.86rem] hover:-translate-y-px disabled:opacity-35 disabled:cursor-not-allowed disabled:hover:translate-y-0"
            style={{ transition: 'opacity var(--duration-quick) var(--ease-out), transform var(--duration-quick) var(--ease-spring)' }}
          >
            <svg viewBox="0 0 24 24" width={14} height={14} fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
              <line x1="12" y1="5" x2="12" y2="19" />
              <line x1="5" y1="12" x2="19" y2="12" />
            </svg>
            Add note
          </button>
        </div>

        {/* Note list */}
        <div className="flex flex-col gap-1 flex-1 min-h-0 overflow-y-auto -mx-2 px-2">
          {sorted.length === 0 ? (
            <p className="text-sm text-ink-mute italic">
              Seek to a moment, write a note, hit enter.
            </p>
          ) : (
            sorted.map((n) => (
              <div
                key={n.originalIdx}
                className="cc-note-row group grid items-start gap-3.5 px-3.5 py-2.5 rounded-[10px] bg-shade dark:bg-surface cursor-pointer"
                style={{ gridTemplateColumns: 'auto 1fr auto', transition: 'background var(--duration-quick) var(--ease-out)' }}
                onMouseEnter={(e) => {
                  (e.currentTarget as HTMLElement).style.background =
                    'color-mix(in srgb, var(--color-shade) 80%, var(--color-paper))';
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget as HTMLElement).style.background = '';
                }}
                onClick={() => handleSeek(n.timestamp_sec)}
              >
                <span
                  className="inline-flex items-center px-2.5 py-1 rounded-full font-mono text-[0.72rem] font-medium mt-0.5"
                  style={{
                    background: '#E1E7A6',
                    color: '#6E7522',
                    letterSpacing: '0.06em',
                    fontFeatureSettings: '"tnum"',
                  }}
                >
                  {fmtTs(n.timestamp_sec)}
                </span>
                <span className="font-display font-normal text-[0.98rem] leading-snug text-ink">
                  {n.text}
                </span>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    onDelete(n.originalIdx);
                  }}
                  className="inline-flex items-center justify-center w-7 h-7 rounded-full border border-line bg-paper text-ink-mute hover:border-clay hover:text-clay hover:bg-paper cursor-pointer self-center shrink-0"
                  aria-label="Delete note"
                  title="Delete note"
                  style={{ transition: 'color var(--duration-quick) var(--ease-out), border-color var(--duration-quick) var(--ease-out)' }}
                >
                  <svg viewBox="0 0 24 24" width={14} height={14} fill="none" stroke="currentColor" strokeWidth={1.75} strokeLinecap="round" strokeLinejoin="round">
                    <line x1="18" y1="6" x2="6" y2="18" />
                    <line x1="6" y1="6" x2="18" y2="18" />
                  </svg>
                </button>
              </div>
            ))
          )}
        </div>
      </div>

      <style>{`
        .cc-notes-card {
          position: sticky;
          top: 92px;
          max-height: calc(100vh - 120px);
        }
        @media (max-width: 1000px) {
          .cc-notes-card {
            position: static;
            max-height: none;
          }
        }
      `}</style>
    </div>
  );
}
