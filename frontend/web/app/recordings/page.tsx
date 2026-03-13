'use client';

import { useEffect, useRef, useState } from 'react';
import Link from 'next/link';

interface Recording {
  id: string;
  status: 'pending' | 'processing' | 'done' | 'failed';
  progress: number;
  error: string | null;
  createdAt: string;
  name: string;
  filename: string;
  fps: number | null;
  numFrames: number | null;
  bounceCount: number | null;
  shotCount: number | null;
  rallyCount: number | null;
  forehandCount: number | null;
  backhandCount: number | null;
  serveCount: number | null;
  inBoundsBounces: number | null;
  outBoundsBounces: number | null;
  hasBounceHeatmap: boolean;
  hasPlayerHeatmap: boolean;
}

function formatDate(iso: string) {
  return new Date(iso).toLocaleDateString('en-US', {
    month: 'short', day: 'numeric', year: 'numeric',
  });
}

function formatDuration(fps: number | null, frames: number | null): string {
  if (!fps || !frames) return '—';
  const secs = Math.round(frames / fps);
  return `${Math.floor(secs / 60)}m ${secs % 60}s`;
}

function inPct(inB: number | null, outB: number | null): number | null {
  if (inB === null || outB === null) return null;
  const t = inB + outB;
  return t > 0 ? Math.round((inB / t) * 100) : null;
}

const STATUS_CONFIG = {
  done:       { label: 'Complete',   bg: 'rgba(34,197,94,0.08)',  text: '#4ADE80', dot: '#4ADE80'  },
  processing: { label: 'Processing', bg: 'rgba(234,179,8,0.08)',  text: '#FACC15', dot: '#FACC15' },
  pending:    { label: 'Pending',    bg: 'rgba(96,165,250,0.08)', text: '#60A5FA', dot: '#60A5FA'  },
  failed:     { label: 'Failed',     bg: 'rgba(239,68,68,0.08)',  text: '#F87171', dot: '#F87171'  },
};

function StatusPill({ status }: { status: Recording['status'] }) {
  const c = STATUS_CONFIG[status];
  return (
    <span
      className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-semibold"
      style={{ background: c.bg, color: c.text }}
    >
      <span className="w-1.5 h-1.5 rounded-full" style={{ background: c.dot }} />
      {c.label}
    </span>
  );
}

function StatChip({ label, value }: { label: string; value: string | number }) {
  return (
    <span
      className="text-[10px] px-1.5 py-0.5 rounded"
      style={{ background: 'rgba(255,255,255,0.05)', color: '#6B7280' }}
    >
      <span style={{ color: '#3A3A44' }}>{label} </span>{value}
    </span>
  );
}

function InlineName({
  recordingId,
  value,
  onRename,
}: {
  recordingId: string;
  value: string;
  onRename: (id: string, newName: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(value);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editing) inputRef.current?.select();
  }, [editing]);

  const save = async () => {
    const trimmed = draft.trim();
    if (!trimmed || trimmed === value) {
      setDraft(value);
      setEditing(false);
      return;
    }
    // Optimistic update
    onRename(recordingId, trimmed);
    setEditing(false);
    try {
      const res = await fetch(`/api/recordings/${recordingId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: trimmed }),
      });
      if (!res.ok) {
        // Rollback on failure
        onRename(recordingId, value);
        setDraft(value);
      }
    } catch {
      onRename(recordingId, value);
      setDraft(value);
    }
  };

  if (editing) {
    return (
      <input
        ref={inputRef}
        className="text-sm font-medium text-white bg-transparent border-b outline-none truncate max-w-50"
        style={{ borderColor: '#B4F000' }}
        value={draft}
        maxLength={100}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter') save();
          if (e.key === 'Escape') { setDraft(value); setEditing(false); }
        }}
        onBlur={save}
      />
    );
  }

  return (
    <button
      className="group/name flex items-center gap-1.5 text-sm font-medium text-white truncate cursor-text max-w-50"
      onClick={() => { setDraft(value); setEditing(true); }}
      title="Click to rename"
    >
      <span className="truncate">{value}</span>
      <svg
        viewBox="0 0 16 16"
        className="w-3 h-3 shrink-0 opacity-0 group-hover/name:opacity-40 transition-opacity"
        fill="currentColor"
      >
        <path d="M11.013 1.427a1.75 1.75 0 012.474 0l1.086 1.086a1.75 1.75 0 010 2.474l-8.61 8.61c-.21.21-.47.364-.756.445l-3.251.93a.75.75 0 01-.927-.928l.929-3.25a1.75 1.75 0 01.445-.758l8.61-8.61zm1.414 1.06a.25.25 0 00-.354 0L3.463 11.098a.25.25 0 00-.064.108l-.631 2.208 2.208-.63a.25.25 0 00.108-.064l8.61-8.61a.25.25 0 000-.354l-1.086-1.086z" />
      </svg>
    </button>
  );
}

export default function RecordingsPage() {
  const [recordings, setRecordings] = useState<Recording[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const handleRename = (id: string, newName: string) => {
    setRecordings((prev) =>
      prev.map((r) => (r.id === id ? { ...r, name: newName } : r))
    );
  };

  const handleDelete = async (id: string) => {
    setDeleteError(null);
    setDeletingId(id);
    setConfirmDeleteId(null);
    try {
      const res = await fetch(`/api/recordings/${id}`, { method: 'DELETE' });
      if (res.ok) {
        setRecordings((prev) => prev.filter((r) => r.id !== id));
      } else {
        setDeleteError('Failed to delete recording. Please try again.');
      }
    } finally {
      setDeletingId(null);
    }
  };

  useEffect(() => {
    fetch('/api/recordings')
      .then((r) => { if (!r.ok) throw new Error('Failed to fetch'); return r.json(); })
      .then((d) => setRecordings(d.recordings))
      .catch((e) => setError((e as Error).message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] gap-3">
        <div
          className="w-8 h-8 rounded-full animate-spin"
          style={{ border: '2px solid rgba(180,240,0,0.15)', borderTopColor: '#B4F000' }}
        />
        <p className="text-sm" style={{ color: '#5A5A66' }}>Loading recordings…</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="px-6 py-8 text-center">
        <p className="text-sm text-red-400">Error: {error}</p>
      </div>
    );
  }

  return (
    <div className="px-6 py-8 max-w-4xl">
      {/* Page header */}
      <div className="flex items-start justify-between mb-8">
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest mb-2" style={{ color: '#B4F000' }}>
            Library
          </p>
          <h1 className="text-3xl font-black tracking-tight text-white">Recordings</h1>
          <p className="text-sm mt-1" style={{ color: '#5A5A66' }}>
            {recordings.length === 0
              ? 'No recordings yet'
              : `${recordings.length} match${recordings.length !== 1 ? 'es' : ''}`}
          </p>
        </div>
        <Link
          href="/upload"
          className="mt-1 px-4 py-2 rounded-xl text-sm font-semibold transition-all"
          style={{ background: 'rgba(180,240,0,0.1)', color: '#B4F000', border: '1px solid rgba(180,240,0,0.15)' }}
          onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = 'rgba(180,240,0,0.18)'; }}
          onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = 'rgba(180,240,0,0.1)'; }}
        >
          + Upload
        </Link>
      </div>

      {deleteError && (
        <p className="text-sm text-red-400 mb-4">{deleteError}</p>
      )}

      {recordings.length === 0 && (
        <div
          className="rounded-2xl p-12 text-center"
          style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }}
        >
          <svg viewBox="0 0 24 24" className="w-10 h-10 mx-auto mb-4" fill="none" stroke="currentColor" strokeWidth={1.25} style={{ color: '#2A2A33' }}>
            <path d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
          </svg>
          <p className="text-white font-semibold mb-1">No recordings yet</p>
          <p className="text-sm mb-6" style={{ color: '#5A5A66' }}>Upload a tennis match to start your analysis.</p>
          <Link
            href="/upload"
            className="inline-block px-6 py-2.5 rounded-xl text-sm font-bold transition-all"
            style={{ background: '#B4F000', color: '#07070A' }}
            onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = '#C7FF00'; }}
            onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = '#B4F000'; }}
          >
            Upload Video
          </Link>
        </div>
      )}

      {recordings.length > 0 && (
        <div className="flex flex-col gap-2">
          {recordings.map((rec) => {
            const hasStats = rec.shotCount !== null || rec.bounceCount !== null;
            const accuracy = inPct(rec.inBoundsBounces, rec.outBoundsBounces);
            const heatmapCount = (rec.hasBounceHeatmap ? 1 : 0) + (rec.hasPlayerHeatmap ? 1 : 0);

            return (
              <div
                key={rec.id}
                className="group rounded-2xl p-4 transition-all duration-150"
                style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }}
                onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.border = '1px solid rgba(255,255,255,0.12)'; }}
                onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.border = '1px solid rgba(255,255,255,0.07)'; }}
              >
                <div className="flex items-start gap-4">
                  {/* Date column */}
                  <div className="shrink-0 w-9 text-center pt-0.5">
                    <p className="text-sm font-black leading-none" style={{ color: '#B4F000' }}>
                      {new Date(rec.createdAt).getDate()}
                    </p>
                    <p className="text-[10px] mt-0.5" style={{ color: '#3A3A44' }}>
                      {new Date(rec.createdAt).toLocaleDateString('en-US', { month: 'short' })}
                    </p>
                  </div>

                  {/* Main content */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap mb-1">
                      <InlineName recordingId={rec.id} value={rec.name} onRename={handleRename} />
                      <StatusPill status={rec.status} />
                      {heatmapCount > 0 && (
                        <span
                          className="text-[10px] px-1.5 py-0.5 rounded"
                          style={{ background: 'rgba(180,240,0,0.08)', color: '#B4F000' }}
                        >
                          {heatmapCount} heatmap{heatmapCount > 1 ? 's' : ''}
                        </span>
                      )}
                    </div>

                    <p className="text-xs mb-2" style={{ color: '#3A3A44' }}>
                      {formatDate(rec.createdAt)} · {formatDuration(rec.fps, rec.numFrames)}
                    </p>

                    {rec.status === 'done' && (
                      hasStats ? (
                        <div className="flex flex-wrap gap-1.5">
                          {rec.shotCount   !== null && <StatChip label="shots"    value={rec.shotCount.toLocaleString()} />}
                          {rec.bounceCount !== null && <StatChip label="bounces"  value={rec.bounceCount.toLocaleString()} />}
                          {rec.rallyCount  !== null && <StatChip label="rallies"  value={rec.rallyCount.toLocaleString()} />}
                          {accuracy        !== null && <StatChip label="accuracy" value={`${accuracy}%`} />}
                          {rec.forehandCount !== null && <StatChip label="FH"  value={rec.forehandCount.toLocaleString()} />}
                          {rec.backhandCount !== null && <StatChip label="BH"  value={rec.backhandCount.toLocaleString()} />}
                          {rec.serveCount    !== null && <StatChip label="srv" value={rec.serveCount.toLocaleString()} />}
                        </div>
                      ) : (
                        <p className="text-[10px] italic" style={{ color: '#2A2A33' }}>
                          Stats not available — processed before the analytics update
                        </p>
                      )
                    )}

                    {rec.status === 'processing' && (
                      <div className="mt-2 max-w-xs">
                        <div className="h-1 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.08)' }}>
                          <div
                            className="h-full rounded-full transition-all"
                            style={{ width: `${Math.round(rec.progress * 100)}%`, background: '#B4F000' }}
                          />
                        </div>
                        <p className="text-[10px] mt-1" style={{ color: '#3A3A44' }}>{Math.round(rec.progress * 100)}%</p>
                      </div>
                    )}

                    {rec.status === 'failed' && rec.error && (
                      <p className="text-xs text-red-400 mt-1">{rec.error}</p>
                    )}
                  </div>

                  {/* Action */}
                  <div className="shrink-0 flex items-center gap-2">
                    {rec.status === 'done' && (
                      <Link
                        href={`/recordings/${rec.id}`}
                        className="px-3 py-1.5 rounded-lg text-xs font-semibold transition-all"
                        style={{ background: 'rgba(180,240,0,0.08)', color: '#B4F000', border: '1px solid rgba(180,240,0,0.15)' }}
                        onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = 'rgba(180,240,0,0.16)'; }}
                        onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = 'rgba(180,240,0,0.08)'; }}
                      >
                        View →
                      </Link>
                    )}

                    {confirmDeleteId === rec.id ? (
                      <div className="flex items-center gap-1.5">
                        <button
                          onClick={() => handleDelete(rec.id)}
                          disabled={deletingId === rec.id}
                          className="px-2 py-1 rounded-lg text-xs font-semibold transition-all"
                          style={{ background: 'rgba(239,68,68,0.12)', color: '#F87171', border: '1px solid rgba(239,68,68,0.2)' }}
                        >
                          {deletingId === rec.id ? '…' : 'Delete'}
                        </button>
                        <button
                          onClick={() => setConfirmDeleteId(null)}
                          disabled={deletingId === rec.id}
                          className="px-2 py-1 rounded-lg text-xs transition-all disabled:opacity-40"
                          style={{ color: '#5A5A66' }}
                        >
                          Cancel
                        </button>
                      </div>
                    ) : (
                      <button
                        onClick={() => setConfirmDeleteId(rec.id)}
                        className="p-1.5 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer"
                        style={{ color: '#3A3A44' }}
                        title="Delete recording"
                        aria-label="Delete recording"
                      >
                        <svg viewBox="0 0 16 16" className="w-3.5 h-3.5" fill="currentColor">
                          <path d="M6.5 1.75a.25.25 0 01.25-.25h2.5a.25.25 0 01.25.25V3h-3V1.75zm4.5 0V3h2.25a.75.75 0 010 1.5H2.75a.75.75 0 010-1.5H5V1.75C5 .784 5.784 0 6.75 0h2.5C10.216 0 11 .784 11 1.75zM4.496 6.675a.75.75 0 10-1.492.15l.66 6.6A1.75 1.75 0 005.405 15h5.19a1.75 1.75 0 001.741-1.575l.66-6.6a.75.75 0 00-1.492-.15l-.66 6.6a.25.25 0 01-.249.225H5.405a.25.25 0 01-.249-.225l-.66-6.6z" />
                        </svg>
                      </button>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
