'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';

interface Recording {
  id: string;
  status: 'pending' | 'processing' | 'done' | 'failed';
  progress: number;
  error: string | null;
  createdAt: string;
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
  done:       { label: 'Complete',   bg: 'bg-green-500/15',  text: 'text-green-400',  dot: 'bg-green-400'  },
  processing: { label: 'Processing', bg: 'bg-yellow-500/15', text: 'text-yellow-400', dot: 'bg-yellow-400' },
  pending:    { label: 'Pending',    bg: 'bg-blue-500/15',   text: 'text-blue-400',   dot: 'bg-blue-400'   },
  failed:     { label: 'Failed',     bg: 'bg-red-500/15',    text: 'text-red-400',    dot: 'bg-red-400'    },
};

function StatusPill({ status }: { status: Recording['status'] }) {
  const c = STATUS_CONFIG[status];
  return (
    <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium ${c.bg} ${c.text}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${c.dot}`} />
      {c.label}
    </span>
  );
}

function StatChip({ label, value }: { label: string; value: string | number }) {
  return (
    <span className="bg-gray-800/80 text-gray-300 text-[10px] px-2 py-0.5 rounded">
      <span className="text-gray-500 mr-0.5">{label}</span>{value}
    </span>
  );
}

export default function RecordingsPage() {
  const [recordings, setRecordings] = useState<Recording[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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
        <div className="w-8 h-8 border-2 border-accent border-t-transparent rounded-full animate-spin" />
        <p className="text-sm text-gray-400">Loading recordings…</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="px-5 py-8 text-center">
        <p className="text-red-400 text-sm">Error: {error}</p>
      </div>
    );
  }

  return (
    <div className="px-5 py-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold text-white">Recordings</h2>
          <p className="text-sm text-gray-400 mt-0.5">
            {recordings.length === 0 ? 'No recordings yet' : `${recordings.length} match${recordings.length !== 1 ? 'es' : ''}`}
          </p>
        </div>
        <Link
          href="/upload"
          className="px-4 py-2 bg-accent/10 hover:bg-accent/20 border border-accent/20 rounded-xl text-sm text-accent font-medium transition-colors"
        >
          + Upload
        </Link>
      </div>

      {recordings.length === 0 && (
        <div className="bg-secondary rounded-2xl border border-gray-700/40 p-10 text-center">
          <svg viewBox="0 0 24 24" className="w-10 h-10 text-gray-600 mx-auto mb-3" fill="none" stroke="currentColor" strokeWidth={1.5}>
            <path d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
          </svg>
          <p className="text-white font-semibold mb-1">No recordings yet</p>
          <p className="text-sm text-gray-400 mb-4">Upload a tennis match to start your analysis.</p>
          <Link href="/upload" className="inline-block px-5 py-2 bg-accent text-primary font-semibold rounded-lg text-sm">
            Upload Video
          </Link>
        </div>
      )}

      {recordings.length > 0 && (
        <div className="flex flex-col gap-3">
          {recordings.map((rec) => {
            const hasStats = rec.shotCount !== null || rec.bounceCount !== null;
            const accuracy = inPct(rec.inBoundsBounces, rec.outBoundsBounces);
            const heatmapCount = (rec.hasBounceHeatmap ? 1 : 0) + (rec.hasPlayerHeatmap ? 1 : 0);

            return (
              <div
                key={rec.id}
                className="bg-secondary rounded-2xl border border-gray-700/40 p-4 hover:border-gray-600/60 transition-colors"
              >
                <div className="flex items-start gap-4">
                  {/* Date column */}
                  <div className="shrink-0 w-10 text-center pt-0.5">
                    <p className="text-sm font-bold text-accent leading-none">
                      {new Date(rec.createdAt).getDate()}
                    </p>
                    <p className="text-[10px] text-gray-500 mt-0.5">
                      {new Date(rec.createdAt).toLocaleDateString('en-US', { month: 'short' })}
                    </p>
                  </div>

                  {/* Main content */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap mb-1">
                      <p className="text-sm font-medium text-white truncate">{rec.filename}</p>
                      <StatusPill status={rec.status} />
                      {heatmapCount > 0 && (
                        <span className="bg-accent/10 text-accent text-[10px] px-1.5 py-0.5 rounded">
                          {heatmapCount} heatmap{heatmapCount > 1 ? 's' : ''}
                        </span>
                      )}
                    </div>

                    <p className="text-xs text-gray-500 mb-2">{formatDate(rec.createdAt)} · {formatDuration(rec.fps, rec.numFrames)}</p>

                    {/* Tennis stats chips */}
                    {rec.status === 'done' && (
                      hasStats ? (
                        <div className="flex flex-wrap gap-1.5">
                          {rec.shotCount   !== null && <StatChip label="shots"   value={rec.shotCount.toLocaleString()} />}
                          {rec.bounceCount !== null && <StatChip label="bounces" value={rec.bounceCount.toLocaleString()} />}
                          {rec.rallyCount  !== null && <StatChip label="rallies" value={rec.rallyCount.toLocaleString()} />}
                          {accuracy        !== null && <StatChip label="accuracy" value={`${accuracy}%`} />}
                          {rec.forehandCount !== null && <StatChip label="FH" value={rec.forehandCount.toLocaleString()} />}
                          {rec.backhandCount !== null && <StatChip label="BH" value={rec.backhandCount.toLocaleString()} />}
                          {rec.serveCount    !== null && <StatChip label="srv" value={rec.serveCount.toLocaleString()} />}
                        </div>
                      ) : (
                        <p className="text-[10px] text-gray-600 italic">
                          Stats not available — processed before the analytics update
                        </p>
                      )
                    )}

                    {/* Processing bar */}
                    {rec.status === 'processing' && (
                      <div className="mt-2 max-w-xs">
                        <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-accent rounded-full transition-all"
                            style={{ width: `${Math.round(rec.progress * 100)}%` }}
                          />
                        </div>
                        <p className="text-xs text-gray-500 mt-0.5">{Math.round(rec.progress * 100)}%</p>
                      </div>
                    )}

                    {rec.status === 'failed' && rec.error && (
                      <p className="text-xs text-red-400 mt-1">{rec.error}</p>
                    )}
                  </div>

                  {/* Action */}
                  <div className="shrink-0">
                    {rec.status === 'done' && (
                      <Link
                        href={`/recordings/${rec.id}`}
                        className="px-3 py-1.5 bg-accent/10 hover:bg-accent/20 border border-accent/20 rounded-lg text-xs text-accent font-medium transition-colors"
                      >
                        View →
                      </Link>
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
