'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';

interface TennisStats {
  totalShots: number;
  totalBounces: number;
  totalRallies: number;
  totalForehands: number;
  totalBackhands: number;
  totalServes: number;
  totalInBounds: number;
  totalOutBounds: number;
}

interface GameEntry {
  id: string;
  createdAt: string;
  durationSeconds: number | null;
  shotCount: number | null;
  bounceCount: number | null;
  rallyCount: number | null;
  forehandCount: number | null;
  backhandCount: number | null;
  serveCount: number | null;
  inBounces: number | null;
  outBounces: number | null;
  hasBallHeatmap: boolean;
  hasPlayerHeatmap: boolean;
}

interface Summary {
  totals: { total: number; done: number; processing: number; failed: number };
  tennisStats: TennisStats;
  hasTennisStats: boolean;
  totalGameplaySeconds: number;
  avgDurationSeconds: number;
  withHeatmapsCount: number;
  games: GameEntry[];
}

function fmtDur(s: number | null): string {
  if (!s) return '—';
  return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
}

function fmtDate(iso: string): string {
  return new Date(iso).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

function MiniBar({ label, count, total, color }: { label: string; count: number; total: number; color: string }) {
  const pct = total > 0 ? Math.round((count / total) * 100) : 0;
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="text-gray-300">{label}</span>
        <span className="text-white font-semibold">
          {count.toLocaleString()} <span className="text-gray-500">({pct}%)</span>
        </span>
      </div>
      <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
        <div className={`h-full ${color} rounded-full transition-all`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function BigStat({ label, value, sub, accent = false }: { label: string; value: string; sub?: string; accent?: boolean }) {
  return (
    <div className="bg-secondary rounded-xl p-5 border border-gray-700/40">
      <p className="text-xs text-gray-400 mb-1">{label}</p>
      <p className={`text-3xl font-bold ${accent ? 'text-accent' : 'text-white'}`}>{value}</p>
      {sub && <p className="text-xs text-gray-500 mt-0.5">{sub}</p>}
    </div>
  );
}

export default function OverallStatsPage() {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/dashboard/summary')
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => setSummary(d))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const ts = summary?.tennisStats;
  const has = !loading && summary?.hasTennisStats;
  const totals = summary?.totals ?? { total: 0, done: 0, processing: 0, failed: 0 };

  const totalStrokes = ts ? ts.totalForehands + ts.totalBackhands + ts.totalServes : 0;
  const totalBounced = ts ? ts.totalInBounds + ts.totalOutBounds : 0;
  const inPct = totalBounced > 0 ? Math.round((ts!.totalInBounds / totalBounced) * 100) : null;

  const sessions = summary?.games ?? [];

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="w-8 h-8 border-2 border-accent border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="px-5 py-6 max-w-5xl">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-white">Overall Stats</h2>
        <p className="text-sm text-gray-400 mt-1">Your career performance across all matches</p>
      </div>

      {/* Legacy notice */}
      {totals.done > 0 && !has && (
        <div className="mb-5 flex items-start gap-3 bg-yellow-500/8 border border-yellow-500/20 rounded-xl px-4 py-3 text-sm text-yellow-400/80">
          <svg viewBox="0 0 24 24" className="w-4 h-4 shrink-0 mt-0.5" fill="none" stroke="currentColor" strokeWidth={2}>
            <circle cx="12" cy="12" r="10" />
            <path d="M12 8v4M12 16h.01" />
          </svg>
          <span>
            Your existing {totals.done} recording{totals.done !== 1 ? 's were' : ' was'} processed before shot/stroke tracking was added.
            Shot counts, stroke types, and court accuracy will appear here once you{' '}
            <Link href="/upload" className="underline underline-offset-2 text-yellow-400 hover:text-yellow-300">
              re-upload a match
            </Link>
            .
          </span>
        </div>
      )}

      {/* Primary tennis stats */}
      {has ? (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-6">
          <BigStat label="Total Shots"   value={ts!.totalShots.toLocaleString()}   sub="across all sessions" accent />
          <BigStat label="Ball Bounces"  value={ts!.totalBounces.toLocaleString()} sub={inPct !== null ? `${inPct}% in bounds` : undefined} />
          <BigStat label="Total Rallies" value={ts!.totalRallies.toLocaleString()} sub="rally exchanges" />
          <BigStat label="Court Accuracy" value={inPct !== null ? `${inPct}%` : '—'} sub={`${ts!.totalInBounds} in / ${ts!.totalOutBounds} out`} accent={inPct !== null && inPct >= 70} />
        </div>
      ) : (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-6">
          {['Total Shots', 'Ball Bounces', 'Total Rallies', 'Court Accuracy'].map((l) => (
            <div key={l} className="bg-secondary rounded-xl p-5 border border-gray-700/40">
              <p className="text-xs text-gray-400 mb-1">{l}</p>
              <p className="text-3xl font-bold text-gray-600">—</p>
              <p className="text-xs text-gray-600 mt-0.5">no data</p>
            </div>
          ))}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 mb-5">
        {/* Stroke Breakdown */}
        <div className="bg-secondary rounded-2xl p-5 border border-gray-700/40">
          <h3 className="text-sm font-semibold text-white mb-4">Stroke Breakdown</h3>
          {has && totalStrokes > 0 ? (
            <div className="flex flex-col gap-3">
              {/* Stacked bar */}
              <div className="flex h-3 rounded-full overflow-hidden gap-px mb-1">
                {ts!.totalForehands > 0 && (
                  <div className="bg-accent" style={{ width: `${Math.round((ts!.totalForehands / totalStrokes) * 100)}%` }} />
                )}
                {ts!.totalBackhands > 0 && (
                  <div className="bg-blue-500" style={{ width: `${Math.round((ts!.totalBackhands / totalStrokes) * 100)}%` }} />
                )}
                {ts!.totalServes > 0 && (
                  <div className="bg-purple-500" style={{ width: `${Math.round((ts!.totalServes / totalStrokes) * 100)}%` }} />
                )}
              </div>
              <MiniBar label="Forehand"      count={ts!.totalForehands} total={totalStrokes} color="bg-accent"     />
              <MiniBar label="Backhand"      count={ts!.totalBackhands} total={totalStrokes} color="bg-blue-500"   />
              <MiniBar label="Serve / Smash" count={ts!.totalServes}    total={totalStrokes} color="bg-purple-500" />
              <p className="text-xs text-gray-500 pt-2 border-t border-gray-700/40">
                {totalStrokes.toLocaleString()} strokes classified in total
              </p>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-8 text-center gap-2">
              <svg viewBox="0 0 24 24" className="w-8 h-8 text-gray-600" fill="none" stroke="currentColor" strokeWidth={1.5}>
                <path d="M9 17H7A5 5 0 0 1 7 7h2M15 7h2a5 5 0 1 1 0 10h-2M8 12h8" />
              </svg>
              <p className="text-xs text-gray-500">No stroke data yet</p>
            </div>
          )}
        </div>

        {/* Shot Quality */}
        <div className="bg-secondary rounded-2xl p-5 border border-gray-700/40">
          <h3 className="text-sm font-semibold text-white mb-4">Shot Quality</h3>
          {has && totalBounced > 0 ? (
            <div className="flex flex-col gap-4">
              <div className="flex items-center gap-4">
                <div className="shrink-0 w-20 h-20 relative flex items-center justify-center">
                  <svg viewBox="0 0 36 36" className="w-20 h-20 -rotate-90">
                    <circle cx="18" cy="18" r="15.9" fill="none" stroke="#374151" strokeWidth="3" />
                    <circle
                      cx="18" cy="18" r="15.9" fill="none"
                      stroke="var(--color-accent, #22d3ee)" strokeWidth="3"
                      strokeDasharray={`${inPct} ${100 - inPct!}`}
                      strokeLinecap="round"
                    />
                  </svg>
                  <span className="absolute text-sm font-bold text-white">{inPct}%</span>
                </div>
                <div>
                  <p className="text-xs text-gray-400">In bounds accuracy</p>
                  <p className="text-2xl font-bold text-white">{ts!.totalInBounds.toLocaleString()}</p>
                  <p className="text-xs text-gray-500">of {totalBounced.toLocaleString()} bounces</p>
                </div>
              </div>
              <MiniBar label="In bounds"     count={ts!.totalInBounds}  total={totalBounced} color="bg-accent"  />
              <MiniBar label="Out of bounds" count={ts!.totalOutBounds} total={totalBounced} color="bg-red-500" />
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-8 text-center gap-2">
              <svg viewBox="0 0 24 24" className="w-8 h-8 text-gray-600" fill="none" stroke="currentColor" strokeWidth={1.5}>
                <rect x="3" y="3" width="18" height="18" rx="1" />
                <path d="M3 12h18M12 3v18" />
              </svg>
              <p className="text-xs text-gray-500">No bounce quality data yet</p>
            </div>
          )}
        </div>
      </div>

      {/* Sessions table */}
      <div className="bg-secondary rounded-2xl border border-gray-700/40 overflow-hidden mb-5">
        <div className="px-5 py-4 border-b border-gray-700/40 flex items-center justify-between">
          <h3 className="text-sm font-semibold text-white">Match Sessions</h3>
          <span className="text-xs text-gray-500">{totals.done} completed</span>
        </div>

        {sessions.length === 0 ? (
          <div className="px-5 py-8 text-center">
            <p className="text-xs text-gray-500">No completed sessions yet</p>
          </div>
        ) : (
          <div>
            {/* Header row */}
            <div className="grid grid-cols-7 gap-2 px-5 py-2 text-[10px] text-gray-500 border-b border-gray-700/30">
              <span className="col-span-2">Date</span>
              <span className="text-right">Duration</span>
              <span className="text-right">Shots</span>
              <span className="text-right">Bounces</span>
              <span className="text-right">Rallies</span>
              <span className="text-right">Accuracy</span>
            </div>
            {sessions.map((g) => {
              const bounced = (g.inBounces ?? 0) + (g.outBounces ?? 0);
              const acc = bounced > 0 ? `${Math.round(((g.inBounces ?? 0) / bounced) * 100)}%` : null;
              const hasSessionStats = g.shotCount !== null || g.bounceCount !== null;
              return (
                <Link
                  key={g.id}
                  href={`/recordings/${g.id}`}
                  className="grid grid-cols-7 gap-2 px-5 py-3 text-xs hover:bg-white/5 transition-colors border-b border-gray-700/20 last:border-0"
                >
                  <div className="col-span-2 flex items-center gap-2">
                    <span className="text-gray-300">{fmtDate(g.createdAt)}</span>
                    {(g.hasBallHeatmap || g.hasPlayerHeatmap) && (
                      <span className="text-[9px] bg-accent/10 text-accent px-1 py-0.5 rounded">report</span>
                    )}
                  </div>
                  <span className="text-right text-gray-400">{fmtDur(g.durationSeconds)}</span>
                  <span className={`text-right font-medium ${hasSessionStats ? 'text-white' : 'text-gray-600'}`}>
                    {g.shotCount ?? '—'}
                  </span>
                  <span className={`text-right font-medium ${hasSessionStats ? 'text-white' : 'text-gray-600'}`}>
                    {g.bounceCount ?? '—'}
                  </span>
                  <span className={`text-right font-medium ${hasSessionStats ? 'text-white' : 'text-gray-600'}`}>
                    {g.rallyCount ?? '—'}
                  </span>
                  <span className={`text-right font-medium ${acc ? 'text-accent' : 'text-gray-600'}`}>
                    {acc ?? '—'}
                  </span>
                </Link>
              );
            })}
          </div>
        )}
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-2 gap-3">
        <Link
          href="/upload"
          className="flex items-center gap-3 p-4 bg-secondary rounded-2xl border border-gray-700/40 hover:border-accent/30 transition-colors group"
        >
          <div className="w-9 h-9 rounded-xl bg-accent/10 border border-accent/20 flex items-center justify-center shrink-0">
            <svg className="h-4 w-4 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
          </div>
          <div>
            <p className="text-sm text-white font-medium">Upload New Match</p>
            <p className="text-xs text-gray-500">Analyse your next session</p>
          </div>
        </Link>

        <Link
          href="/recordings"
          className="flex items-center gap-3 p-4 bg-secondary rounded-2xl border border-gray-700/40 hover:border-gray-600/60 transition-colors group"
        >
          <div className="w-9 h-9 rounded-xl bg-gray-700/40 border border-gray-700/40 flex items-center justify-center shrink-0">
            <svg className="h-4 w-4 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
          </div>
          <div>
            <p className="text-sm text-white font-medium">View All Recordings</p>
            <p className="text-xs text-gray-500">Browse your match library</p>
          </div>
        </Link>
      </div>
    </div>
  );
}
