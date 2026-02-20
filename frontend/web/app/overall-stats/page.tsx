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
        <span style={{ color: '#9CA3AF' }}>{label}</span>
        <span className="text-white font-semibold">
          {count.toLocaleString()} <span style={{ color: '#3A3A44' }}>({pct}%)</span>
        </span>
      </div>
      <div className="h-1.5 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.06)' }}>
        <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}

function BigStat({ label, value, sub, accent = false }: { label: string; value: string; sub?: string; accent?: boolean }) {
  return (
    <div
      className="rounded-xl p-5"
      style={
        accent
          ? { background: 'rgba(180,240,0,0.05)', border: '1px solid rgba(180,240,0,0.15)' }
          : { background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }
      }
    >
      <p className="text-xs mb-1" style={{ color: '#5A5A66' }}>{label}</p>
      <p className="text-3xl font-black tracking-tight" style={{ color: accent ? '#B4F000' : '#FAFAFA' }}>{value}</p>
      {sub && <p className="text-xs mt-0.5" style={{ color: '#3A3A44' }}>{sub}</p>}
    </div>
  );
}

const CARD_STYLE = { background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' };
const SECTION_BORDER = { borderBottom: '1px solid rgba(255,255,255,0.06)' };

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
        <div
          className="w-8 h-8 rounded-full animate-spin"
          style={{ border: '2px solid rgba(180,240,0,0.15)', borderTopColor: '#B4F000' }}
        />
      </div>
    );
  }

  return (
    <div className="px-6 py-8 max-w-5xl">
      {/* Page header */}
      <div className="mb-8">
        <p className="text-xs font-semibold uppercase tracking-widest mb-2" style={{ color: '#B4F000' }}>
          Career
        </p>
        <h1 className="text-3xl font-black tracking-tight text-white">Overall Stats</h1>
        <p className="text-sm mt-1" style={{ color: '#5A5A66' }}>
          Your career performance across all matches
        </p>
      </div>

      {/* Legacy notice */}
      {totals.done > 0 && !has && (
        <div
          className="mb-6 flex items-start gap-3 px-4 py-3 rounded-xl text-sm"
          style={{ background: 'rgba(234,179,8,0.06)', border: '1px solid rgba(234,179,8,0.15)', color: 'rgba(234,179,8,0.7)' }}
        >
          <svg viewBox="0 0 24 24" className="w-4 h-4 shrink-0 mt-0.5" fill="none" stroke="currentColor" strokeWidth={2}>
            <circle cx="12" cy="12" r="10" /><path d="M12 8v4M12 16h.01" />
          </svg>
          <span>
            Your {totals.done} recording{totals.done !== 1 ? 's were' : ' was'} processed before shot tracking was added.
            Stats will appear once you{' '}
            <Link href="/upload" className="underline underline-offset-2 hover:opacity-80" style={{ color: 'rgba(234,179,8,0.9)' }}>
              re-upload a match
            </Link>.
          </span>
        </div>
      )}

      {/* Primary stats grid */}
      {has ? (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-6">
          <BigStat label="Total Shots"    value={ts!.totalShots.toLocaleString()}   sub="across all sessions" accent />
          <BigStat label="Ball Bounces"   value={ts!.totalBounces.toLocaleString()} sub={inPct !== null ? `${inPct}% in bounds` : undefined} />
          <BigStat label="Total Rallies"  value={ts!.totalRallies.toLocaleString()} sub="rally exchanges" />
          <BigStat label="Court Accuracy" value={inPct !== null ? `${inPct}%` : '—'} sub={`${ts!.totalInBounds} in / ${ts!.totalOutBounds} out`} accent={inPct !== null && inPct >= 70} />
        </div>
      ) : (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-6">
          {['Total Shots', 'Ball Bounces', 'Total Rallies', 'Court Accuracy'].map((l) => (
            <div key={l} className="rounded-xl p-5" style={CARD_STYLE}>
              <p className="text-xs mb-1" style={{ color: '#5A5A66' }}>{l}</p>
              <p className="text-3xl font-black" style={{ color: '#2A2A33' }}>—</p>
              <p className="text-xs mt-0.5" style={{ color: '#2A2A33' }}>no data</p>
            </div>
          ))}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-5">
        {/* Stroke Breakdown */}
        <div className="rounded-2xl p-5" style={CARD_STYLE}>
          <h3 className="text-xs font-semibold uppercase tracking-widest mb-4" style={{ color: '#5A5A66' }}>Stroke Breakdown</h3>
          {has && totalStrokes > 0 ? (
            <div className="flex flex-col gap-3">
              <div className="flex h-2 rounded-full overflow-hidden gap-px mb-1">
                {ts!.totalForehands > 0 && <div style={{ width: `${Math.round((ts!.totalForehands / totalStrokes) * 100)}%`, background: '#B4F000' }} />}
                {ts!.totalBackhands > 0 && <div style={{ width: `${Math.round((ts!.totalBackhands / totalStrokes) * 100)}%`, background: '#60A5FA' }} />}
                {ts!.totalServes > 0    && <div style={{ width: `${Math.round((ts!.totalServes    / totalStrokes) * 100)}%`, background: '#A78BFA' }} />}
              </div>
              <MiniBar label="Forehand"      count={ts!.totalForehands} total={totalStrokes} color="#B4F000" />
              <MiniBar label="Backhand"      count={ts!.totalBackhands} total={totalStrokes} color="#60A5FA" />
              <MiniBar label="Serve / Smash" count={ts!.totalServes}    total={totalStrokes} color="#A78BFA" />
              <p className="text-xs pt-2" style={{ color: '#3A3A44', borderTop: '1px solid rgba(255,255,255,0.06)' }}>
                {totalStrokes.toLocaleString()} strokes classified in total
              </p>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-8 text-center gap-2">
              <svg viewBox="0 0 24 24" className="w-7 h-7" fill="none" stroke="currentColor" strokeWidth={1.5} style={{ color: '#2A2A33' }}>
                <path d="M9 17H7A5 5 0 0 1 7 7h2M15 7h2a5 5 0 1 1 0 10h-2M8 12h8" />
              </svg>
              <p className="text-xs" style={{ color: '#3A3A44' }}>No stroke data yet</p>
            </div>
          )}
        </div>

        {/* Shot Quality */}
        <div className="rounded-2xl p-5" style={CARD_STYLE}>
          <h3 className="text-xs font-semibold uppercase tracking-widest mb-4" style={{ color: '#5A5A66' }}>Shot Quality</h3>
          {has && totalBounced > 0 ? (
            <div className="flex flex-col gap-4">
              <div className="flex items-center gap-4">
                <div className="shrink-0 w-20 h-20 relative flex items-center justify-center">
                  <svg viewBox="0 0 36 36" className="w-20 h-20 -rotate-90">
                    <circle cx="18" cy="18" r="15.9" fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="3" />
                    <circle cx="18" cy="18" r="15.9" fill="none" stroke="#B4F000" strokeWidth="3"
                      strokeDasharray={`${inPct} ${100 - inPct!}`} strokeLinecap="round" />
                  </svg>
                  <span className="absolute text-sm font-bold text-white">{inPct}%</span>
                </div>
                <div>
                  <p className="text-xs" style={{ color: '#5A5A66' }}>In bounds accuracy</p>
                  <p className="text-2xl font-black text-white">{ts!.totalInBounds.toLocaleString()}</p>
                  <p className="text-xs" style={{ color: '#3A3A44' }}>of {totalBounced.toLocaleString()} bounces</p>
                </div>
              </div>
              <MiniBar label="In bounds"     count={ts!.totalInBounds}  total={totalBounced} color="#B4F000" />
              <MiniBar label="Out of bounds" count={ts!.totalOutBounds} total={totalBounced} color="#EF4444" />
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-8 text-center gap-2">
              <svg viewBox="0 0 24 24" className="w-7 h-7" fill="none" stroke="currentColor" strokeWidth={1.5} style={{ color: '#2A2A33' }}>
                <rect x="3" y="3" width="18" height="18" rx="1" /><path d="M3 12h18M12 3v18" />
              </svg>
              <p className="text-xs" style={{ color: '#3A3A44' }}>No bounce quality data yet</p>
            </div>
          )}
        </div>
      </div>

      {/* Sessions table */}
      <div className="rounded-2xl overflow-hidden mb-5" style={CARD_STYLE}>
        <div className="px-5 py-4 flex items-center justify-between" style={SECTION_BORDER}>
          <h3 className="text-xs font-semibold uppercase tracking-widest" style={{ color: '#5A5A66' }}>Match Sessions</h3>
          <span className="text-xs" style={{ color: '#3A3A44' }}>{totals.done} completed</span>
        </div>

        {sessions.length === 0 ? (
          <div className="px-5 py-8 text-center">
            <p className="text-xs" style={{ color: '#3A3A44' }}>No completed sessions yet</p>
          </div>
        ) : (
          <div>
            <div className="grid grid-cols-7 gap-2 px-5 py-2.5 text-[10px]" style={{ ...SECTION_BORDER, color: '#3A3A44' }}>
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
                  className="grid grid-cols-7 gap-2 px-5 py-3 text-xs transition-colors last:border-0"
                  style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}
                  onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.03)'; }}
                  onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = 'transparent'; }}
                >
                  <div className="col-span-2 flex items-center gap-2">
                    <span style={{ color: '#9CA3AF' }}>{fmtDate(g.createdAt)}</span>
                    {(g.hasBallHeatmap || g.hasPlayerHeatmap) && (
                      <span
                        className="text-[9px] px-1 py-0.5 rounded"
                        style={{ background: 'rgba(180,240,0,0.08)', color: '#B4F000' }}
                      >
                        report
                      </span>
                    )}
                  </div>
                  <span className="text-right" style={{ color: '#5A5A66' }}>{fmtDur(g.durationSeconds)}</span>
                  <span className="text-right font-medium" style={{ color: hasSessionStats ? '#FAFAFA' : '#2A2A33' }}>{g.shotCount ?? '—'}</span>
                  <span className="text-right font-medium" style={{ color: hasSessionStats ? '#FAFAFA' : '#2A2A33' }}>{g.bounceCount ?? '—'}</span>
                  <span className="text-right font-medium" style={{ color: hasSessionStats ? '#FAFAFA' : '#2A2A33' }}>{g.rallyCount ?? '—'}</span>
                  <span className="text-right font-medium" style={{ color: acc ? '#B4F000' : '#2A2A33' }}>{acc ?? '—'}</span>
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
          className="flex items-center gap-3 p-4 rounded-2xl transition-all"
          style={{ ...CARD_STYLE }}
          onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.border = '1px solid rgba(180,240,0,0.2)'; }}
          onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.border = '1px solid rgba(255,255,255,0.07)'; }}
        >
          <div
            className="w-9 h-9 rounded-xl flex items-center justify-center shrink-0"
            style={{ background: 'rgba(180,240,0,0.08)', border: '1px solid rgba(180,240,0,0.15)' }}
          >
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} style={{ color: '#B4F000' }}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
          </div>
          <div>
            <p className="text-sm text-white font-medium">Upload New Match</p>
            <p className="text-xs mt-0.5" style={{ color: '#5A5A66' }}>Analyze your next session</p>
          </div>
        </Link>

        <Link
          href="/recordings"
          className="flex items-center gap-3 p-4 rounded-2xl transition-all"
          style={CARD_STYLE}
          onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.border = '1px solid rgba(255,255,255,0.14)'; }}
          onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.border = '1px solid rgba(255,255,255,0.07)'; }}
        >
          <div
            className="w-9 h-9 rounded-xl flex items-center justify-center shrink-0"
            style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)' }}
          >
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} style={{ color: '#6B7280' }}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
          </div>
          <div>
            <p className="text-sm text-white font-medium">View All Recordings</p>
            <p className="text-xs mt-0.5" style={{ color: '#5A5A66' }}>Browse your match library</p>
          </div>
        </Link>
      </div>
    </div>
  );
}
