"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

interface TennisStats {
  totalForehands: number;
  totalBackhands: number;
  totalServes: number;
}

interface Summary {
  tennisStats: TennisStats;
  hasTennisStats: boolean;
}

export default function PlayerStats() {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/dashboard/summary")
      .then((r) => r.json())
      .then((d) => setSummary(d))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const fh    = summary?.tennisStats?.totalForehands ?? 0;
  const bh    = summary?.tennisStats?.totalBackhands ?? 0;
  const serve = summary?.tennisStats?.totalServes    ?? 0;
  const total = fh + bh + serve;

  const pct = (n: number) => (total > 0 ? Math.round((n / total) * 100) : 0);

  const strokes = [
    { label: "Forehand",      count: fh,    pct: pct(fh),    color: '#B4F000' },
    { label: "Backhand",      count: bh,    pct: pct(bh),    color: '#60A5FA' },
    { label: "Serve / Smash", count: serve, pct: pct(serve), color: '#A78BFA' },
  ];

  const hasData = !loading && summary?.hasTennisStats && total > 0;

  return (
    <div
      className="rounded-2xl p-5 h-full flex flex-col"
      style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }}
    >
      <h3 className="text-sm font-semibold text-white mb-4">Stroke Breakdown</h3>

      {loading && (
        <div className="flex-1 flex items-center justify-center">
          <div
            className="w-5 h-5 rounded-full animate-spin"
            style={{ border: '2px solid rgba(180,240,0,0.15)', borderTopColor: '#B4F000' }}
          />
        </div>
      )}

      {!loading && !hasData && (
        <div className="flex-1 flex flex-col items-center justify-center text-center gap-2 py-4">
          <svg viewBox="0 0 24 24" className="w-7 h-7" fill="none" stroke="currentColor" strokeWidth={1.5} style={{ color: '#2A2A33' }}>
            <path d="M9 17H7A5 5 0 0 1 7 7h2M15 7h2a5 5 0 1 1 0 10h-2M8 12h8" />
          </svg>
          <p className="text-xs" style={{ color: '#3A3A44' }}>No stroke data yet</p>
          <Link href="/upload" className="text-xs transition-opacity hover:opacity-70" style={{ color: '#B4F000' }}>
            Analyze a match →
          </Link>
        </div>
      )}

      {hasData && (
        <div className="flex-1 flex flex-col gap-4">
          {/* Stacked bar */}
          <div className="flex h-2 rounded-full overflow-hidden gap-px">
            {strokes.map((s) =>
              s.pct > 0 ? (
                <div
                  key={s.label}
                  className="transition-all"
                  style={{ width: `${s.pct}%`, background: s.color }}
                  title={`${s.label}: ${s.pct}%`}
                />
              ) : null
            )}
          </div>

          {/* Legend rows */}
          <div className="flex flex-col gap-2.5">
            {strokes.map((s) => (
              <div key={s.label} className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-sm shrink-0" style={{ background: s.color }} />
                <span className="text-xs flex-1" style={{ color: '#9CA3AF' }}>{s.label}</span>
                <span className="text-xs font-semibold text-white">{s.count.toLocaleString()}</span>
                <span className="text-xs w-8 text-right" style={{ color: '#3A3A44' }}>{s.pct}%</span>
              </div>
            ))}
          </div>

          <div className="mt-auto pt-2 flex items-center justify-between" style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }}>
            <span className="text-xs" style={{ color: '#3A3A44' }}>Total strokes classified</span>
            <span className="text-xs font-bold text-white">{total.toLocaleString()}</span>
          </div>
        </div>
      )}
    </div>
  );
}
