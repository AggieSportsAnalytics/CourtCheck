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
    { label: "Forehand",      count: fh,    pct: pct(fh),    color: "bg-accent" },
    { label: "Backhand",      count: bh,    pct: pct(bh),    color: "bg-blue-500" },
    { label: "Serve / Smash", count: serve, pct: pct(serve), color: "bg-purple-500" },
  ];

  const hasData = !loading && summary?.hasTennisStats && total > 0;

  return (
    <div className="bg-secondary rounded-2xl p-5 border border-gray-700/40 h-full flex flex-col">
      <h3 className="text-sm font-semibold text-white mb-4">Stroke Breakdown</h3>

      {loading && (
        <div className="flex-1 flex items-center justify-center">
          <div className="w-5 h-5 border-2 border-accent border-t-transparent rounded-full animate-spin" />
        </div>
      )}

      {!loading && !hasData && (
        <div className="flex-1 flex flex-col items-center justify-center text-center gap-2 py-4">
          <svg viewBox="0 0 24 24" className="w-8 h-8 text-gray-600" fill="none" stroke="currentColor" strokeWidth={1.5}>
            <path d="M9 17H7A5 5 0 0 1 7 7h2" />
            <path d="M15 7h2a5 5 0 1 1 0 10h-2" />
            <line x1="8" y1="12" x2="16" y2="12" />
          </svg>
          <p className="text-xs text-gray-500">No stroke data yet</p>
          <Link href="/upload" className="text-xs text-accent hover:underline">
            Analyse a match →
          </Link>
        </div>
      )}

      {hasData && (
        <div className="flex-1 flex flex-col gap-4">
          {/* Stacked bar */}
          <div className="flex h-3 rounded-full overflow-hidden gap-px">
            {strokes.map((s) =>
              s.pct > 0 ? (
                <div
                  key={s.label}
                  className={`${s.color} transition-all`}
                  style={{ width: `${s.pct}%` }}
                  title={`${s.label}: ${s.pct}%`}
                />
              ) : null
            )}
          </div>

          {/* Legend rows */}
          <div className="flex flex-col gap-2.5">
            {strokes.map((s) => (
              <div key={s.label} className="flex items-center gap-2">
                <span className={`w-2.5 h-2.5 rounded-sm shrink-0 ${s.color}`} />
                <span className="text-xs text-gray-300 flex-1">{s.label}</span>
                <span className="text-xs font-semibold text-white">{s.count.toLocaleString()}</span>
                <span className="text-xs text-gray-500 w-8 text-right">{s.pct}%</span>
              </div>
            ))}
          </div>

          <div className="mt-auto pt-2 border-t border-gray-700/40 flex items-center justify-between">
            <span className="text-xs text-gray-500">Total strokes classified</span>
            <span className="text-xs font-bold text-white">{total.toLocaleString()}</span>
          </div>
        </div>
      )}
    </div>
  );
}
