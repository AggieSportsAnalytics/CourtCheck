"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

interface TennisStats {
  totalInBounds: number;
  totalOutBounds: number;
}

interface Summary {
  tennisStats: TennisStats;
  hasTennisStats: boolean;
}

export default function ShotPercentages() {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/dashboard/summary")
      .then((r) => r.json())
      .then((d) => setSummary(d))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const inBounds  = summary?.tennisStats?.totalInBounds  ?? 0;
  const outBounds = summary?.tennisStats?.totalOutBounds ?? 0;
  const total     = inBounds + outBounds;
  const inPct     = total > 0 ? Math.round((inBounds / total) * 100)  : 0;
  const outPct    = total > 0 ? Math.round((outBounds / total) * 100) : 0;
  const hasData   = !loading && summary?.hasTennisStats && total > 0;

  return (
    <div
      className="rounded-2xl p-5 h-full flex flex-col"
      style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }}
    >
      <h3 className="text-sm font-semibold text-white mb-1">Shot Quality</h3>
      <p className="text-xs mb-4" style={{ color: '#3A3A44' }}>In bounds vs out of bounds</p>

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
            <rect x="3" y="3" width="18" height="18" rx="1" />
            <path d="M3 12h18M12 3v18" />
          </svg>
          <p className="text-xs" style={{ color: '#3A3A44' }}>No bounce data yet</p>
          <Link href="/upload" className="text-xs transition-opacity hover:opacity-70" style={{ color: '#B4F000' }}>
            Analyze a match →
          </Link>
        </div>
      )}

      {hasData && (
        <div className="flex-1 flex flex-col gap-4">
          <div className="flex items-center gap-4">
            {/* Ring */}
            <div className="shrink-0 w-20 h-20 relative flex items-center justify-center">
              <svg viewBox="0 0 36 36" className="w-20 h-20 -rotate-90">
                <circle cx="18" cy="18" r="15.9" fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="3" />
                <circle
                  cx="18" cy="18" r="15.9" fill="none"
                  stroke="#B4F000" strokeWidth="3"
                  strokeDasharray={`${inPct} ${100 - inPct}`}
                  strokeLinecap="round"
                />
              </svg>
              <span className="absolute text-sm font-bold text-white">{inPct}%</span>
            </div>
            <div className="flex flex-col gap-1 min-w-0">
              <p className="text-xs" style={{ color: '#5A5A66' }}>In bounds accuracy</p>
              <p className="text-lg font-bold text-white">
                {inBounds.toLocaleString()}{' '}
                <span className="text-xs font-normal" style={{ color: '#5A5A66' }}>shots</span>
              </p>
            </div>
          </div>

          <div className="flex flex-col gap-2">
            {[
              { label: 'In bounds',     pct: inPct,  color: '#B4F000' },
              { label: 'Out of bounds', pct: outPct, color: '#EF4444' },
            ].map((item) => (
              <div key={item.label}>
                <div className="flex justify-between text-xs mb-1">
                  <span className="flex items-center gap-1.5" style={{ color: '#9CA3AF' }}>
                    <span className="w-1.5 h-1.5 rounded-full inline-block" style={{ background: item.color }} />
                    {item.label}
                  </span>
                  <span className="text-white font-semibold">{item.pct}%</span>
                </div>
                <div className="h-1.5 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.06)' }}>
                  <div className="h-full rounded-full transition-all" style={{ width: `${item.pct}%`, background: item.color }} />
                </div>
              </div>
            ))}
          </div>

          <div className="mt-auto pt-2 flex items-center justify-between" style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }}>
            <span className="text-xs" style={{ color: '#3A3A44' }}>Total bounces tracked</span>
            <span className="text-xs font-bold text-white">{total.toLocaleString()}</span>
          </div>
        </div>
      )}
    </div>
  );
}
