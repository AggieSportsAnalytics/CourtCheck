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
    <div className="bg-secondary rounded-2xl p-5 border border-gray-700/40 h-full flex flex-col">
      <h3 className="text-sm font-semibold text-white mb-1">Shot Quality</h3>
      <p className="text-xs text-gray-500 mb-4">Ball landing — in bounds vs out of bounds</p>

      {loading && (
        <div className="flex-1 flex items-center justify-center">
          <div className="w-5 h-5 border-2 border-accent border-t-transparent rounded-full animate-spin" />
        </div>
      )}

      {!loading && !hasData && (
        <div className="flex-1 flex flex-col items-center justify-center text-center gap-2 py-4">
          <svg viewBox="0 0 24 24" className="w-8 h-8 text-gray-600" fill="none" stroke="currentColor" strokeWidth={1.5}>
            <rect x="3" y="3" width="18" height="18" rx="1" />
            <path d="M3 12h18M12 3v18" />
          </svg>
          <p className="text-xs text-gray-500">No bounce data yet</p>
          <Link href="/upload" className="text-xs text-accent hover:underline">
            Analyse a match →
          </Link>
        </div>
      )}

      {hasData && (
        <div className="flex-1 flex flex-col gap-4">
          {/* Visual ring / progress circle approximated via two bars */}
          <div className="flex items-center gap-4">
            {/* Big percentage */}
            <div className="shrink-0 w-20 h-20 relative flex items-center justify-center">
              <svg viewBox="0 0 36 36" className="w-20 h-20 -rotate-90">
                <circle cx="18" cy="18" r="15.9" fill="none" stroke="#374151" strokeWidth="3" />
                <circle
                  cx="18" cy="18" r="15.9" fill="none"
                  stroke="var(--color-accent, #22d3ee)" strokeWidth="3"
                  strokeDasharray={`${inPct} ${100 - inPct}`}
                  strokeLinecap="round"
                />
              </svg>
              <span className="absolute text-sm font-bold text-white">{inPct}%</span>
            </div>
            <div className="flex flex-col gap-1.5 min-w-0">
              <p className="text-xs text-gray-400">In bounds accuracy</p>
              <p className="text-lg font-bold text-white">{inBounds.toLocaleString()} <span className="text-xs font-normal text-gray-400">shots</span></p>
            </div>
          </div>

          {/* Bars */}
          <div className="flex flex-col gap-2">
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-gray-300 flex items-center gap-1">
                  <span className="w-2 h-2 rounded-full bg-accent inline-block" />
                  In bounds
                </span>
                <span className="text-white font-semibold">{inPct}%</span>
              </div>
              <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                <div className="h-full bg-accent rounded-full transition-all" style={{ width: `${inPct}%` }} />
              </div>
            </div>
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-gray-300 flex items-center gap-1">
                  <span className="w-2 h-2 rounded-full bg-red-500 inline-block" />
                  Out of bounds
                </span>
                <span className="text-white font-semibold">{outPct}%</span>
              </div>
              <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                <div className="h-full bg-red-500 rounded-full transition-all" style={{ width: `${outPct}%` }} />
              </div>
            </div>
          </div>

          <div className="mt-auto pt-2 border-t border-gray-700/40 flex items-center justify-between">
            <span className="text-xs text-gray-500">Total bounces tracked</span>
            <span className="text-xs font-bold text-white">{total.toLocaleString()}</span>
          </div>
        </div>
      )}
    </div>
  );
}
