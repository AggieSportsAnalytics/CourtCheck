"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

interface GameEntry {
  id: string;
  createdAt: string;
  shotCount: number | null;
  bounceCount: number | null;
  rallyCount: number | null;
  inBounces: number | null;
  outBounces: number | null;
}

export default function GameStatistics() {
  const [games, setGames] = useState<GameEntry[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/dashboard/summary")
      .then((r) => r.json())
      .then((d) => setGames(d.games ?? []))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const withStats = games.filter((g) => g.shotCount !== null || g.bounceCount !== null);
  const maxShots  = Math.max(...withStats.map((g) => g.shotCount ?? 0), 1);

  return (
    <div
      className="rounded-2xl p-5 h-full flex flex-col"
      style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }}
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-white">Shot History</h3>
        <span className="text-xs" style={{ color: '#3A3A44' }}>last {games.length} sessions</span>
      </div>

      {loading && (
        <div className="flex-1 flex items-center justify-center">
          <div
            className="w-5 h-5 rounded-full animate-spin"
            style={{ border: '2px solid rgba(180,240,0,0.15)', borderTopColor: '#B4F000' }}
          />
        </div>
      )}

      {!loading && withStats.length === 0 && (
        <div className="flex-1 flex flex-col items-center justify-center text-center gap-2 py-4">
          <svg viewBox="0 0 24 24" className="w-7 h-7" fill="none" stroke="currentColor" strokeWidth={1.5} style={{ color: '#2A2A33' }}>
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
          </svg>
          <p className="text-xs" style={{ color: '#3A3A44' }}>No session data yet</p>
          <Link href="/upload" className="text-xs transition-opacity hover:opacity-70" style={{ color: '#B4F000' }}>
            Analyze a match →
          </Link>
        </div>
      )}

      {!loading && withStats.length > 0 && (
        <div className="flex-1 flex flex-col gap-3">
          {/* Bar chart */}
          <div className="flex items-end gap-1.5 h-28">
            {withStats.slice(0, 8).reverse().map((g, i) => {
              const shots   = g.shotCount   ?? 0;
              const bounces = g.bounceCount ?? 0;
              const barH    = Math.round((shots / maxShots) * 100);
              const total   = (g.inBounces ?? 0) + (g.outBounces ?? 0);
              const inPct   = total > 0 ? Math.round(((g.inBounces ?? 0) / total) * 100) : null;

              return (
                <Link
                  key={g.id}
                  href={`/recordings/${g.id}`}
                  title={`${new Date(g.createdAt).toLocaleDateString()}\nShots: ${shots} | Bounces: ${bounces}${inPct !== null ? ` | ${inPct}% in` : ""}`}
                  className="flex-1 flex flex-col items-center gap-1 group"
                >
                  <div className="w-full flex flex-col justify-end" style={{ height: "96px" }}>
                    <div
                      className="w-full rounded-t transition-all"
                      style={{
                        height: `${Math.max(barH, 4)}%`,
                        background: 'rgba(180,240,0,0.5)',
                      }}
                      onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = '#B4F000'; }}
                      onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = 'rgba(180,240,0,0.5)'; }}
                    />
                  </div>
                  <span className="text-[9px]" style={{ color: '#3A3A44' }}>
                    {new Date(g.createdAt).toLocaleDateString("en-US", { month: "numeric", day: "numeric" })}
                  </span>
                </Link>
              );
            })}
          </div>

          {/* Table */}
          <div className="pt-3" style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }}>
            <div className="grid grid-cols-4 text-[10px] mb-1 px-1" style={{ color: '#3A3A44' }}>
              <span>Date</span>
              <span className="text-right">Shots</span>
              <span className="text-right">Bounces</span>
              <span className="text-right">Rallies</span>
            </div>
            {withStats.slice(0, 4).map((g) => (
              <Link
                key={g.id}
                href={`/recordings/${g.id}`}
                className="grid grid-cols-4 text-[11px] px-1 py-1 rounded transition-colors"
                onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.04)'; }}
                onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = 'transparent'; }}
              >
                <span style={{ color: '#5A5A66' }}>
                  {new Date(g.createdAt).toLocaleDateString("en-US", { month: "short", day: "numeric" })}
                </span>
                <span className="text-white font-medium text-right">{g.shotCount ?? "—"}</span>
                <span className="text-white font-medium text-right">{g.bounceCount ?? "—"}</span>
                <span className="text-white font-medium text-right">{g.rallyCount ?? "—"}</span>
              </Link>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
