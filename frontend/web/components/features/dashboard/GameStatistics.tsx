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
    <div className="bg-secondary rounded-2xl p-5 border border-gray-700/40 h-full flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-white">Shot History</h3>
        <span className="text-xs text-gray-500">last {games.length} sessions</span>
      </div>

      {loading && (
        <div className="flex-1 flex items-center justify-center">
          <div className="w-5 h-5 border-2 border-accent border-t-transparent rounded-full animate-spin" />
        </div>
      )}

      {!loading && withStats.length === 0 && (
        <div className="flex-1 flex flex-col items-center justify-center text-center gap-2 py-4">
          <svg viewBox="0 0 24 24" className="w-8 h-8 text-gray-600" fill="none" stroke="currentColor" strokeWidth={1.5}>
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
          </svg>
          <p className="text-xs text-gray-500">No session data yet</p>
          <Link href="/upload" className="text-xs text-accent hover:underline">
            Analyse a match →
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
                      className="w-full bg-accent/80 group-hover:bg-accent rounded-t transition-colors"
                      style={{ height: `${Math.max(barH, 4)}%` }}
                    />
                  </div>
                  <span className="text-[9px] text-gray-500 group-hover:text-gray-300 transition-colors">
                    {new Date(g.createdAt).toLocaleDateString("en-US", { month: "numeric", day: "numeric" })}
                  </span>
                </Link>
              );
            })}
          </div>

          {/* Table — last 4 */}
          <div className="border-t border-gray-700/40 pt-3">
            <div className="grid grid-cols-4 text-[10px] text-gray-500 mb-1 px-1">
              <span>Date</span>
              <span className="text-right">Shots</span>
              <span className="text-right">Bounces</span>
              <span className="text-right">Rallies</span>
            </div>
            {withStats.slice(0, 4).map((g) => (
              <Link
                key={g.id}
                href={`/recordings/${g.id}`}
                className="grid grid-cols-4 text-[11px] px-1 py-1 rounded hover:bg-white/5 transition-colors"
              >
                <span className="text-gray-400">
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
