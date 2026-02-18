"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

interface GameEntry {
  id: string;
  createdAt: string;
  bounceCount: number | null;
  shotCount: number | null;
  rallyCount: number | null;
  hasBallHeatmap: boolean;
  hasPlayerHeatmap: boolean;
}

function formatRelativeDate(iso: string): string {
  const d = new Date(iso);
  const now = new Date();
  const diffDays = Math.floor((now.getTime() - d.getTime()) / 86_400_000);
  if (diffDays === 0) return "Today";
  if (diffDays === 1) return "Yesterday";
  if (diffDays < 7) return `${diffDays}d ago`;
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

export default function GamePlay() {
  const [games, setGames] = useState<GameEntry[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/dashboard/summary")
      .then((r) => r.json())
      .then((d) => setGames(d.games ?? []))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="bg-secondary rounded-2xl p-5 border border-gray-700/40 h-full flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-white">Recent Sessions</h3>
        <Link href="/recordings" className="text-xs text-accent hover:underline">
          View all →
        </Link>
      </div>

      {loading && (
        <div className="flex-1 flex items-center justify-center">
          <div className="w-5 h-5 border-2 border-accent border-t-transparent rounded-full animate-spin" />
        </div>
      )}

      {!loading && games.length === 0 && (
        <div className="flex-1 flex flex-col items-center justify-center text-center gap-2 py-4">
          <svg viewBox="0 0 24 24" className="w-8 h-8 text-gray-600" fill="none" stroke="currentColor" strokeWidth={1.5}>
            <circle cx="12" cy="12" r="10" />
            <path d="M8 12h4l2-4" />
          </svg>
          <p className="text-xs text-gray-500">No completed sessions yet</p>
          <Link href="/upload" className="text-xs text-accent hover:underline">
            Upload your first match →
          </Link>
        </div>
      )}

      {!loading && games.length > 0 && (
        <div className="flex flex-col gap-2 flex-1 overflow-y-auto">
          {games.slice(0, 6).map((g) => {
            const hasStats = g.shotCount !== null || g.bounceCount !== null;
            return (
              <Link
                key={g.id}
                href={`/recordings/${g.id}`}
                className="flex items-start gap-3 p-2.5 rounded-xl hover:bg-white/5 transition-colors group"
              >
                {/* Date badge */}
                <div className="shrink-0 w-10 text-center">
                  <p className="text-xs font-bold text-accent">
                    {new Date(g.createdAt).getDate()}
                  </p>
                  <p className="text-[10px] text-gray-500">
                    {new Date(g.createdAt).toLocaleDateString("en-US", { month: "short" })}
                  </p>
                </div>

                {/* Stats */}
                <div className="flex-1 min-w-0">
                  <p className="text-xs text-gray-400">{formatRelativeDate(g.createdAt)}</p>
                  {hasStats ? (
                    <div className="flex gap-2 mt-0.5 flex-wrap">
                      {g.shotCount !== null && (
                        <span className="text-[10px] bg-gray-700/60 text-gray-300 px-1.5 py-0.5 rounded">
                          {g.shotCount} shots
                        </span>
                      )}
                      {g.bounceCount !== null && (
                        <span className="text-[10px] bg-gray-700/60 text-gray-300 px-1.5 py-0.5 rounded">
                          {g.bounceCount} bounces
                        </span>
                      )}
                      {g.rallyCount !== null && (
                        <span className="text-[10px] bg-gray-700/60 text-gray-300 px-1.5 py-0.5 rounded">
                          {g.rallyCount} rallies
                        </span>
                      )}
                    </div>
                  ) : (
                    <p className="text-[10px] text-gray-600 mt-0.5">Stats not available</p>
                  )}
                </div>

                {/* Report badge */}
                {(g.hasBallHeatmap || g.hasPlayerHeatmap) && (
                  <span className="text-[10px] bg-accent/15 text-accent px-1.5 py-0.5 rounded shrink-0">
                    report
                  </span>
                )}
              </Link>
            );
          })}
        </div>
      )}
    </div>
  );
}
