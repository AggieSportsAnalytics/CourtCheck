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
    <div
      className="rounded-2xl p-5 h-full flex flex-col"
      style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }}
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-white">Recent Sessions</h3>
        <Link
          href="/recordings"
          className="text-xs transition-opacity hover:opacity-70"
          style={{ color: '#B4F000' }}
        >
          View all →
        </Link>
      </div>

      {loading && (
        <div className="flex-1 flex items-center justify-center">
          <div
            className="w-5 h-5 rounded-full animate-spin"
            style={{ border: '2px solid rgba(180,240,0,0.15)', borderTopColor: '#B4F000' }}
          />
        </div>
      )}

      {!loading && games.length === 0 && (
        <div className="flex-1 flex flex-col items-center justify-center text-center gap-2 py-4">
          <svg viewBox="0 0 24 24" className="w-7 h-7" fill="none" stroke="currentColor" strokeWidth={1.5} style={{ color: '#2A2A33' }}>
            <circle cx="12" cy="12" r="10" />
            <path d="M8 12h4l2-4" />
          </svg>
          <p className="text-xs" style={{ color: '#3A3A44' }}>No completed sessions yet</p>
          <Link href="/upload" className="text-xs transition-opacity hover:opacity-70" style={{ color: '#B4F000' }}>
            Upload your first match →
          </Link>
        </div>
      )}

      {!loading && games.length > 0 && (
        <div className="flex flex-col gap-0.5 flex-1 overflow-y-auto">
          {games.slice(0, 6).map((g) => {
            const hasStats = g.shotCount !== null || g.bounceCount !== null;
            return (
              <Link
                key={g.id}
                href={`/recordings/${g.id}`}
                className="flex items-start gap-3 p-2.5 rounded-xl transition-colors group"
                style={{ color: 'inherit' }}
                onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.04)'; }}
                onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = 'transparent'; }}
              >
                {/* Date badge */}
                <div className="shrink-0 w-9 text-center">
                  <p className="text-xs font-bold" style={{ color: '#B4F000' }}>
                    {new Date(g.createdAt).getDate()}
                  </p>
                  <p className="text-[10px]" style={{ color: '#3A3A44' }}>
                    {new Date(g.createdAt).toLocaleDateString("en-US", { month: "short" })}
                  </p>
                </div>

                {/* Stats */}
                <div className="flex-1 min-w-0">
                  <p className="text-xs" style={{ color: '#5A5A66' }}>{formatRelativeDate(g.createdAt)}</p>
                  {hasStats ? (
                    <div className="flex gap-1.5 mt-1 flex-wrap">
                      {g.shotCount !== null && (
                        <span
                          className="text-[10px] px-1.5 py-0.5 rounded"
                          style={{ background: 'rgba(255,255,255,0.05)', color: '#9CA3AF' }}
                        >
                          {g.shotCount} shots
                        </span>
                      )}
                      {g.bounceCount !== null && (
                        <span
                          className="text-[10px] px-1.5 py-0.5 rounded"
                          style={{ background: 'rgba(255,255,255,0.05)', color: '#9CA3AF' }}
                        >
                          {g.bounceCount} bounces
                        </span>
                      )}
                      {g.rallyCount !== null && (
                        <span
                          className="text-[10px] px-1.5 py-0.5 rounded"
                          style={{ background: 'rgba(255,255,255,0.05)', color: '#9CA3AF' }}
                        >
                          {g.rallyCount} rallies
                        </span>
                      )}
                    </div>
                  ) : (
                    <p className="text-[10px] mt-0.5" style={{ color: '#2A2A33' }}>Stats not available</p>
                  )}
                </div>

                {(g.hasBallHeatmap || g.hasPlayerHeatmap) && (
                  <span
                    className="text-[10px] px-1.5 py-0.5 rounded shrink-0"
                    style={{ background: 'rgba(180,240,0,0.08)', color: '#B4F000' }}
                  >
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
