"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useAuth } from "@/contexts/AuthContext";

interface Summary {
  totals: { done: number; total: number };
  tennisStats: {
    totalShots: number;
    totalBounces: number;
    totalRallies: number;
    totalForehands: number;
    totalBackhands: number;
    totalServes: number;
    totalInBounds: number;
    totalOutBounds: number;
  };
  hasTennisStats: boolean;
}

export default function ProfilePage() {
  const { user } = useAuth();
  const [summary, setSummary] = useState<Summary | null>(null);
  const [statsLoading, setStatsLoading] = useState(true);

  useEffect(() => {
    fetch("/api/dashboard/summary")
      .then((r) => r.json())
      .then((d) => setSummary(d))
      .catch(console.error)
      .finally(() => setStatsLoading(false));
  }, []);

  const displayName = user?.user_metadata?.name || user?.email?.split("@")[0] || "Player";
  const email = user?.email || "";
  const avatarUrl: string | undefined = user?.user_metadata?.avatar_url;
  const joinedDate = user?.created_at
    ? new Date(user.created_at).toLocaleDateString("en-US", {
        year: "numeric",
        month: "long",
        day: "numeric",
      })
    : "—";

  const initials = displayName
    .split(" ")
    .filter(Boolean)
    .slice(0, 2)
    .map((p: string) => p[0]?.toUpperCase())
    .join("");

  const ts = summary?.tennisStats;
  const totalStrokes = ts ? ts.totalForehands + ts.totalBackhands + ts.totalServes : 0;
  const totalBounced = ts ? ts.totalInBounds + ts.totalOutBounds : 0;
  const inPct = totalBounced > 0 ? Math.round((ts!.totalInBounds / totalBounced) * 100) : 0;

  const stats = [
    { label: "Sessions analyzed",  value: summary?.totals?.done ?? 0 },
    { label: "Total shots",        value: ts?.totalShots    ?? 0 },
    { label: "Total bounces",      value: ts?.totalBounces  ?? 0 },
    { label: "Total rallies",      value: ts?.totalRallies  ?? 0 },
  ];

  return (
    <div className="px-5 py-6 max-w-2xl mx-auto">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-white">Profile</h2>
        <p className="text-sm text-gray-400 mt-1">Your player profile and career stats</p>
      </div>

      {/* Avatar + identity card */}
      <div className="bg-secondary rounded-2xl p-6 border border-gray-700/40 mb-5 flex items-center gap-5">
        {avatarUrl ? (
          <img
            src={avatarUrl}
            alt={displayName}
            className="w-16 h-16 rounded-full object-cover border-2 border-accent/40 shrink-0"
          />
        ) : (
          <div className="w-16 h-16 rounded-full bg-accent/20 border-2 border-accent/30 flex items-center justify-center text-accent font-bold text-xl shrink-0">
            {initials || "P"}
          </div>
        )}
        <div className="flex-1 min-w-0">
          <h3 className="text-lg font-bold text-white truncate">{displayName}</h3>
          <p className="text-sm text-gray-400 truncate">{email}</p>
          <p className="text-xs text-gray-500 mt-0.5">Joined {joinedDate}</p>
        </div>
        <Link
          href="/settings"
          className="shrink-0 text-xs text-accent hover:underline hidden sm:block"
        >
          Edit profile →
        </Link>
      </div>

      {/* Career stats */}
      <div className="bg-secondary rounded-2xl p-5 border border-gray-700/40 mb-5">
        <h3 className="text-sm font-semibold text-white mb-4">Career Stats</h3>
        {statsLoading ? (
          <div className="flex justify-center py-6">
            <div className="w-5 h-5 border-2 border-accent border-t-transparent rounded-full animate-spin" />
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-3">
            {stats.map((s) => (
              <div key={s.label} className="bg-primary/40 rounded-xl p-3">
                <p className="text-xs text-gray-400">{s.label}</p>
                <p className="text-xl font-bold text-white mt-0.5">{s.value.toLocaleString()}</p>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Stroke breakdown */}
      {!statsLoading && summary?.hasTennisStats && totalStrokes > 0 && (
        <div className="bg-secondary rounded-2xl p-5 border border-gray-700/40 mb-5">
          <h3 className="text-sm font-semibold text-white mb-4">Stroke Breakdown</h3>
          {[
            { label: "Forehand",      count: ts!.totalForehands, color: "bg-accent"     },
            { label: "Backhand",      count: ts!.totalBackhands, color: "bg-blue-500"   },
            { label: "Serve / Smash", count: ts!.totalServes,    color: "bg-purple-500" },
          ].map((s) => {
            const pct = Math.round((s.count / totalStrokes) * 100);
            return (
              <div key={s.label} className="mb-3 last:mb-0">
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-gray-300 flex items-center gap-1.5">
                    <span className={`w-2 h-2 rounded-sm ${s.color}`} />
                    {s.label}
                  </span>
                  <span className="text-white font-semibold">
                    {s.count.toLocaleString()} ({pct}%)
                  </span>
                </div>
                <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                  <div className={`h-full ${s.color} rounded-full`} style={{ width: `${pct}%` }} />
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Court accuracy */}
      {!statsLoading && totalBounced > 0 && (
        <div className="bg-secondary rounded-2xl p-5 border border-gray-700/40 mb-5">
          <h3 className="text-sm font-semibold text-white mb-3">Court Accuracy</h3>
          <div className="flex items-center gap-4">
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
            <div>
              <p className="text-xs text-gray-400 mb-0.5">Balls landing in bounds</p>
              <p className="text-2xl font-bold text-white">{ts!.totalInBounds.toLocaleString()}</p>
              <p className="text-xs text-gray-500">of {totalBounced.toLocaleString()} tracked bounces</p>
            </div>
          </div>
        </div>
      )}

      {/* Quick actions */}
      <div className="flex gap-3">
        <Link
          href="/upload"
          className="flex-1 text-center py-2.5 bg-accent/10 hover:bg-accent/20 border border-accent/20 rounded-xl text-sm text-accent font-medium transition-colors"
        >
          + New Analysis
        </Link>
        <Link
          href="/recordings"
          className="flex-1 text-center py-2.5 bg-secondary hover:bg-white/5 border border-gray-700/40 rounded-xl text-sm text-gray-300 font-medium transition-colors"
        >
          View Recordings
        </Link>
        <Link
          href="/settings"
          className="flex-1 text-center py-2.5 bg-secondary hover:bg-white/5 border border-gray-700/40 rounded-xl text-sm text-gray-300 font-medium transition-colors"
        >
          Settings
        </Link>
      </div>
    </div>
  );
}
