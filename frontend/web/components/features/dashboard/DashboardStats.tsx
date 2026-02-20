"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

interface Summary {
  tennisStats: {
    totalBounces: number;
    totalShots: number;
    totalRallies: number;
    totalInBounds: number;
    totalOutBounds: number;
  };
  hasTennisStats: boolean;
}

export default function DashboardStats() {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/dashboard/summary")
      .then((r) => r.json())
      .then((d) => setSummary(d))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const has = !loading && summary?.hasTennisStats;
  const ts = summary?.tennisStats;

  const inB  = ts?.totalInBounds  ?? 0;
  const outB = ts?.totalOutBounds ?? 0;
  const totalBounced = inB + outB;
  const courtAccuracyPct = totalBounced > 0 ? Math.round((inB / totalBounced) * 100) : null;

  const items = [
    {
      label: "Total Shots",
      value: loading ? "—" : has ? ts!.totalShots.toLocaleString() : "—",
      sub: has ? "detected hits" : "no data yet",
      accent: false,
      icon: (
        <svg viewBox="0 0 24 24" className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={1.75}>
          <circle cx="12" cy="12" r="10" />
          <path d="M5 12h14M12 5l7 7-7 7" />
        </svg>
      ),
    },
    {
      label: "Ball Bounces",
      value: loading ? "—" : has ? ts!.totalBounces.toLocaleString() : "—",
      sub: has ? "total detected" : "no data yet",
      accent: false,
      icon: (
        <svg viewBox="0 0 24 24" className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={1.75}>
          <path d="M12 22c-4.97 0-9-3.13-9-7 0-2.21 1.34-4.2 3.5-5.5" />
          <path d="M12 2c4.97 0 9 3.13 9 7 0 2.21-1.34 4.2-3.5 5.5" />
          <path d="M3 9l9-7 9 7M3 15l9 7 9-7" />
        </svg>
      ),
    },
    {
      label: "Rallies",
      value: loading ? "—" : has ? ts!.totalRallies.toLocaleString() : "—",
      sub: has ? "across all sessions" : "no data yet",
      accent: false,
      icon: (
        <svg viewBox="0 0 24 24" className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={1.75}>
          <path d="M17 3L7 12l10 9M7 3l10 9-10 9" />
        </svg>
      ),
    },
    {
      label: "Court Accuracy",
      value: loading ? "—" : courtAccuracyPct !== null ? `${courtAccuracyPct}%` : "—",
      sub: courtAccuracyPct !== null ? `${inB} in / ${outB} out` : "no bounce data",
      accent: courtAccuracyPct !== null && courtAccuracyPct >= 70,
      icon: (
        <svg viewBox="0 0 24 24" className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={1.75}>
          <rect x="3" y="3" width="18" height="18" rx="1" />
          <path d="M3 12h18M12 3v18M8 8h8v8H8z" />
        </svg>
      ),
    },
  ];

  return (
    <div className="mb-6">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {items.map((item, i) => (
          <div
            key={i}
            className="rounded-xl p-4 flex gap-3 items-start"
            style={
              item.accent
                ? {
                    background: 'rgba(180,240,0,0.05)',
                    border: '1px solid rgba(180,240,0,0.15)',
                    boxShadow: '0 0 24px rgba(180,240,0,0.06)',
                  }
                : {
                    background: 'rgba(255,255,255,0.02)',
                    border: '1px solid rgba(255,255,255,0.07)',
                  }
            }
          >
            <div
              className="mt-0.5 shrink-0"
              style={{ color: item.accent ? '#B4F000' : has ? '#4A4A55' : '#2A2A33' }}
            >
              {item.icon}
            </div>
            <div className="min-w-0">
              <p className="text-xs mb-1 truncate" style={{ color: '#5A5A66' }}>{item.label}</p>
              <p
                className="text-2xl font-black leading-none tracking-tight"
                style={{ color: item.accent ? '#B4F000' : has ? '#FAFAFA' : '#2A2A33' }}
              >
                {item.value}
              </p>
              <p className="text-xs mt-1 truncate" style={{ color: '#3A3A44' }}>{item.sub}</p>
            </div>
          </div>
        ))}
      </div>

      {!loading && !summary?.hasTennisStats && (
        <div
          className="mt-3 flex items-center gap-2.5 px-4 py-2.5 rounded-xl text-xs"
          style={{
            background: 'rgba(234,179,8,0.06)',
            border: '1px solid rgba(234,179,8,0.15)',
            color: 'rgba(234,179,8,0.7)',
          }}
        >
          <svg viewBox="0 0 24 24" className="w-4 h-4 shrink-0" fill="none" stroke="currentColor" strokeWidth={2}>
            <circle cx="12" cy="12" r="10" />
            <path d="M12 8v4M12 16h.01" />
          </svg>
          <span>
            Stroke and shot stats require the updated pipeline.{" "}
            <Link href="/upload" className="underline underline-offset-2 transition-opacity hover:opacity-80" style={{ color: 'rgba(234,179,8,0.9)' }}>
              Re-upload a match
            </Link>{" "}
            to generate them.
          </span>
        </div>
      )}
    </div>
  );
}
