"use client";

import { useEffect, useState, useRef } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import ReactMarkdown from "react-markdown";

interface Recording {
  id: string;
  status: string;
  progress: number;
  error: string | null;
  videoUrl: string | null;
  bounceHeatmapUrl: string | null;
  playerHeatmapUrl: string | null;
  createdAt: string;
  filename: string;
  fps: number | null;
  numFrames: number | null;
  bounceCount: number | null;
  shotCount: number | null;
  rallyCount: number | null;
  forehandCount: number | null;
  backhandCount: number | null;
  serveCount: number | null;
  inBoundsBounces: number | null;
  outBoundsBounces: number | null;
  scoutingReport: string | null;
}

function StatCard({
  label,
  value,
  sub,
  accent,
}: {
  label: string;
  value: string | number;
  sub?: string;
  accent?: boolean;
}) {
  return (
    <div className="bg-secondary rounded-xl p-4 border border-gray-700/40">
      <p className="text-xs text-gray-400 mb-1">{label}</p>
      <p className={`text-2xl font-bold ${accent ? "text-accent" : "text-white"}`}>{value}</p>
      {sub && <p className="text-xs text-gray-500 mt-0.5">{sub}</p>}
    </div>
  );
}

function MiniBar({ label, count, total, color }: { label: string; count: number; total: number; color: string }) {
  const pct = total > 0 ? Math.round((count / total) * 100) : 0;
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="text-gray-300">{label}</span>
        <span className="text-white font-semibold">
          {count.toLocaleString()} <span className="text-gray-500">({pct}%)</span>
        </span>
      </div>
      <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
        <div className={`h-full ${color} rounded-full transition-all`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export default function RecordingDetailPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const [recording, setRecording] = useState<Recording | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchRecording = async () => {
    try {
      const res = await fetch(`/api/recordings/${id}`);
      if (!res.ok) {
        setError("Recording not found.");
        return;
      }
      const data = await res.json();
      setRecording(data.recording);

      if (data.recording.status === "done" || data.recording.status === "failed") {
        if (pollRef.current) clearInterval(pollRef.current);
      }
    } catch {
      setError("Failed to load recording.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRecording();
    pollRef.current = setInterval(fetchRecording, 5000);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] gap-3">
        <div className="w-8 h-8 border-2 border-accent border-t-transparent rounded-full animate-spin" />
        <p className="text-sm text-gray-400">Loading match data…</p>
      </div>
    );
  }

  if (error || !recording) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4 px-4 text-center">
        <svg viewBox="0 0 24 24" className="w-12 h-12 text-gray-600" fill="none" stroke="currentColor" strokeWidth={1.5}>
          <circle cx="12" cy="12" r="10" />
          <path d="M12 8v4M12 16h.01" />
        </svg>
        <p className="text-white font-semibold">{error ?? "Something went wrong"}</p>
        <Link href="/recordings" className="text-sm text-accent hover:underline">
          ← Back to recordings
        </Link>
      </div>
    );
  }

  const durationSec =
    recording.fps && recording.numFrames
      ? Math.round(recording.numFrames / recording.fps)
      : null;
  const durationStr = durationSec
    ? `${Math.floor(durationSec / 60)}m ${durationSec % 60}s`
    : "—";

  const datePlayed = new Date(recording.createdAt).toLocaleDateString("en-US", {
    weekday: "short",
    year: "numeric",
    month: "long",
    day: "numeric",
  });

  // Stroke totals
  const fh    = recording.forehandCount ?? null;
  const bh    = recording.backhandCount ?? null;
  const serve = recording.serveCount    ?? null;
  const totalStrokes = fh !== null && bh !== null && serve !== null ? fh + bh + serve : null;

  // Bounce quality
  const inB   = recording.inBoundsBounces  ?? null;
  const outB  = recording.outBoundsBounces ?? null;
  const totalBounceTracked = inB !== null && outB !== null ? inB + outB : null;
  const inPct  = totalBounceTracked && totalBounceTracked > 0 ? Math.round((inB! / totalBounceTracked) * 100) : null;

  const hasStrokes = totalStrokes !== null && totalStrokes > 0;
  const hasBounceQuality = totalBounceTracked !== null && totalBounceTracked > 0;

  // Processing state
  if (recording.status === "processing") {
    return (
      <div className="px-5 py-6 max-w-4xl mx-auto">
        <Link href="/recordings" className="text-xs text-gray-400 hover:text-white flex items-center gap-1 mb-6">
          ← Back
        </Link>
        <div className="bg-secondary rounded-2xl p-8 border border-gray-700/40 text-center flex flex-col items-center gap-4">
          <div className="w-10 h-10 border-2 border-accent border-t-transparent rounded-full animate-spin" />
          <div>
            <p className="text-white font-semibold mb-1">Analysing your match…</p>
            <p className="text-sm text-gray-400">
              Your court report will be ready shortly. This page updates automatically.
            </p>
          </div>
          <div className="w-full max-w-xs">
            <div className="flex justify-between text-xs text-gray-500 mb-1">
              <span>Progress</span>
              <span>{Math.round((recording.progress ?? 0) * 100)}%</span>
            </div>
            <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-accent rounded-full transition-all"
                style={{ width: `${Math.round((recording.progress ?? 0) * 100)}%` }}
              />
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (recording.status === "failed") {
    return (
      <div className="px-5 py-6 max-w-4xl mx-auto">
        <Link href="/recordings" className="text-xs text-gray-400 hover:text-white flex items-center gap-1 mb-6">
          ← Back
        </Link>
        <div className="bg-secondary rounded-2xl p-8 border border-red-800/40 text-center flex flex-col items-center gap-3">
          <svg viewBox="0 0 24 24" className="w-10 h-10 text-red-400" fill="none" stroke="currentColor" strokeWidth={1.5}>
            <circle cx="12" cy="12" r="10" />
            <path d="M12 8v4M12 16h.01" />
          </svg>
          <p className="text-white font-semibold">Analysis failed</p>
          {recording.error && <p className="text-xs text-gray-400">{recording.error}</p>}
          <Link href="/upload" className="text-sm text-accent hover:underline mt-2">
            Try uploading again →
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="px-5 py-6 max-w-5xl mx-auto">
      {/* Back nav */}
      <Link href="/recordings" className="text-xs text-gray-400 hover:text-white flex items-center gap-1 mb-5 w-fit">
        <svg viewBox="0 0 24 24" className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth={2}>
          <path d="M19 12H5M12 5l-7 7 7 7" />
        </svg>
        Back to recordings
      </Link>

      {/* Header */}
      <div className="flex flex-wrap items-start justify-between gap-3 mb-6">
        <div>
          <h1 className="text-xl font-bold text-white">{recording.filename}</h1>
          <p className="text-sm text-gray-400 mt-0.5">{datePlayed}</p>
        </div>
        <span className="bg-accent/15 text-accent text-xs font-medium px-3 py-1 rounded-full">
          Complete
        </span>
      </div>

      {/* Video */}
      {recording.videoUrl && (
        <div className="rounded-2xl overflow-hidden border border-gray-700/40 mb-6 bg-black">
          <video
            src={recording.videoUrl}
            controls
            className="w-full max-h-[480px] object-contain"
          />
        </div>
      )}

      {/* Primary stats strip */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
        <StatCard
          label="Duration"
          value={durationStr}
          sub={durationSec ? `${recording.numFrames?.toLocaleString()} frames` : undefined}
        />
        <StatCard
          label="Total Shots"
          value={recording.shotCount ?? "—"}
          sub={recording.shotCount !== null ? "detected hits" : "stats not available"}
          accent={recording.shotCount !== null}
        />
        <StatCard
          label="Ball Bounces"
          value={recording.bounceCount ?? "—"}
          sub={
            inPct !== null
              ? `${inPct}% in bounds`
              : recording.bounceCount !== null
              ? "detected bounces"
              : "stats not available"
          }
        />
        <StatCard
          label="Rallies"
          value={recording.rallyCount ?? "—"}
          sub={recording.rallyCount !== null ? "rally exchanges" : "stats not available"}
        />
      </div>

      {/* Stroke Breakdown + Shot Quality side by side */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        {/* Stroke breakdown */}
        <div className="bg-secondary rounded-2xl p-5 border border-gray-700/40">
          <h3 className="text-sm font-semibold text-white mb-4">Stroke Breakdown</h3>
          {hasStrokes ? (
            <div className="flex flex-col gap-3">
              {/* Stacked bar */}
              <div className="flex h-3 rounded-full overflow-hidden gap-px mb-1">
                {fh! > 0 && (
                  <div
                    className="bg-accent"
                    style={{ width: `${Math.round((fh! / totalStrokes!) * 100)}%` }}
                    title={`Forehand: ${fh}`}
                  />
                )}
                {bh! > 0 && (
                  <div
                    className="bg-blue-500"
                    style={{ width: `${Math.round((bh! / totalStrokes!) * 100)}%` }}
                    title={`Backhand: ${bh}`}
                  />
                )}
                {serve! > 0 && (
                  <div
                    className="bg-purple-500"
                    style={{ width: `${Math.round((serve! / totalStrokes!) * 100)}%` }}
                    title={`Serve/Smash: ${serve}`}
                  />
                )}
              </div>
              <MiniBar label="Forehand"      count={fh!}    total={totalStrokes!} color="bg-accent"     />
              <MiniBar label="Backhand"      count={bh!}    total={totalStrokes!} color="bg-blue-500"   />
              <MiniBar label="Serve / Smash" count={serve!} total={totalStrokes!} color="bg-purple-500" />
              <p className="text-xs text-gray-500 pt-1 border-t border-gray-700/40">
                {totalStrokes!.toLocaleString()} strokes classified
              </p>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-6 text-center gap-2">
              <svg viewBox="0 0 24 24" className="w-8 h-8 text-gray-600" fill="none" stroke="currentColor" strokeWidth={1.5}>
                <path d="M9 17H7A5 5 0 0 1 7 7h2M15 7h2a5 5 0 1 1 0 10h-2M8 12h8" />
              </svg>
              <p className="text-xs text-gray-500">
                {recording.forehandCount === null
                  ? "Stroke data not available — re-process to generate"
                  : "No strokes detected"}
              </p>
            </div>
          )}
        </div>

        {/* Shot quality */}
        <div className="bg-secondary rounded-2xl p-5 border border-gray-700/40">
          <h3 className="text-sm font-semibold text-white mb-4">Shot Quality</h3>
          {hasBounceQuality ? (
            <div className="flex flex-col gap-3">
              {/* Circle indicator */}
              <div className="flex items-center gap-4 mb-1">
                <div className="shrink-0 w-20 h-20 relative flex items-center justify-center">
                  <svg viewBox="0 0 36 36" className="w-20 h-20 -rotate-90">
                    <circle cx="18" cy="18" r="15.9" fill="none" stroke="#374151" strokeWidth="3" />
                    <circle
                      cx="18" cy="18" r="15.9" fill="none"
                      stroke="var(--color-accent, #22d3ee)" strokeWidth="3"
                      strokeDasharray={`${inPct} ${100 - inPct!}`}
                      strokeLinecap="round"
                    />
                  </svg>
                  <span className="absolute text-sm font-bold text-white">{inPct}%</span>
                </div>
                <div>
                  <p className="text-xs text-gray-400">In bounds accuracy</p>
                  <p className="text-2xl font-bold text-white">{inB!.toLocaleString()}</p>
                  <p className="text-xs text-gray-500">of {totalBounceTracked!.toLocaleString()} bounces</p>
                </div>
              </div>
              <MiniBar label="In bounds"     count={inB!}  total={totalBounceTracked!} color="bg-accent"   />
              <MiniBar label="Out of bounds" count={outB!} total={totalBounceTracked!} color="bg-red-500"  />
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-6 text-center gap-2">
              <svg viewBox="0 0 24 24" className="w-8 h-8 text-gray-600" fill="none" stroke="currentColor" strokeWidth={1.5}>
                <rect x="3" y="3" width="18" height="18" rx="1" />
                <path d="M3 12h18M12 3v18" />
              </svg>
              <p className="text-xs text-gray-500">
                {recording.inBoundsBounces === null
                  ? "Bounce quality data not available — re-process to generate"
                  : "No bounce data detected"}
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Court Report heatmaps */}
      <div className="bg-secondary rounded-2xl p-5 border border-gray-700/40 mb-6">
        <h3 className="text-sm font-semibold text-white mb-4">Court Report</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {/* Bounce heatmap */}
          <div className="rounded-xl overflow-hidden border border-gray-700/40 bg-gray-900/50">
            {recording.bounceHeatmapUrl ? (
              <>
                <img
                  src={recording.bounceHeatmapUrl}
                  alt="Ball Bounce Map"
                  className="w-full object-contain"
                />
                <div className="px-3 py-2">
                  <p className="text-xs font-medium text-white">Ball Bounce Map</p>
                  <p className="text-[10px] text-gray-500">
                    Where the ball landed — useful for identifying opponent patterns
                  </p>
                </div>
              </>
            ) : (
              <div className="aspect-9/16 flex flex-col items-center justify-center gap-2 p-4 text-center">
                <svg viewBox="0 0 24 24" className="w-8 h-8 text-gray-600" fill="none" stroke="currentColor" strokeWidth={1.5}>
                  <rect x="3" y="3" width="18" height="18" rx="1" />
                  <path d="M3 9h18M3 15h18M9 3v18M15 3v18" />
                </svg>
                <p className="text-xs text-gray-500">Ball Bounce Map</p>
                <p className="text-[10px] text-gray-600">Not generated</p>
              </div>
            )}
          </div>

          {/* Player heatmap */}
          <div className="rounded-xl overflow-hidden border border-gray-700/40 bg-gray-900/50">
            {recording.playerHeatmapUrl ? (
              <>
                <img
                  src={recording.playerHeatmapUrl}
                  alt="Player Movement Map"
                  className="w-full object-contain"
                />
                <div className="px-3 py-2">
                  <p className="text-xs font-medium text-white">Player Movement Map</p>
                  <p className="text-[10px] text-gray-500">
                    Your court coverage — identify areas to improve
                  </p>
                </div>
              </>
            ) : (
              <div className="aspect-9/16 flex flex-col items-center justify-center gap-2 p-4 text-center">
                <svg viewBox="0 0 24 24" className="w-8 h-8 text-gray-600" fill="none" stroke="currentColor" strokeWidth={1.5}>
                  <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
                  <circle cx="9" cy="7" r="4" />
                  <path d="M23 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75" />
                </svg>
                <p className="text-xs text-gray-500">Player Movement Map</p>
                <p className="text-[10px] text-gray-600">Not generated</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Match summary table */}
      <div className="bg-secondary rounded-2xl p-5 border border-gray-700/40">
        <h3 className="text-sm font-semibold text-white mb-3">Match Summary</h3>
        <div className="divide-y divide-gray-700/40">
          {[
            { label: "Date Played",    value: datePlayed },
            { label: "Duration",       value: durationStr },
            { label: "Total Shots",    value: recording.shotCount   !== null ? recording.shotCount.toLocaleString()   : "—" },
            { label: "Ball Bounces",   value: recording.bounceCount !== null ? recording.bounceCount.toLocaleString() : "—" },
            { label: "Rallies",        value: recording.rallyCount  !== null ? recording.rallyCount.toLocaleString()  : "—" },
            { label: "Forehand",       value: recording.forehandCount !== null ? recording.forehandCount.toLocaleString() : "—" },
            { label: "Backhand",       value: recording.backhandCount !== null ? recording.backhandCount.toLocaleString() : "—" },
            { label: "Serve / Smash",  value: recording.serveCount  !== null ? recording.serveCount.toLocaleString()  : "—" },
            { label: "In Bounds",      value: recording.inBoundsBounces  !== null ? `${recording.inBoundsBounces.toLocaleString()} (${inPct ?? "—"}%)` : "—" },
            { label: "Out of Bounds",  value: recording.outBoundsBounces !== null ? recording.outBoundsBounces.toLocaleString() : "—" },
            { label: "Court Report",   value: [recording.bounceHeatmapUrl && "Bounce map", recording.playerHeatmapUrl && "Player map"].filter(Boolean).join(", ") || "Not generated" },
          ].map(({ label, value }) => (
            <div key={label} className="flex justify-between py-2 text-sm">
              <span className="text-gray-400">{label}</span>
              <span className="text-white font-medium">{value}</span>
            </div>
          ))}
        </div>
      </div>

      {/* AI Scouting Report */}
      {recording.scoutingReport && (
        <div className="bg-secondary rounded-2xl p-5 border border-gray-700/40 mt-4">
          <div className="flex items-center gap-2 mb-4">
            <h3 className="text-sm font-semibold text-white">AI Scouting Report</h3>
            <span className="text-[10px] bg-accent/15 text-accent px-2 py-0.5 rounded-full font-medium">
              GPT-4o mini
            </span>
          </div>
          <div className="prose prose-sm prose-invert max-w-none
            prose-headings:text-white prose-headings:font-semibold
            prose-p:text-gray-300 prose-p:leading-relaxed
            prose-strong:text-white
            prose-ul:text-gray-300 prose-li:marker:text-accent">
            <ReactMarkdown>{recording.scoutingReport}</ReactMarkdown>
          </div>
        </div>
      )}
    </div>
  );
}
