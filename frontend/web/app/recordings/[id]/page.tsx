"use client";

import { useEffect, useState, useRef } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import ReactMarkdown from "react-markdown";
import rehypeSanitize from "rehype-sanitize";

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

// ─── Shared primitives ──────────────────────────────────────────────────────

function Card({ children, accent = false, className = '' }: { children: React.ReactNode; accent?: boolean; className?: string }) {
  return (
    <div
      className={`rounded-2xl p-5 ${className}`}
      style={
        accent
          ? { background: 'rgba(180,240,0,0.03)', border: '1px solid rgba(180,240,0,0.12)' }
          : { background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }
      }
    >
      {children}
    </div>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <h3 className="text-xs font-semibold uppercase tracking-widest mb-4" style={{ color: '#5A5A66' }}>
      {children}
    </h3>
  );
}

function StatCard({ label, value, sub, accent }: { label: string; value: string | number; sub?: string; accent?: boolean }) {
  return (
    <div
      className="rounded-xl p-4"
      style={
        accent
          ? { background: 'rgba(180,240,0,0.05)', border: '1px solid rgba(180,240,0,0.15)' }
          : { background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)' }
      }
    >
      <p className="text-xs mb-1" style={{ color: '#5A5A66' }}>{label}</p>
      <p className="text-2xl font-black leading-none tracking-tight" style={{ color: accent ? '#B4F000' : '#FAFAFA' }}>{value}</p>
      {sub && <p className="text-xs mt-1" style={{ color: '#3A3A44' }}>{sub}</p>}
    </div>
  );
}

function MiniBar({ label, count, total, color }: { label: string; count: number; total: number; color: string }) {
  const pct = total > 0 ? Math.round((count / total) * 100) : 0;
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span style={{ color: '#9CA3AF' }}>{label}</span>
        <span className="text-white font-semibold">
          {count.toLocaleString()} <span style={{ color: '#3A3A44' }}>({pct}%)</span>
        </span>
      </div>
      <div className="h-1.5 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.06)' }}>
        <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}

// ─── Page ───────────────────────────────────────────────────────────────────

export default function RecordingDetailPage() {
  const { id } = useParams<{ id: string }>();
  const [recording, setRecording] = useState<Recording | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchRecording = async () => {
    try {
      const res = await fetch(`/api/recordings/${id}`);
      if (!res.ok) { setError("Recording not found."); return; }
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
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  // ── Loading ──
  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] gap-3">
        <div
          className="w-8 h-8 rounded-full animate-spin"
          style={{ border: '2px solid rgba(180,240,0,0.15)', borderTopColor: '#B4F000' }}
        />
        <p className="text-sm" style={{ color: '#5A5A66' }}>Loading match data…</p>
      </div>
    );
  }

  // ── Error ──
  if (error || !recording) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4 px-4 text-center">
        <svg viewBox="0 0 24 24" className="w-10 h-10" fill="none" stroke="currentColor" strokeWidth={1.25} style={{ color: '#2A2A33' }}>
          <circle cx="12" cy="12" r="10" />
          <path d="M12 8v4M12 16h.01" />
        </svg>
        <p className="text-white font-semibold">{error ?? "Something went wrong"}</p>
        <Link href="/recordings" className="text-sm transition-opacity hover:opacity-70" style={{ color: '#B4F000' }}>
          ← Back to recordings
        </Link>
      </div>
    );
  }

  // ── Processing ──
  if (recording.status === "processing") {
    return (
      <div className="px-6 py-8 max-w-2xl mx-auto">
        <Link href="/recordings" className="flex items-center gap-1 text-xs mb-8 transition-opacity hover:opacity-70" style={{ color: '#5A5A66' }}>
          <svg viewBox="0 0 24 24" className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth={2}><path d="M19 12H5M12 5l-7 7 7 7" /></svg>
          Back
        </Link>
        <Card>
          <div className="py-6 flex flex-col items-center gap-5 text-center">
            <div
              className="w-12 h-12 rounded-full animate-spin"
              style={{ border: '2px solid rgba(180,240,0,0.15)', borderTopColor: '#B4F000' }}
            />
            <div>
              <p className="text-white font-semibold mb-1">Analysing your match…</p>
              <p className="text-sm" style={{ color: '#5A5A66' }}>Your court report will be ready shortly. This page updates automatically.</p>
            </div>
            <div className="w-full max-w-xs">
              <div className="flex justify-between text-xs mb-1.5" style={{ color: '#3A3A44' }}>
                <span>Progress</span>
                <span>{Math.round((recording.progress ?? 0) * 100)}%</span>
              </div>
              <div className="h-1.5 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.08)' }}>
                <div
                  className="h-full rounded-full transition-all"
                  style={{ width: `${Math.round((recording.progress ?? 0) * 100)}%`, background: '#B4F000' }}
                />
              </div>
            </div>
          </div>
        </Card>
      </div>
    );
  }

  // ── Failed ──
  if (recording.status === "failed") {
    return (
      <div className="px-6 py-8 max-w-2xl mx-auto">
        <Link href="/recordings" className="flex items-center gap-1 text-xs mb-8 transition-opacity hover:opacity-70" style={{ color: '#5A5A66' }}>
          <svg viewBox="0 0 24 24" className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth={2}><path d="M19 12H5M12 5l-7 7 7 7" /></svg>
          Back
        </Link>
        <div
          className="rounded-2xl p-8 text-center flex flex-col items-center gap-3"
          style={{ background: 'rgba(239,68,68,0.05)', border: '1px solid rgba(239,68,68,0.2)' }}
        >
          <svg viewBox="0 0 24 24" className="w-10 h-10 text-red-400" fill="none" stroke="currentColor" strokeWidth={1.5}>
            <circle cx="12" cy="12" r="10" /><path d="M12 8v4M12 16h.01" />
          </svg>
          <p className="text-white font-semibold">Analysis failed</p>
          {recording.error && <p className="text-xs text-red-400/70">{recording.error}</p>}
          <Link href="/upload" className="mt-2 text-sm transition-opacity hover:opacity-70" style={{ color: '#B4F000' }}>
            Try uploading again →
          </Link>
        </div>
      </div>
    );
  }

  // ── Done ──
  const durationSec = recording.fps && recording.numFrames ? Math.round(recording.numFrames / recording.fps) : null;
  const durationStr = durationSec ? `${Math.floor(durationSec / 60)}m ${durationSec % 60}s` : "—";

  const datePlayed = new Date(recording.createdAt).toLocaleDateString("en-US", {
    weekday: "short", year: "numeric", month: "long", day: "numeric",
  });

  const fh    = recording.forehandCount ?? null;
  const bh    = recording.backhandCount ?? null;
  const serve = recording.serveCount    ?? null;
  const totalStrokes = fh !== null && bh !== null && serve !== null ? fh + bh + serve : null;

  const inB  = recording.inBoundsBounces  ?? null;
  const outB = recording.outBoundsBounces ?? null;
  const totalBounceTracked = inB !== null && outB !== null ? inB + outB : null;
  const inPct = totalBounceTracked && totalBounceTracked > 0 ? Math.round((inB! / totalBounceTracked) * 100) : null;

  const hasStrokes = totalStrokes !== null && totalStrokes > 0;
  const hasBounceQuality = totalBounceTracked !== null && totalBounceTracked > 0;

  return (
    <div className="px-6 py-8 max-w-5xl">
      {/* Back nav */}
      <Link
        href="/recordings"
        className="flex items-center gap-1.5 text-xs mb-8 w-fit transition-opacity hover:opacity-70"
        style={{ color: '#5A5A66' }}
      >
        <svg viewBox="0 0 24 24" className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth={2}>
          <path d="M19 12H5M12 5l-7 7 7 7" />
        </svg>
        Back to recordings
      </Link>

      {/* Header */}
      <div className="flex flex-wrap items-start justify-between gap-3 mb-8">
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest mb-2" style={{ color: '#B4F000' }}>
            Match Report
          </p>
          <h1 className="text-2xl font-black text-white tracking-tight">{recording.filename}</h1>
          <p className="text-sm mt-0.5" style={{ color: '#5A5A66' }}>{datePlayed}</p>
        </div>
        <span
          className="text-xs font-semibold px-3 py-1 rounded-full"
          style={{ background: 'rgba(180,240,0,0.1)', color: '#B4F000', border: '1px solid rgba(180,240,0,0.2)' }}
        >
          Complete
        </span>
      </div>

      {/* Video */}
      {recording.videoUrl && (
        <div
          className="rounded-2xl overflow-hidden mb-6 bg-black"
          style={{ border: '1px solid rgba(255,255,255,0.07)' }}
        >
          <video
            src={recording.videoUrl}
            controls
            className="w-full max-h-125 object-contain"
          />
        </div>
      )}

      {/* Primary stats */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-5">
        <StatCard label="Duration"     value={durationStr}            sub={durationSec ? `${recording.numFrames?.toLocaleString()} frames` : undefined} />
        <StatCard label="Total Shots"  value={recording.shotCount ?? "—"}   sub={recording.shotCount !== null ? "detected hits" : "stats not available"}    accent={recording.shotCount !== null} />
        <StatCard label="Ball Bounces" value={recording.bounceCount ?? "—"} sub={inPct !== null ? `${inPct}% in bounds` : recording.bounceCount !== null ? "detected" : "stats not available"} />
        <StatCard label="Rallies"      value={recording.rallyCount ?? "—"}  sub={recording.rallyCount !== null ? "rally exchanges" : "stats not available"} />
      </div>

      {/* Stroke Breakdown + Shot Quality */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-5">
        {/* Stroke breakdown */}
        <Card>
          <SectionLabel>Stroke Breakdown</SectionLabel>
          {hasStrokes ? (
            <div className="flex flex-col gap-3">
              <div className="flex h-2 rounded-full overflow-hidden gap-px mb-1">
                {fh! > 0   && <div style={{ width: `${Math.round((fh! / totalStrokes!) * 100)}%`, background: '#B4F000' }} title={`Forehand: ${fh}`} />}
                {bh! > 0   && <div style={{ width: `${Math.round((bh! / totalStrokes!) * 100)}%`, background: '#60A5FA' }} title={`Backhand: ${bh}`} />}
                {serve! > 0 && <div style={{ width: `${Math.round((serve! / totalStrokes!) * 100)}%`, background: '#A78BFA' }} title={`Serve: ${serve}`} />}
              </div>
              <MiniBar label="Forehand"      count={fh!}    total={totalStrokes!} color="#B4F000" />
              <MiniBar label="Backhand"      count={bh!}    total={totalStrokes!} color="#60A5FA" />
              <MiniBar label="Serve / Smash" count={serve!} total={totalStrokes!} color="#A78BFA" />
              <p className="text-xs pt-2" style={{ color: '#3A3A44', borderTop: '1px solid rgba(255,255,255,0.06)' }}>
                {totalStrokes!.toLocaleString()} strokes classified
              </p>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-6 text-center gap-2">
              <svg viewBox="0 0 24 24" className="w-7 h-7" fill="none" stroke="currentColor" strokeWidth={1.5} style={{ color: '#2A2A33' }}>
                <path d="M9 17H7A5 5 0 0 1 7 7h2M15 7h2a5 5 0 1 1 0 10h-2M8 12h8" />
              </svg>
              <p className="text-xs" style={{ color: '#3A3A44' }}>
                {recording.forehandCount === null ? "Stroke data not available — re-process to generate" : "No strokes detected"}
              </p>
            </div>
          )}
        </Card>

        {/* Shot quality */}
        <Card>
          <SectionLabel>Shot Quality</SectionLabel>
          {hasBounceQuality ? (
            <div className="flex flex-col gap-3">
              <div className="flex items-center gap-4 mb-1">
                <div className="shrink-0 w-20 h-20 relative flex items-center justify-center">
                  <svg viewBox="0 0 36 36" className="w-20 h-20 -rotate-90">
                    <circle cx="18" cy="18" r="15.9" fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="3" />
                    <circle
                      cx="18" cy="18" r="15.9" fill="none"
                      stroke="#B4F000" strokeWidth="3"
                      strokeDasharray={`${inPct} ${100 - inPct!}`}
                      strokeLinecap="round"
                    />
                  </svg>
                  <span className="absolute text-sm font-bold text-white">{inPct}%</span>
                </div>
                <div>
                  <p className="text-xs" style={{ color: '#5A5A66' }}>In bounds accuracy</p>
                  <p className="text-2xl font-black text-white">{inB!.toLocaleString()}</p>
                  <p className="text-xs" style={{ color: '#3A3A44' }}>of {totalBounceTracked!.toLocaleString()} bounces</p>
                </div>
              </div>
              <MiniBar label="In bounds"     count={inB!}  total={totalBounceTracked!} color="#B4F000"  />
              <MiniBar label="Out of bounds" count={outB!} total={totalBounceTracked!} color="#EF4444"  />
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-6 text-center gap-2">
              <svg viewBox="0 0 24 24" className="w-7 h-7" fill="none" stroke="currentColor" strokeWidth={1.5} style={{ color: '#2A2A33' }}>
                <rect x="3" y="3" width="18" height="18" rx="1" /><path d="M3 12h18M12 3v18" />
              </svg>
              <p className="text-xs" style={{ color: '#3A3A44' }}>
                {recording.inBoundsBounces === null ? "Bounce quality not available — re-process to generate" : "No bounce data detected"}
              </p>
            </div>
          )}
        </Card>
      </div>

      {/* Court Report heatmaps */}
      <Card className="mb-5">
        <SectionLabel>Court Report</SectionLabel>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {[
            { url: recording.bounceHeatmapUrl, label: 'Ball Bounce Map', sub: 'Where the ball landed — useful for identifying opponent patterns' },
            { url: recording.playerHeatmapUrl, label: 'Player Movement Map', sub: 'Your court coverage — identify areas to improve' },
          ].map(({ url, label, sub }) => (
            <div
              key={label}
              className="rounded-xl overflow-hidden"
              style={{ border: '1px solid rgba(255,255,255,0.07)', background: 'rgba(255,255,255,0.02)' }}
            >
              {url ? (
                <>
                  <img src={url} alt={label} className="w-full object-contain" />
                  <div className="px-3 py-2">
                    <p className="text-xs font-semibold text-white">{label}</p>
                    <p className="text-[10px] mt-0.5" style={{ color: '#3A3A44' }}>{sub}</p>
                  </div>
                </>
              ) : (
                <div className="aspect-video flex flex-col items-center justify-center gap-2 p-4 text-center">
                  <svg viewBox="0 0 24 24" className="w-7 h-7" fill="none" stroke="currentColor" strokeWidth={1.5} style={{ color: '#2A2A33' }}>
                    <rect x="3" y="3" width="18" height="18" rx="1" /><path d="M3 9h18M3 15h18M9 3v18M15 3v18" />
                  </svg>
                  <p className="text-xs" style={{ color: '#3A3A44' }}>{label}</p>
                  <p className="text-[10px]" style={{ color: '#2A2A33' }}>Not generated</p>
                </div>
              )}
            </div>
          ))}
        </div>
      </Card>

      {/* Match Summary table */}
      <Card className="mb-5">
        <SectionLabel>Match Summary</SectionLabel>
        <div style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }}>
          {[
            { label: "Date Played",   value: datePlayed },
            { label: "Duration",      value: durationStr },
            { label: "Total Shots",   value: recording.shotCount   !== null ? recording.shotCount.toLocaleString()   : "—" },
            { label: "Ball Bounces",  value: recording.bounceCount !== null ? recording.bounceCount.toLocaleString() : "—" },
            { label: "Rallies",       value: recording.rallyCount  !== null ? recording.rallyCount.toLocaleString()  : "—" },
            { label: "Forehand",      value: recording.forehandCount !== null ? recording.forehandCount.toLocaleString() : "—" },
            { label: "Backhand",      value: recording.backhandCount !== null ? recording.backhandCount.toLocaleString() : "—" },
            { label: "Serve / Smash", value: recording.serveCount  !== null ? recording.serveCount.toLocaleString()  : "—" },
            { label: "In Bounds",     value: recording.inBoundsBounces  !== null ? `${recording.inBoundsBounces.toLocaleString()} (${inPct ?? "—"}%)` : "—" },
            { label: "Out of Bounds", value: recording.outBoundsBounces !== null ? recording.outBoundsBounces.toLocaleString() : "—" },
            { label: "Court Report",  value: [recording.bounceHeatmapUrl && "Bounce map", recording.playerHeatmapUrl && "Player map"].filter(Boolean).join(", ") || "Not generated" },
          ].map(({ label, value }) => (
            <div key={label} className="flex justify-between py-2.5 text-sm" style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
              <span style={{ color: '#5A5A66' }}>{label}</span>
              <span className="text-white font-medium">{value}</span>
            </div>
          ))}
        </div>
      </Card>

      {/* AI Scouting Report */}
      {recording.scoutingReport && (
        <Card accent>
          <div className="flex items-center gap-2 mb-4">
            <div className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: '#B4F000' }} />
            <h3 className="text-xs font-semibold uppercase tracking-widest" style={{ color: '#B4F000' }}>
              AI Scouting Report
            </h3>
            <span
              className="ml-auto text-[10px] font-mono px-2 py-0.5 rounded-full"
              style={{ background: 'rgba(180,240,0,0.08)', color: '#B4F000' }}
            >
              GPT-4o mini
            </span>
          </div>
          <div className="prose prose-sm prose-invert max-w-none
            prose-headings:text-white prose-headings:font-semibold prose-headings:text-sm
            prose-p:text-gray-300 prose-p:leading-relaxed prose-p:text-sm
            prose-strong:text-white
            prose-ul:text-gray-300 prose-li:marker:text-accent prose-li:text-sm">
            <ReactMarkdown rehypePlugins={[rehypeSanitize]}>{recording.scoutingReport}</ReactMarkdown>
          </div>
        </Card>
      )}
    </div>
  );
}
