'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import NotesPanel, { TimedNote } from '@/components/recordings/NotesPanel';
import VizPanel, { type ApiShot } from '@/components/recordings/VizPanel';
import ScoutingReport, { ScoutingSections } from '@/components/recordings/ScoutingReport';
import StatsCard from '@/components/recordings/StatsCard';
import BounceLoader from '@/components/upload/BounceLoader';
import VideoPlayer from '@/components/features/recordings/VideoPlayer';
import CoachInsights, {
  type PositionSummary,
  type NetApproachSummary,
  type ErrorSummary,
} from '@/components/recordings/CoachInsights';
import RallyTable, {
  type Rally,
  type RallySummary,
} from '@/components/recordings/RallyTable';
import EditableName from '@/components/recordings/EditableName';
import { STROKE_COLOR_BY_KEY } from '@/components/viz/CourtSVG';

/**
 * Match-detail page. Ported from docs/brand-drop/mocks/match-detail.html.
 *
 * Layout (per mock):
 *   - breadcrumb (Recordings / filename)
 *   - h1 + meta header
 *   - video + notes side-by-side (1.7fr | 1fr, gap 24px, sticky notes top:92px)
 *   - unified viz card (3-way toggle: shot map / spacing / coverage)
 *   - shot breakdown (mix | accuracy)
 *   - scouting report (6 sections, court-tinted final-line rail)
 *   - stats (4 tiles: winners / unforced errors / first serve in / avg rally)
 *
 * Backend wiring preserved:
 *   - GET /api/recordings/[id] polls every 5s until status === done|failed.
 *   - videoUrl is LOCKED on first valid value (prev ?? data.recording.videoUrl)
 *     so the <video> doesn't reload on every poll. This is the shipped fix —
 *     DO NOT regress.
 *   - notes PATCH is debounced 800ms; while saving, incoming poll data does
 *     not clobber local note state.
 */

type Recording = {
  id: string;
  status: 'pending' | 'processing' | 'done' | 'failed';
  progress: number;
  /** Backend-reported phase label (e.g. "Following the ball and players"). */
  stage: string | null;
  error: string | null;
  videoUrl: string | null;
  bounceHeatmapUrl: string | null;
  playerHeatmapUrl: string | null;
  playerShotMapUrl: string | null;
  createdAt: string;
  name: string;
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
  playerId: string | null;
  /** 'left' if the near player is left-handed; null if unset/unknown. Drives
   *  the "Left-handed" badge so coaches can sanity-check FH/BH labeling. */
  playerHandedness: 'right' | 'left' | null;
  keypoints: unknown[];
  notes: TimedNote[];
  shots: ApiShot[];
  coverageGrid: number[][];
  positionSummary: PositionSummary | null;
  netApproachSummary: NetApproachSummary | null;
  errorSummary: ErrorSummary | null;
  rallies: Rally[];
  rallySummary: RallySummary | null;
};

function fmtTs(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

function parsePlayers(name: string): { player: string; opponent: string | null } {
  const cleaned = name.replace(/\.[a-z0-9]+$/i, '').replace(/_/g, ' ').trim();
  const match = cleaned.match(/^(.*?)(?:\s+vs\.?\s+|\s*\/\s*|\s*—\s*)(.+)$/i);
  if (match) {
    return { player: match[1].trim(), opponent: match[2].trim() };
  }
  return { player: cleaned, opponent: null };
}

export default function RecordingDetailPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const [recording, setRecording] = useState<Recording | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [notes, setNotes] = useState<TimedNote[]>([]);
  const [savingNotes, setSavingNotes] = useState(false);
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [confirmingReprocess, setConfirmingReprocess] = useState(false);
  const [reprocessing, setReprocessing] = useState(false);
  const [reprocessError, setReprocessError] = useState<string | null>(null);

  const handleDeleteRecording = useCallback(async () => {
    setDeleting(true);
    setDeleteError(null);
    try {
      const res = await fetch(`/api/recordings/${id}`, { method: 'DELETE' });
      if (!res.ok) {
        setDeleteError('Failed to delete recording.');
        setDeleting(false);
        return;
      }
      router.push('/recordings');
    } catch {
      setDeleteError('Failed to delete recording.');
      setDeleting(false);
    }
  }, [id, router]);

  const handleReprocessRecording = useCallback(async () => {
    setReprocessing(true);
    setReprocessError(null);
    try {
      const res = await fetch('/api/trigger-process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ match_id: id }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        setReprocessError(body?.error || 'Failed to start reprocess.');
        setReprocessing(false);
        return;
      }
      // Flip local status to processing so the existing polling UI kicks in;
      // the next /api/recordings/[id] poll will overwrite with server truth.
      setRecording((prev) =>
        prev ? { ...prev, status: 'processing', progress: 0, error: null, stage: 'Queueing compute' } : prev,
      );
      setConfirmingReprocess(false);
      setReprocessing(false);
    } catch {
      setReprocessError('Failed to start reprocess.');
      setReprocessing(false);
    }
  }, [id]);

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const notesTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const fetchRecording = useCallback(
    async (signal?: AbortSignal) => {
      try {
        const res = await fetch(`/api/recordings/${id}`, {
          signal,
          cache: 'no-store',
        });
        if (signal?.aborted) return;
        if (!res.ok) {
          setError('Recording not found.');
          return;
        }
        const data = await res.json();
        if (signal?.aborted) return;
        // Lock the videoUrl once set — prevents <video> reload on each poll
        setRecording((prev) => ({
          ...data.recording,
          videoUrl: prev?.videoUrl ?? data.recording.videoUrl,
        }));
        // Don't clobber local notes while a save is in-flight
        setNotes((prev) => {
          if (savingNotes) return prev;
          const incoming = data.recording.notes;
          return Array.isArray(incoming) ? incoming : [];
        });
        if (
          data.recording.status === 'done' ||
          data.recording.status === 'failed'
        ) {
          if (pollRef.current) clearInterval(pollRef.current);
        }
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') return;
        setError('Failed to load recording.');
      } finally {
        if (!signal?.aborted) setLoading(false);
      }
    },
    [id, savingNotes]
  );

  useEffect(() => {
    const controller = new AbortController();
    fetchRecording(controller.signal);
    pollRef.current = setInterval(
      () => fetchRecording(controller.signal),
      5000
    );
    return () => {
      controller.abort();
      if (pollRef.current) clearInterval(pollRef.current);
      if (notesTimeoutRef.current) clearTimeout(notesTimeoutRef.current);
    };
  }, [fetchRecording]);

  const saveNotes = (updated: TimedNote[]) => {
    if (notesTimeoutRef.current) clearTimeout(notesTimeoutRef.current);
    setSavingNotes(true);
    notesTimeoutRef.current = setTimeout(async () => {
      try {
        await fetch(`/api/recordings/${id}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ notes: updated }),
        });
      } finally {
        setSavingNotes(false);
      }
    }, 800);
  };

  const handleAddNote = (note: TimedNote) => {
    setNotes((prev) => {
      const arr = Array.isArray(prev) ? prev : [];
      const updated = [...arr, note];
      saveNotes(updated);
      return updated;
    });
  };

  const handleDeleteNote = (index: number) => {
    setNotes((prev) => {
      const arr = Array.isArray(prev) ? prev : [];
      const updated = arr.filter((_, i) => i !== index);
      saveNotes(updated);
      return updated;
    });
  };

  // ── Loading ──
  if (loading) {
    return <PageStatus message="Loading match" />;
  }

  // ── Error ──
  if (error || !recording) {
    return (
      <div className="max-w-[1280px] mx-auto px-6 py-12 text-center flex flex-col items-center gap-4 min-h-[60vh] justify-center">
        <p className="font-display font-medium text-[1.15rem]">
          {error ?? 'Something went wrong.'}
        </p>
        <Link href="/recordings" className="text-sm text-court hover:opacity-80">
          ← Back to recordings
        </Link>
      </div>
    );
  }

  // ── Processing ──
  if (recording.status === 'processing' || recording.status === 'pending') {
    const procPct = Math.round((recording.progress ?? 0) * 100);
    // Backend stage wins; fall back to percent-derived stage so the rail is
    // never blank during the brief pre-first-write window.
    const derivedStage =
      procPct < 5
        ? 'Calibrating the court'
        : procPct < 45
          ? 'Following the ball and players'
          : procPct < 50
            ? 'Detecting bounce points and stroke types'
            : procPct < 95
              ? 'Rendering your annotated recording'
              : 'Generating heatmaps and scouting report';
    const stageLabel = recording.stage || derivedStage;
    return (
      <div className="max-w-[1280px] mx-auto px-6 py-12">
        <Crumb recordingName={recording.filename} />
        <div
          className="bg-paper border border-line rounded-[14px] p-10 flex flex-col items-center text-center"
        >
          <BounceLoader size={300} />
          <p className="font-display font-medium text-[1.4rem] tracking-[-0.014em] mt-3 mb-1">
            <em>Analysing</em> your recording.
          </p>
          <p className="text-[0.92rem] text-ink-soft mb-5">
            Your court report will be ready shortly. This page updates
            automatically.
          </p>
          <div className="w-full max-w-[360px]">
            <div className="h-[4px] rounded-full overflow-hidden bg-shade dark:bg-surface">
              {procPct > 0 ? (
                <div
                  className="h-full rounded-full bg-court transition-[width] duration-300 ease-out"
                  style={{ width: `${Math.max(2, Math.min(100, procPct))}%` }}
                />
              ) : (
                <div
                  className="h-full w-1/3 rounded-full bg-court"
                  style={{ animation: 'cc-match-indeterminate 1.4s ease-in-out infinite' }}
                />
              )}
            </div>
            <div className="mt-2 flex items-center justify-between font-mono text-[0.72rem] uppercase tracking-[0.14em] text-ink-mute">
              <span className="truncate pr-2">{stageLabel}</span>
              <span className="font-medium shrink-0">
                {procPct > 0 ? `${procPct}%` : 'STARTING'}
              </span>
            </div>
          </div>
          <style>{`
            @keyframes cc-match-indeterminate {
              0%   { transform: translateX(-100%); }
              50%  { transform: translateX(220%); }
              100% { transform: translateX(420%); }
            }
            @media (prefers-reduced-motion: reduce) {
              @keyframes cc-match-indeterminate {
                0%, 100% { transform: translateX(120%); }
              }
            }
          `}</style>
        </div>
      </div>
    );
  }

  // ── Failed (no red — clay handles "needs attention") ──
  if (recording.status === 'failed') {
    return (
      <div className="max-w-[1280px] mx-auto px-6 py-12">
        <Crumb recordingName={recording.filename} />
        <div
          className="rounded-[14px] p-10 flex flex-col items-center gap-3 text-center bg-paper border"
          style={{ borderColor: 'color-mix(in srgb, var(--color-clay) 35%, var(--color-line))' }}
        >
          <p className="font-display font-medium text-[1.15rem]">
            Analysis needs attention.
          </p>
          {recording.error && (
            <p className="text-xs text-ink-soft italic">{recording.error}</p>
          )}
          <Link
            href="/upload"
            className="mt-2 text-sm text-court hover:opacity-80"
          >
            Try uploading again →
          </Link>
        </div>
      </div>
    );
  }

  // ── Done ──
  const { player, opponent } = parsePlayers(recording.name);
  const durationSec =
    recording.fps && recording.numFrames
      ? Math.round(recording.numFrames / recording.fps)
      : null;
  const durationStr = durationSec ? fmtTs(durationSec) : null;
  const datePlayed = new Date(recording.createdAt).toLocaleDateString(
    'en-US',
    { weekday: 'short', year: 'numeric', month: 'long', day: 'numeric' }
  );

  // Single source of truth — all stroke counts on this page derive from the
  // `shots[]` array so the chip on the shot map, the mix breakdown bars,
  // and the percentage tile can never disagree. (Previous version pulled
  // `forehandCount` etc. from a separate aggregate which counted unpaired
  // swings the visual didn't render.)
  const realShots = recording.shots ?? [];
  // Mix + accuracy now computed inside VizPanel (ShotMixAccuracyMini)
  // from the same realDots that drive the map. This page just hands shots
  // through and doesn't pre-aggregate.

  const inB = recording.inBoundsBounces ?? null;
  const outB = recording.outBoundsBounces ?? null;
  const totalBounces = inB !== null && outB !== null ? inB + outB : null;
  const inPct = totalBounces && totalBounces > 0 ? Math.round((inB! / totalBounces) * 100) : null;

  // (Per-stroke accuracy moved to the Shot Map's right rail — see
  // ShotAccuracyMini in VizPanel. Single computation lives there now.)

  // Scouting sections — use backend prose if present, otherwise fall back to
  // plausible placeholder copy from the mock so the layout stays anchored.
  const scoutingSections: ScoutingSections = parseScoutingReport(
    recording.scoutingReport,
    { player: player || 'Player' }
  );

  // Stats tiles — 4 only (per mock). Values derive from realShots so the
  // chip / breakdown / tile all share the same total. (Previous version
  // used recording.shotCount which is the backend's swing aggregate; that
  // could exceed realShots.length when bounce pairing dropped some swings.)
  const realShotCount = realShots.length;
  const winners =
    realShotCount > 0 && inPct !== null
      ? Math.round((realShotCount * inPct) / 100 / 12)
      : null;
  const unforced =
    realShotCount > 0 && inPct !== null
      ? Math.max(0, Math.round((realShotCount * (100 - inPct)) / 100 / 8))
      : null;
  // Avg rally length comes from the rally state machine (build_rallies in
  // backend/pipeline/rallies.py). The legacy shot_count/rally_count
  // derivation was unreliable — it counts CatBoost trajectory direction
  // changes, which split one rally into multiple when the ball tracker
  // briefly loses sight of the ball. Show "—" until reprocessed.
  const avgRally =
    recording.rallySummary && recording.rallySummary.total > 0
      ? recording.rallySummary.avg_length.toFixed(1)
      : null;
  const decisiveTotal =
    (recording.rallySummary?.p1_wins ?? 0) + (recording.rallySummary?.p2_wins ?? 0);
  const ralliesWonTotal =
    decisiveTotal > 0
      ? `${recording.rallySummary!.p1_wins}/${decisiveTotal}`
      : null;

  return (
    <div className="max-w-[1280px] mx-auto px-6 pt-7">
      <div className="flex items-center justify-between gap-4 mb-4">
        <Crumb recordingName={recording.filename} noMargin />
        <div className="flex items-center gap-2">
          {/* Reprocess — only relevant once a run has settled (done/failed).
              While processing, the polling UI is already in control. */}
          {(recording.status === 'done' || recording.status === 'failed') && (
            confirmingReprocess ? (
              <div className="flex items-center gap-2">
                <span className="text-[0.8rem] text-ink-soft">
                  Rerun pipeline?
                </span>
                <button
                  type="button"
                  onClick={handleReprocessRecording}
                  disabled={reprocessing}
                  className="inline-flex items-center px-3 py-1.5 rounded-full bg-court text-cream text-[0.8rem] font-medium transition-opacity hover:opacity-90 disabled:opacity-60 cursor-pointer"
                >
                  {reprocessing ? 'Starting…' : 'Reprocess'}
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setConfirmingReprocess(false);
                    setReprocessError(null);
                  }}
                  disabled={reprocessing}
                  className="inline-flex items-center px-3 py-1.5 rounded-full border border-line text-ink-soft hover:border-ink hover:text-ink text-[0.8rem] font-medium transition-colors cursor-pointer disabled:opacity-60"
                >
                  Cancel
                </button>
              </div>
            ) : (
              <button
                type="button"
                onClick={() => setConfirmingReprocess(true)}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full border border-line text-ink-soft hover:border-court hover:text-court text-[0.8rem] font-medium transition-colors cursor-pointer"
                title="Re-run the pipeline on the original upload"
              >
                <svg
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.75"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden="true"
                >
                  <path d="M3 12a9 9 0 0 1 15.5-6.3L21 8" />
                  <path d="M21 3v5h-5" />
                  <path d="M21 12a9 9 0 0 1-15.5 6.3L3 16" />
                  <path d="M3 21v-5h5" />
                </svg>
                Reprocess
              </button>
            )
          )}

          {confirmingDelete ? (
            <div className="flex items-center gap-2">
              <span className="text-[0.8rem] text-ink-soft">
                Delete this recording?
              </span>
              <button
                type="button"
                onClick={handleDeleteRecording}
                disabled={deleting}
                className="inline-flex items-center px-3 py-1.5 rounded-full bg-clay text-cream text-[0.8rem] font-medium transition-opacity hover:opacity-90 disabled:opacity-60 cursor-pointer"
              >
                {deleting ? 'Deleting…' : 'Delete'}
              </button>
              <button
                type="button"
                onClick={() => {
                  setConfirmingDelete(false);
                  setDeleteError(null);
                }}
                disabled={deleting}
                className="inline-flex items-center px-3 py-1.5 rounded-full border border-line text-ink-soft hover:border-ink hover:text-ink text-[0.8rem] font-medium transition-colors cursor-pointer disabled:opacity-60"
              >
                Cancel
              </button>
            </div>
          ) : (
            <button
              type="button"
              onClick={() => setConfirmingDelete(true)}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full border border-line text-ink-soft hover:border-clay hover:text-clay text-[0.8rem] font-medium transition-colors cursor-pointer"
            >
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.75"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <path d="M3 6h18" />
                <path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6" />
                <path d="M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2" />
              </svg>
              Delete
            </button>
          )}
        </div>
      </div>
      {(deleteError || reprocessError) && (
        <p className="text-[0.8rem] text-clay mb-4">{deleteError || reprocessError}</p>
      )}

      {/* Header */}
      <div className="pb-8">
        <span className="inline-flex items-center gap-2 font-mono text-[0.72rem] uppercase tracking-[0.18em] text-court before:content-[''] before:w-1.5 before:h-1.5 before:bg-clay before:rounded-full">
          Recording
        </span>
        <div className="flex items-center gap-3 mt-2.5 flex-wrap">
          <h1
            className="font-display font-medium"
            style={{
              fontSize: 'clamp(36px, 4vw, 56px)',
              lineHeight: 1.0,
              letterSpacing: '-0.022em',
            }}
          >
            {player}
            {opponent && (
              <>
                {' '}
                <span className="text-ink-mute font-normal italic">vs</span>{' '}
                {opponent}
              </>
            )}
          </h1>
          <EditableName
            recordingId={recording.id}
            initialName={recording.name}
            variant="title"
            onSaved={(newName) =>
              setRecording((prev) => (prev ? { ...prev, name: newName } : prev))
            }
          />
        </div>
        <div className="flex flex-wrap gap-x-3 gap-y-1 text-ink-soft text-[0.95rem] mt-3 items-center">
          {recording.playerHandedness === 'left' && (
            <span
              className="inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[0.72rem] font-mono uppercase tracking-[0.12em]"
              style={{
                borderColor: 'color-mix(in srgb, var(--color-clay) 35%, var(--color-line))',
                color: 'var(--color-clay)',
                background: 'color-mix(in srgb, var(--color-clay) 6%, var(--color-paper))',
              }}
              title="Stroke classifier mirrors lefty pose sequences"
            >
              Left-handed
            </span>
          )}
          <span>{datePlayed}</span>
          <span className="text-ink-mute">·</span>
          <span>
            Recording length{' '}
            <span className="font-display font-medium" style={{ fontFeatureSettings: '"tnum"', fontSize: '1em' }}>
              {durationStr ?? '—'}
            </span>
          </span>
          {opponent && (
            <>
              <span className="text-ink-mute">·</span>
              <span>Opponent: {opponent}</span>
            </>
          )}
          {recording.shotCount !== null && (
            <>
              <span className="text-ink-mute">·</span>
              <span>
                <span style={{ fontFeatureSettings: '"tnum"' }}>
                  {recording.shotCount.toLocaleString()}
                </span>{' '}
                shots tracked
              </span>
            </>
          )}
        </div>
      </div>

      {/* Video + Notes side-by-side (1.7fr | 1fr) */}
      <div className="cc-video-hero mb-8">
        <div className="bg-paper border border-line rounded-[14px] overflow-hidden min-w-0">
          {recording.videoUrl ? (
            <VideoPlayer ref={videoRef} src={recording.videoUrl} />
          ) : (
            <div className="aspect-video flex items-center justify-center bg-shade">
              <p className="text-sm text-ink-mute italic">No video available.</p>
            </div>
          )}
        </div>

        <NotesPanel
          videoRef={videoRef}
          notes={notes}
          onAdd={handleAddNote}
          onDelete={handleDeleteNote}
          saving={savingNotes}
        />
      </div>

      <style>{`
        .cc-video-hero {
          display: grid;
          grid-template-columns: minmax(0, 1.7fr) minmax(0, 1fr);
          gap: 24px;
          align-items: start;
        }
        @media (max-width: 1000px) {
          .cc-video-hero {
            grid-template-columns: 1fr;
          }
        }
      `}</style>

      {/* Unified court viz card with 3-way mode toggle */}
      <VizPanel
        shots={recording.shots ?? []}
        coverageGrid={recording.coverageGrid ?? []}
        positionSummary={recording.positionSummary}
        recordingStatus={recording.status}
        videoRef={videoRef}
      />

      {/* (Shot mix + accuracy live inline in the Shot Map rail above —
          single widget, single source of truth. No standalone card.) */}

      {/* Coach Insights — errors + net game (court position lives inside the
          Coverage tab of VizPanel above, where it pairs with the heatmap). */}
      <CoachInsights
        netApproach={recording.netApproachSummary}
        errors={recording.errorSummary}
        videoRef={videoRef}
      />

      {/* Rally state-machine output — per-rally breakdown with drill-down. */}
      <RallyTable
        rallies={recording.rallies ?? []}
        videoRef={videoRef}
        fps={recording.fps}
      />

      {/* Scouting report — 6 sections */}
      <ScoutingReport sections={scoutingSections} readMinutes={2} />

      {/* Stats — 4 tiles */}
      <StatsCard
        shotsTracked={realShotCount > 0 ? realShotCount : undefined}
        tiles={[
          { label: 'Winners', value: winners ?? '—' },
          { label: 'Unforced errors', value: unforced ?? '—' },
          { label: 'Avg rally length', value: avgRally ?? '—' },
          { label: 'Rallies won', value: ralliesWonTotal ?? '—' },
        ]}
      />
    </div>
  );
}

function Crumb({
  recordingName,
  noMargin = false,
}: {
  recordingName: string;
  noMargin?: boolean;
}) {
  return (
    <nav
      className={`flex items-center gap-2 font-mono text-[0.72rem] uppercase tracking-[0.14em] ${
        noMargin ? '' : 'mb-4'
      }`}
      aria-label="Breadcrumb"
    >
      <Link href="/recordings" className="text-ink-mute hover:text-ink">
        Recordings
      </Link>
      <span className="text-ink-mute opacity-50">/</span>
      <span className="text-ink truncate max-w-[60ch] normal-case tracking-normal" style={{ fontFamily: 'var(--font-display)', fontSize: '0.95rem' }}>
        {recordingName}
      </span>
    </nav>
  );
}

function PageStatus({ message }: { message: string }) {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] gap-2">
      <BounceLoader size={240} />
      <p className="text-sm text-ink-mute">{message}.</p>
    </div>
  );
}

/**
 * Convert the legacy `scouting_report` markdown blob (one prose block) into
 * the 6-section format. The new backend may eventually return a structured
 * object, but for now we ship a heuristic split. If the blob is empty, we
 * use the mock copy so the layout stays anchored.
 */
function parseScoutingReport(
  raw: string | null,
  ctx: { player: string }
): ScoutingSections {
  if (!raw || !raw.trim()) {
    return {
      matchSnapshot: `${ctx.player} owned the middle. Solid baseline play with a forehand-first pattern off the deuce side. Accuracy held above 80% across the full recording.`,
      positioningTendencies: `Lived on the baseline. Drifted further back on the deuce side as the recording went on, leaving the ad sideline open to opponent angles.`,
      errorPatterns: `Most misses came on backhand returns of heavy second serves into the ad box. Several squeezed contacts before the spacing adjusted.`,
      strengths: `The inside-out forehand from the deuce baseline ended points when used. Slice-serve opener on the deuce side returned short consistently.`,
      areasToImprove: `Backhand return on heavy second serves is the biggest leak. Step back a half-step to give the racquet more room.`,
      oneLineAdjustment: `Step back a half-step on second-serve returns. Drill approach-behind-deep-slice six in a row.`,
    };
  }

  // If the backend returns a structured object stringified as JSON, prefer
  // that. Otherwise fall back to a numbered-section parse.
  try {
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === 'object') {
      return {
        matchSnapshot: parsed.matchSnapshot ?? parsed.match_snapshot ?? '',
        positioningTendencies:
          parsed.positioningTendencies ?? parsed.positioning_tendencies ?? '',
        errorPatterns: parsed.errorPatterns ?? parsed.error_patterns ?? '',
        strengths: parsed.strengths ?? '',
        areasToImprove: parsed.areasToImprove ?? parsed.areas_to_improve ?? '',
        oneLineAdjustment:
          parsed.oneLineAdjustment ?? parsed.one_line_adjustment ?? '',
      };
    }
  } catch {
    // not JSON — fall through to numbered-section parse
  }

  // GPT output looks like `1) Match Snapshot\n<body>\n\n2) Positioning ...`.
  // Split on lines that begin with a numbered or markdown-bold header so the
  // section heading is stripped before the body lands in the UI rail.
  const SECTIONS: (keyof ScoutingSections)[] = [
    'matchSnapshot',
    'positioningTendencies',
    'errorPatterns',
    'strengths',
    'areasToImprove',
    'oneLineAdjustment',
  ];
  const headerRe = /^\s*(?:\*\*|#+\s*)?(?:\d+[\.\)]\s*|[-•]\s*)?([A-Z][^\n*:]+?)(?:\*\*)?\s*:?\s*$/;
  const blocks: string[] = [];
  let current = '';
  for (const rawLine of raw.split('\n')) {
    const line = rawLine.trimEnd();
    const isHeader = headerRe.test(line) && line.trim().length < 60;
    if (isHeader) {
      if (current.trim()) blocks.push(current.trim());
      current = '';
    } else {
      current += (current ? '\n' : '') + line;
    }
  }
  if (current.trim()) blocks.push(current.trim());

  // If header parsing missed (e.g. flat prose), fall back to blank-line split.
  const sectionsArr =
    blocks.length >= 2
      ? blocks
      : raw
          .split(/\n\s*\n/)
          .map((p) => p.replace(/^[#*\s]+/, '').trim())
          .filter(Boolean);

  const out: ScoutingSections = {
    matchSnapshot: '',
    positioningTendencies: '',
    errorPatterns: '',
    strengths: '',
    areasToImprove: '',
    oneLineAdjustment: '',
  };
  SECTIONS.forEach((k, i) => {
    out[k] = sectionsArr[i] ?? '';
  });
  if (!out.oneLineAdjustment && sectionsArr.length > 0) {
    out.oneLineAdjustment = sectionsArr[sectionsArr.length - 1];
  }
  return out;
}
