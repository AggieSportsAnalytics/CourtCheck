'use client';

import { Prose } from '@/components/ui/display';

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
 *   - stats (measured bounce and rally totals)
 *
 * Backend wiring preserved:
 *   - GET /api/recordings/[id] polls every 5s until status === done|failed.
 *   - videoUrl stays stable during polls and refreshes after a media error.
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
  inputPath: string | null;
  bounceHeatmapUrl: string | null;
  playerHeatmapUrl: string | null;
  playerShotMapUrl: string | null;
  createdAt: string;
  name: string;
  filename: string;
  favorited: boolean;
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
  playerName: string | null;
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
  const match = cleaned.match(/^(.*?)(?:\s+vs\.?\s+|\s*\/\s*|\s*\u2014\s*)(.+)$/i);
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
  const [notesError, setNotesError] = useState<string | null>(null);
  const savingNotesRef = useRef(false);
  const pendingNotesRef = useRef<{ id: string; notes: TimedNote[] } | null>(null);
  const notesRequestRef = useRef<Promise<void>>(Promise.resolve());
  const flushNotesRef = useRef<() => void>(() => {});
  const notesUnmountedRef = useRef(false);
  const [favoriteError, setFavoriteError] = useState<string | null>(null);
  const [favoriting, setFavoriting] = useState(false);
  const videoErroredRef = useRef(false);
  const videoRefreshRef = useRef(false);
  const videoResumeRef = useRef<number | null>(null);
  const [videoError, setVideoError] = useState<string | null>(null);

  useEffect(() => {
    if (!favoriteError) return;
    const timer = setTimeout(() => setFavoriteError(null), 4500);
    return () => clearTimeout(timer);
  }, [favoriteError]);
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [confirmingReprocess, setConfirmingReprocess] = useState(false);
  const [reprocessing, setReprocessing] = useState(false);
  const [reprocessError, setReprocessError] = useState<string | null>(null);

  const toggleFavorite = useCallback(async () => {
    if (!recording || favoriting) return;
    const next = !recording.favorited;
    setFavoriting(true);
    setFavoriteError(null);
    setRecording((prev) => prev ? { ...prev, favorited: next } : prev);
    try {
      const res = await fetch(`/api/recordings/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ favorited: next }),
      });
      if (!res.ok) throw new Error('Failed to update favorite');
    } catch (err) {
      console.error('Favorite update failed', err);
      setRecording((prev) => prev ? { ...prev, favorited: !next } : prev);
      setFavoriteError("Couldn't update favorite. Try again.");
    } finally {
      setFavoriting(false);
    }
  }, [id, recording, favoriting]);

  const handleDeleteRecording = useCallback(async () => {
    setDeleting(true);
    setDeleteError(null);
    try {
      const res = await fetch(`/api/recordings/${id}`, { method: 'DELETE' });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        const message = typeof body?.error === 'string' ? body.error.trim() : '';
        setDeleteError(
          message && message !== 'Internal server error'
            ? message
            : "Couldn't complete that request. Try again.",
        );
        setDeleting(false);
        return;
      }
      router.push('/recordings');
    } catch {
      setDeleteError("Couldn't complete that request. Try again.");
      setDeleting(false);
    }
  }, [id, router]);

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
          setError(res.status === 404 ? 'Recording not found.' : res.status === 401
            ? 'Your session expired. Sign in again.'
            : "We couldn't load that recording. Go back to Recordings and try again.");
          return;
        }
        const data = await res.json();
        if (signal?.aborted) return;
        setError(null);
        const refreshVideo = videoErroredRef.current;
        // Keep playback stable during polls; replace an expired URL after an error.
        setRecording((prev) => ({
          ...data.recording,
          videoUrl: !prev?.videoUrl || (refreshVideo && prev.videoUrl !== data.recording.videoUrl)
            ? data.recording.videoUrl : prev.videoUrl,
        }));
        // Don't clobber local notes while a save is in-flight
        setNotes((prev) => {
          if (savingNotesRef.current) return prev;
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
        console.error('Recording fetch failed', err);
        setError("We couldn't load that recording. Go back to Recordings and try again.");
      } finally {
        if (!signal?.aborted) setLoading(false);
      }
    },
    [id]
  );

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
        const message = typeof body?.error === 'string' ? body.error.trim() : '';
        setReprocessError(
          message && message !== 'Internal server error'
            ? message
            : "Couldn't complete that request. Try again.",
        );
        setReprocessing(false);
        return;
      }
      // Flip local status to processing so the page swaps to the
      // BounceLoader / progress-bar view immediately, before the first
      // poll lands.
      setRecording((prev) =>
        prev ? { ...prev, status: 'processing', progress: 0, error: null, stage: 'Queueing compute' } : prev,
      );
      // Polling was cleared the first time this recording reached 'done'
      // (see fetchRecording above). A reprocess kicks the row back to
      // 'processing' on the server, so we restart the 5s poll loop here
      // — otherwise the page stays frozen on the processing screen
      // forever even after the new pipeline completes.
      if (pollRef.current) {
        clearInterval(pollRef.current);
      }
      pollRef.current = setInterval(() => fetchRecording(), 5000);
      // Kick one immediate fetch so the real backend progress / stage
      // shows up within a second rather than waiting for the first tick.
      fetchRecording();
      setConfirmingReprocess(false);
      setReprocessing(false);
    } catch {
      setReprocessError("Couldn't complete that request. Try again.");
      setReprocessing(false);
    }
  }, [id, fetchRecording]);

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
    };
  }, [fetchRecording]);

  const persistNotes = useCallback((keepalive = false) => {
    const pending = pendingNotesRef.current;
    if (!pending) return;
    // Serialize requests so an older response cannot replace newer notes.
    const request = async () => {
      if (!keepalive && notesUnmountedRef.current) return;
      try {
        const res = await fetch(`/api/recordings/${pending.id}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ notes: pending.notes }),
          keepalive,
        });
        if (!res.ok) throw new Error(`Notes save failed (${res.status})`);
        if (pendingNotesRef.current === pending) {
          pendingNotesRef.current = null;
          savingNotesRef.current = false;
          setNotesError(null);
        }
      } catch (err) {
        console.error('Notes save failed', err);
        if (pendingNotesRef.current === pending) setNotesError("Couldn't save. Retry");
      } finally {
        if (!pendingNotesRef.current || pendingNotesRef.current === pending) setSavingNotes(false);
      }
    };
    if (keepalive) void request();
    else notesRequestRef.current = notesRequestRef.current.then(request);
  }, []);
  flushNotesRef.current = () => persistNotes(true);

  useEffect(() => {
    notesUnmountedRef.current = false;
    return () => {
      notesUnmountedRef.current = true;
      if (notesTimeoutRef.current) clearTimeout(notesTimeoutRef.current);
      flushNotesRef.current();
    };
  }, []);

  const saveNotes = (updated: TimedNote[]) => {
    if (notesTimeoutRef.current) clearTimeout(notesTimeoutRef.current);
    pendingNotesRef.current = { id, notes: updated };
    savingNotesRef.current = true;
    setSavingNotes(true);
    setNotesError(null);
    notesTimeoutRef.current = setTimeout(() => {
      notesTimeoutRef.current = null;
      persistNotes();
    }, 800);
  };

  const handleAddNote = (note: TimedNote) => {
    const updated = [...notes, note];
    setNotes(updated);
    saveNotes(updated);
  };

  const handleDeleteNote = (index: number) => {
    const updated = notes.filter((_, i) => i !== index);
    setNotes(updated);
    saveNotes(updated);
  };

  const handleVideoError = useCallback(async () => {
    if (videoRefreshRef.current) return;
    videoRefreshRef.current = true;
    videoErroredRef.current = true;
    videoResumeRef.current = videoRef.current?.currentTime ?? 0;
    setVideoError("We couldn't play the recording. Refresh to try again.");
    await fetchRecording();
  }, [fetchRecording]);

  const handleVideoLoaded = useCallback(() => {
    if (videoResumeRef.current !== null && videoRef.current) {
      videoRef.current.currentTime = videoResumeRef.current;
      videoResumeRef.current = null;
    }
    videoErroredRef.current = false;
    videoRefreshRef.current = false;
    setVideoError(null);
  }, []);

  // ── Loading ──
  if (loading) {
    return <PageStatus message="Loading recording" />;
  }

  // ── Error ──
  if (error || !recording) {
    return (
      <div className="max-w-[1280px] mx-auto px-6 py-12 text-center flex flex-col items-center gap-4 min-h-[60vh] justify-center">
        <p className="font-display font-medium text-[1.15rem]">
          {error ?? "We couldn't load that recording. Go back to Recordings and try again."}
        </p>
        <Link href="/recordings" className="text-[0.95rem] text-court hover:opacity-80">
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
            Analyzing your recording.
          </p>
          <Prose className="text-ink-soft mb-5">
            This page updates on its own. Processing takes about 15 minutes.
          </Prose>
          <div className="w-full max-w-[360px]">
            <div className="h-[4px] rounded-full overflow-hidden bg-shade">
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
            <div className="mt-2 flex items-center justify-between font-mono text-[0.82rem] uppercase tracking-[0.12em] text-ink-mute">
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
            Processing failed.
          </p>
          {recording.error && (
            <p className="text-[0.82rem] text-ink-soft">{recording.error}</p>
          )}
          {recording.inputPath && (
            <button type="button" onClick={handleReprocessRecording} disabled={reprocessing}
              className="min-h-11 px-4 py-2 rounded-full bg-court text-cream disabled:opacity-60">
              {reprocessing ? 'Starting…' : 'Reprocess'}
            </button>
          )}
          {reprocessError && <p role="alert" className="text-clay">{reprocessError}</p>}
          <Link
            href="/upload"
            className="mt-2 text-[0.95rem] text-court hover:opacity-80"
          >
            Upload it again
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

  // Scouting sections — backend prose parsed into 6 sections, or null when
  // report generation failed/never ran (renders an explicit empty state; a
  // coach must never read fabricated placeholder analysis as real).
  const scoutingSections: ScoutingSections | null = parseScoutingReport(
    recording.scoutingReport
  );

  // Measured totals from the recorded shots and rally state machine.
  const realShotCount = realShots.length;
  const rallyLengths = (recording.rallies ?? []).map((r) => r.shot_count).filter(Number.isFinite);
  const longestRally = rallyLengths.length > 0 ? Math.max(...rallyLengths) : null;
  // Avg rally length comes from the rally state machine (build_rallies in
  // backend/pipeline/rallies.py). The legacy shot_count/rally_count
  // derivation was unreliable — it counts CatBoost trajectory direction
  // changes, which split one rally into multiple when the ball tracker
  // briefly loses sight of the ball. Show "—" until reprocessed.
  const avgRally =
    recording.rallySummary && recording.rallySummary.total > 0 && Number.isFinite(recording.rallySummary.avg_length)
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
      <div className="flex items-center justify-between gap-4 mb-4 flex-wrap">
        <Crumb recordingName={recording.filename} noMargin />
        <div className="flex items-center gap-2 flex-wrap">
          {/* Reprocess — only relevant once a run has settled (done/failed).
              While processing, the polling UI is already in control. */}
          {recording.inputPath && (
            confirmingReprocess ? (
              <div className="flex items-center gap-2">
                <span className="text-[0.88rem] text-ink-soft">
                  Reprocess this recording?
                </span>
                <button
                  type="button"
                  onClick={handleReprocessRecording}
                  disabled={reprocessing}
                  className="inline-flex items-center px-3 py-1.5 rounded-full bg-court text-cream text-[0.88rem] font-medium transition-opacity hover:opacity-90 disabled:opacity-60 cursor-pointer"
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
                  className="inline-flex items-center px-3 py-1.5 rounded-full border border-line text-ink-soft hover:border-ink hover:text-ink text-[0.88rem] font-medium transition-colors cursor-pointer disabled:opacity-60"
                >
                  Cancel
                </button>
              </div>
            ) : (
              <button
                type="button"
                onClick={() => setConfirmingReprocess(true)}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full border border-line text-ink-soft hover:border-court hover:text-court text-[0.88rem] font-medium transition-colors cursor-pointer"
                title="Process the original recording again"
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
              <span className="text-[0.88rem] text-ink-soft">
                Delete this recording?
              </span>
              <button
                type="button"
                onClick={handleDeleteRecording}
                disabled={deleting}
                className="inline-flex items-center px-3 py-1.5 rounded-full bg-clay text-cream text-[0.88rem] font-medium transition-opacity hover:opacity-90 disabled:opacity-60 cursor-pointer"
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
                className="inline-flex items-center px-3 py-1.5 rounded-full border border-line text-ink-soft hover:border-ink hover:text-ink text-[0.88rem] font-medium transition-colors cursor-pointer disabled:opacity-60"
              >
                Cancel
              </button>
            </div>
          ) : (
            <button
              type="button"
              onClick={() => setConfirmingDelete(true)}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full border border-line text-ink-soft hover:border-clay hover:text-clay text-[0.88rem] font-medium transition-colors cursor-pointer"
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
        <p className="text-[0.88rem] text-clay mb-4">{deleteError || reprocessError}</p>
      )}

      {/* Header */}
      <div className="pb-8">
        <span className="inline-flex items-center gap-2 font-mono text-[0.82rem] uppercase tracking-[0.12em] text-court before:content-[''] before:w-1.5 before:h-1.5 before:bg-clay before:rounded-full">
          Recording · {recording.playerName && recording.playerId ? (
            <Link href={`/players/${recording.playerId}`} className="underline">{recording.playerName}</Link>
          ) : 'Unassigned'}
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
                <span className="text-ink-mute font-normal">vs</span>{' '}
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
          <button
            type="button"
            aria-label={recording.favorited ? 'Remove from favorites' : 'Add to favorites'}
            aria-pressed={recording.favorited}
            onClick={toggleFavorite}
            disabled={favoriting}
            className={`w-9 h-9 shrink-0 rounded-full border inline-flex items-center justify-center cursor-pointer transition-colors ${
              recording.favorited
                ? 'border-amber text-amber'
                : 'border-line text-ink-mute hover:border-amber hover:text-amber'
            }`}
          >
            <svg viewBox="0 0 24 24" width={18} height={18} fill={recording.favorited ? 'currentColor' : 'none'} stroke="currentColor" strokeWidth={1.75} strokeLinecap="round" strokeLinejoin="round">
              <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
            </svg>
          </button>
        </div>
        {favoriteError && <p role="alert" className="text-[0.88rem] text-clay mt-2">{favoriteError}</p>}
        <div className="flex flex-wrap gap-x-3 gap-y-1 text-ink-soft text-[1.02rem] mt-3 items-center">
          {recording.playerHandedness === 'left' && (
            <span
              className="inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[0.82rem] font-mono uppercase tracking-[0.12em]"
              style={{
                borderColor: 'color-mix(in srgb, var(--color-clay) 35%, var(--color-line))',
                color: 'var(--color-clay)',
                background: 'color-mix(in srgb, var(--color-clay) 6%, var(--color-paper))',
              }}
              title="Uses the left-handed stroke setting"
            >
              Left-handed
            </span>
          )}
          <span>{datePlayed}</span>
          <span className="text-ink-mute">·</span>
          <span>
            Recording length{' '}
            <span className="font-display font-medium" style={{ fontFeatureSettings: '"tnum"', fontSize: '1em' }}>
              {durationStr ?? '–'}
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
            <VideoPlayer ref={videoRef} src={recording.videoUrl} onError={handleVideoError} onLoadedMetadata={handleVideoLoaded} />
          ) : (
            <div className="aspect-video flex items-center justify-center bg-shade">
              <p className="text-[0.95rem] text-ink-mute">No recording available.</p>
            </div>
          )}
          {videoError && <p role="alert" className="text-clay p-3">{videoError}</p>}
        </div>

        <NotesPanel
          videoRef={videoRef}
          notes={notes}
          onAdd={handleAddNote}
          onDelete={handleDeleteNote}
          saving={savingNotes}
          error={notesError}
          onRetry={() => saveNotes(notes)}
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
        handedness={recording.playerHandedness}
        playerId={recording.playerId}
        fps={recording.fps}
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

      {/* Scouting report — 6 sections, or explicit empty state when the
          backend never produced one (generation failure / legacy row). */}
      {scoutingSections ? (
        <ScoutingReport sections={scoutingSections} readMinutes={2} />
      ) : (
        <article
          className="bg-paper border border-line rounded-[14px] mb-8"
          style={{ padding: '36px 44px', boxShadow: 'var(--shadow-card)' }}
          aria-label="Scouting report unavailable"
        >
          <span className="inline-flex items-center gap-2 font-mono text-[0.82rem] uppercase tracking-[0.12em] text-court before:content-[''] before:w-1.5 before:h-1.5 before:bg-clay before:rounded-full">
            Scouting report
          </span>
          <p
            className="font-display font-normal text-ink-mute mt-3 m-0"
            style={{ fontSize: '1.12rem', lineHeight: 1.7 }}
          >
            No report was generated. Reprocess the recording to write one.
          </p>
        </article>
      )}

      {/* Measured totals; rally wins appear only when classified. */}
      <StatsCard
        shotsTracked={realShotCount > 0 ? realShotCount : undefined}
        tiles={[
          { label: 'In bounds', value: inPct === null ? '–' : `${inPct}%`, unit: inPct === null ? undefined : `${inB} of ${totalBounces}` },
          { label: 'Rallies', value: recording.rallySummary?.total ?? '–' },
          { label: 'Avg rally length', value: avgRally ?? '–' },
          { label: 'Longest rally', value: longestRally ?? '–' },
          ...(ralliesWonTotal ? [{ label: 'Rallies won', value: ralliesWonTotal }] : []),
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
      className={`flex items-center gap-2 min-w-0 max-w-full font-mono text-[0.82rem] uppercase tracking-[0.12em] ${
        noMargin ? '' : 'mb-4'
      }`}
      aria-label="Breadcrumb"
    >
      <Link href="/recordings" className="text-ink-mute hover:text-ink">
        Recordings
      </Link>
      <span className="text-ink-mute opacity-50">/</span>
      <span className="text-ink truncate min-w-0 max-w-[min(60ch,100%)] normal-case tracking-normal" style={{ fontFamily: 'var(--font-display)', fontSize: '0.95rem' }}>
        {recordingName}
      </span>
    </nav>
  );
}

function PageStatus({ message }: { message: string }) {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] gap-2">
      <BounceLoader size={240} />
      <p className="text-[0.95rem] text-ink-mute">{message}.</p>
    </div>
  );
}

/**
 * Convert the legacy `scouting_report` markdown blob (one prose block) into
 * the 6-section format. The new backend may eventually return a structured
 * object, but for now we ship a heuristic split. If the blob is empty
 * (report generation failed or never ran), return null — the page renders
 * an honest empty state instead of fabricated placeholder prose.
 */
function parseScoutingReport(raw: string | null): ScoutingSections | null {
  if (!raw || !raw.trim()) {
    return null;
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
  // Blocks are assigned to sections by MATCHING THE HEADER TEXT, not by
  // position — the prompt tells the model to omit N/A sections, so a purely
  // positional map shifts every later section under the wrong heading.
  const SECTIONS: (keyof ScoutingSections)[] = [
    'matchSnapshot',
    'positioningTendencies',
    'errorPatterns',
    'strengths',
    'areasToImprove',
    'oneLineAdjustment',
  ];
  const HEADER_KEYWORDS: [RegExp, keyof ScoutingSections][] = [
    [/snapshot/i, 'matchSnapshot'],
    [/position/i, 'positioningTendencies'],
    [/error/i, 'errorPatterns'],
    [/strength/i, 'strengths'],
    [/improve/i, 'areasToImprove'],
    [/adjustment|coaching/i, 'oneLineAdjustment'],
  ];
  // Header text may not contain sentence punctuation — a short prose line
  // ("Kaia held serve well.") must not register as a header.
  const headerRe = /^\s*(?:\*\*|#+\s*)?(?:\d+[\.\)]\s*|[-•]\s*)?([A-Z][^\n*:.,;!?]+?)(?:\*\*)?\s*:?\s*$/;
  const blocks: { header: string; body: string }[] = [];
  let currentHeader = '';
  let current = '';
  for (const rawLine of raw.split('\n')) {
    const line = rawLine.trimEnd();
    const m = line.trim().length < 60 ? line.match(headerRe) : null;
    if (m) {
      if (current.trim()) blocks.push({ header: currentHeader, body: current.trim() });
      currentHeader = m[1] ?? '';
      current = '';
    } else {
      current += (current ? '\n' : '') + line;
    }
  }
  if (current.trim()) blocks.push({ header: currentHeader, body: current.trim() });

  const out: ScoutingSections = {
    matchSnapshot: '',
    positioningTendencies: '',
    errorPatterns: '',
    strengths: '',
    areasToImprove: '',
    oneLineAdjustment: '',
  };

  // Pass 1: label-match blocks to sections via header keywords.
  const unmatched: string[] = [];
  for (const b of blocks) {
    const hit = HEADER_KEYWORDS.find(([re]) => re.test(b.header));
    if (hit && !out[hit[1]]) {
      out[hit[1]] = b.body;
    } else {
      unmatched.push(b.body);
    }
  }

  // Pass 2: unlabeled leftovers (or flat prose with no headers at all) fill
  // the remaining empty sections in order — the old positional behavior.
  const leftovers =
    blocks.length >= 2
      ? unmatched
      : raw
          .split(/\n\s*\n/)
          .map((p) => p.replace(/^[#*\s]+/, '').trim())
          .filter(Boolean);
  let li = 0;
  for (const k of SECTIONS) {
    if (!out[k] && li < leftovers.length) {
      out[k] = leftovers[li++];
    }
  }
  if (!out.oneLineAdjustment && leftovers.length > 0) {
    out.oneLineAdjustment = leftovers[leftovers.length - 1];
  }
  return out;
}
