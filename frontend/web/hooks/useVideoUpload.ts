import { useCallback, useEffect, useRef, useState } from 'react';

export type UploadStatus =
  | 'idle'
  | 'creating upload'
  | 'uploading'
  | 'pending'
  | 'processing'
  | 'done'
  | 'failed';

interface UseVideoUploadReturn {
  match_id: string | null;
  status: UploadStatus;
  /** Upload-phase progress, 0..1 (bytes uploaded / total). */
  uploadProgress: number;
  /** Bytes uploaded so far during the upload phase. */
  bytesUploaded: number;
  /** Total bytes of the selected file. */
  bytesTotal: number;
  /** Processing-phase progress, 0..1 (from /api/status). Pipeline-side. */
  progress: number;
  /** Backend-reported stage label (e.g. "Following the ball and players").
   *  Null while the pipeline hasn't stamped one yet — caller falls back to
   *  deriving a stage from `progress`. */
  stage: string | null;
  error: string | null;
  videoUrl: string | null;
  filename: string | null;
  handleFile: (file: File, options?: UploadOptions) => Promise<void>;
  reset: () => void;
  getStatusMessage: () => string;
}

export interface UploadOptions {
  name?: string;
  matchDate?: string;
}

/**
 * Drives the upload state machine end-to-end:
 *   idle → creating upload → uploading → pending → processing → done | failed
 *
 * Uses raw XHR PUT against the Supabase signed-upload URL so we get real
 * `upload.onprogress` events (the JS client's uploadToSignedUrl swallows them).
 *
 * API contract unchanged: still calls /api/create-upload, /api/trigger-process,
 * and polls /api/status every 1.5s.
 */
export function useVideoUpload(
  onUploadComplete?: (matchId: string) => void,
  playerId?: string | null,
): UseVideoUploadReturn {
  const [match_id, setMatchId] = useState<string | null>(null);
  const [status, setStatus] = useState<UploadStatus>('idle');
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [bytesUploaded, setBytesUploaded] = useState<number>(0);
  const [bytesTotal, setBytesTotal] = useState<number>(0);
  const [progress, setProgress] = useState<number>(0);
  const [stage, setStage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [filename, setFilename] = useState<string | null>(null);

  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const pollAbortRef = useRef<AbortController | null>(null);
  const xhrRef = useRef<XMLHttpRequest | null>(null);
  const onCompleteRef = useRef(onUploadComplete);
  onCompleteRef.current = onUploadComplete;

  useEffect(() => {
    return () => {
      if (intervalRef.current !== null) clearInterval(intervalRef.current);
      if (pollAbortRef.current !== null) pollAbortRef.current.abort();
      if (xhrRef.current !== null) xhrRef.current.abort();
    };
  }, []);

  const reset = useCallback(() => {
    if (intervalRef.current !== null) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (pollAbortRef.current !== null) {
      pollAbortRef.current.abort();
      pollAbortRef.current = null;
    }
    if (xhrRef.current !== null) {
      xhrRef.current.abort();
      xhrRef.current = null;
    }
    setMatchId(null);
    setStatus('idle');
    setUploadProgress(0);
    setBytesUploaded(0);
    setBytesTotal(0);
    setProgress(0);
    setStage(null);
    setError(null);
    setVideoUrl(null);
    setFilename(null);
  }, []);

  const putWithProgress = useCallback(
    (url: string, file: File): Promise<void> =>
      new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhrRef.current = xhr;
        xhr.open('PUT', url, true);
        xhr.setRequestHeader('Content-Type', file.type || 'video/mp4');
        xhr.setRequestHeader('x-upsert', 'false');
        xhr.upload.onprogress = (evt) => {
          if (!evt.lengthComputable) return;
          setBytesUploaded(evt.loaded);
          setBytesTotal(evt.total);
          setUploadProgress(evt.total > 0 ? evt.loaded / evt.total : 0);
        };
        xhr.onload = () => {
          xhrRef.current = null;
          if (xhr.status >= 200 && xhr.status < 300) {
            setUploadProgress(1);
            setBytesUploaded(file.size);
            resolve();
          } else {
            reject(new Error(`Upload failed (${xhr.status}): ${xhr.responseText || xhr.statusText}`));
          }
        };
        xhr.onerror = () => {
          xhrRef.current = null;
          reject(new Error('Network error during upload'));
        };
        xhr.onabort = () => {
          xhrRef.current = null;
          reject(new Error('Upload aborted'));
        };
        xhr.send(file);
      }),
    [],
  );

  const pollStatus = useCallback((id: string) => {
    // Do NOT abort in-flight requests on each tick — that was the source of
    // "progress bar never moves." When Modal is busy hammering Supabase, the
    // /api/status round trip can take >1.5s; aborting every 1.5s meant every
    // fetch was killed before it returned and the state never updated. Just
    // let them resolve; concurrent ones will arrive out-of-order but the last
    // write wins, which is fine for a monotonically-increasing progress.
    let inFlight = false;
    intervalRef.current = setInterval(async () => {
      if (inFlight) return; // skip tick if previous is still running
      inFlight = true;
      try {
        const t0 = performance.now();
        const res = await fetch(`/api/status?match_id=${id}`, {
          cache: 'no-store',
        });
        const data = await res.json();
        const dtMs = Math.round(performance.now() - t0);
        // Diagnostic: log every poll response so we can see in browser devtools
        // whether the chain Modal → Supabase → /api/status → hook is alive.
        // Look for `[Poll]` lines; progress should monotonically rise.
        // eslint-disable-next-line no-console
        console.log(
          `[Poll] +${dtMs}ms status=${data.status} progress=${data.progress} stage=${data.stage ?? 'null'}`,
        );
        setStatus(data.status);
        setProgress(data.progress || 0);
        if (typeof data.stage === 'string') setStage(data.stage || null);
        if (data.error) setError(data.error);
        if (data.videoUrl) setVideoUrl(data.videoUrl);
        if (data.status === 'done') {
          if (intervalRef.current !== null) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
          }
          onCompleteRef.current?.(id);
        } else if (data.status === 'failed') {
          if (intervalRef.current !== null) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
          }
        }
      } catch (err) {
        console.error('Poll fetch error:', err);
      } finally {
        inFlight = false;
      }
    }, 1500);
  }, []);

  const handleFile = useCallback(
    async (file: File, options?: UploadOptions) => {
      try {
        setError(null);
        setFilename(file.name);
        setBytesTotal(file.size);
        setBytesUploaded(0);
        setUploadProgress(0);
        setStatus('creating upload');

        const trimmedName = options?.name?.trim();
        const trimmedMatchDate = options?.matchDate?.trim();
        const createBody: Record<string, unknown> = {
          filename: file.name,
          player_id: playerId ?? null,
        };
        if (trimmedName) createBody.name = trimmedName;
        if (trimmedMatchDate) createBody.matchDate = trimmedMatchDate;

        const createRes = await fetch('/api/create-upload', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(createBody),
        });
        if (!createRes.ok) {
          const body = await createRes.json().catch(() => ({}));
          throw new Error(body?.error || 'Failed to start upload');
        }
        const { upload_url, file_key, match_id: id } = await createRes.json();
        setMatchId(id);

        setStatus('uploading');
        await putWithProgress(upload_url, file);

        setStatus('pending');
        const trigRes = await fetch('/api/trigger-process', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ file_key, match_id: id }),
        });
        if (!trigRes.ok) {
          const body = await trigRes.json().catch(() => ({}));
          throw new Error(body?.error || 'Failed to start processing');
        }

        pollStatus(id);
      } catch (err) {
        console.error(err);
        setStatus('failed');
        setError((err as Error).message || 'Upload failed. Please try again.');
      }
    },
    [playerId, putWithProgress, pollStatus],
  );

  const getStatusMessage = useCallback(() => {
    if (status === 'idle') return 'Ready to upload';
    if (status === 'creating upload') return 'Preparing upload.';
    if (status === 'uploading') return 'Uploading your recording.';
    if (status === 'pending') return 'Starting analysis.';
    if (status === 'processing') {
      const pct = Math.round(progress * 100);
      if (pct < 5) return 'Calibrating the court';
      if (pct < 45) return 'Following the ball and players';
      if (pct < 50) return 'Detecting bounce points and stroke types';
      if (pct < 95) return 'Rendering your annotated recording';
      return 'Generating heatmaps and scouting report';
    }
    if (status === 'done') return 'Recording analyzed.';
    if (status === 'failed') return 'Upload failed.';
    return status;
  }, [status, progress]);

  return {
    match_id,
    status,
    uploadProgress,
    bytesUploaded,
    bytesTotal,
    progress,
    stage,
    error,
    videoUrl,
    filename,
    handleFile,
    reset,
    getStatusMessage,
  };
}
