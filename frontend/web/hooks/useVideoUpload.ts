import { useState } from 'react';

export type UploadStatus = 'idle' | 'creating upload' | 'uploading' | 'pending' | 'processing' | 'done' | 'failed';

export function useVideoUpload(onUploadComplete?: () => void) {
  const [match_id, setMatchId] = useState<string | null>(null);
  const [status, setStatus] = useState<UploadStatus>('idle');
  const [progress, setProgress] = useState<number>(0);
  const [error, setError] = useState<string | null>(null);

  async function handleFile(file: File) {
    try {
      setStatus('creating upload');
      setError(null);

      // 1️⃣ create upload
      const res = await fetch('/api/create-upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: file.name }),
      });

      const { upload_url, file_key, match_id } = await res.json();
      setMatchId(match_id);

      // 2️⃣ upload file directly to Supabase
      setStatus('uploading');
      const uploadRes = await fetch(upload_url, {
        method: 'PUT',
        headers: {
          'Content-Type': file.type || 'video/mp4',
        },
        body: file,
      });

      if (!uploadRes.ok) {
        const text = await uploadRes.text().catch(() => uploadRes.statusText);
        throw new Error(`Upload failed (${uploadRes.status}): ${text}`);
      }

      // 3️⃣ trigger processing (fire and forget - don't wait!)
      setStatus('pending');
      fetch('/api/trigger-process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file_key, match_id }),
      }).catch(err => console.error('Failed to trigger processing:', err));

      // 4️⃣ start polling
      pollStatus(match_id);

    } catch (err) {
      console.error(err);
      setStatus('failed');
      setError((err as Error).message || 'Upload failed. Please try again.');
    }
  }

  async function pollStatus(match_id: string) {
    const interval = setInterval(async () => {
      const res = await fetch(`/api/status?match_id=${match_id}`);
      const data = await res.json();

      console.log('Poll response:', { status: data.status, progress: data.progress });

      setStatus(data.status);
      setProgress(data.progress || 0);
      setError(data.error);

      if (data.status === 'done') {
        clearInterval(interval);
        console.log('Video URL:', data.videoUrl);
        onUploadComplete && onUploadComplete();
      } else if (data.status === 'failed') {
        clearInterval(interval);
      }
    }, 1500);  // Poll every 1.5 seconds
  }

  const getStatusMessage = () => {
    if (status === 'idle') return 'Ready to upload';
    if (status === 'creating upload') return 'Preparing upload...';
    if (status === 'uploading') return 'Uploading video...';
    if (status === 'pending') return 'Starting processing...';
    if (status === 'processing') {
      const percentage = Math.round(progress * 100);
      if (percentage < 5) return 'Initializing...';
      if (percentage < 45) return `Tracking ball (${percentage}%)`;
      if (percentage < 50) return `Detecting bounces (${percentage}%)`;
      if (percentage < 95) return `Rendering visualization (${percentage}%)`;
      if (percentage < 100) return `Finalizing video (${percentage}%)`;
      return `Processing (${percentage}%)`;
    }
    if (status === 'done') return 'Processing complete!';
    if (status === 'failed') return 'Processing failed';
    return status;
  };

  return {
    match_id,
    status,
    progress,
    error,
    handleFile,
    getStatusMessage,
  };
}
