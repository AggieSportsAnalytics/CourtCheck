'use client';

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { ArrowLeft, Film, Clock, Gauge } from 'lucide-react';
import VideoPlayer from '@/components/features/recordings/VideoPlayer';

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
}

export default function WatchPage() {
  const { id } = useParams<{ id: string }>();
  const [recording, setRecording] = useState<Recording | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchRecording() {
      try {
        const res = await fetch(`/api/recordings/${id}`);
        if (!res.ok) {
          if (res.status === 404) throw new Error('Recording not found');
          throw new Error('Failed to fetch recording');
        }
        const data = await res.json();
        setRecording(data.recording);
      } catch (e) {
        setError((e as Error).message);
      } finally {
        setLoading(false);
      }
    }
    fetchRecording();
  }, [id]);

  if (loading) {
    return (
      <div className="min-h-full flex items-center justify-center">
        <div className="text-gray-400">Loading...</div>
      </div>
    );
  }

  if (error || !recording) {
    return (
      <div className="min-h-full flex flex-col items-center justify-center gap-4">
        <p className="text-red-400">{error || 'Recording not found'}</p>
        <Link href="/recordings" className="text-accent hover:underline text-sm">
          Back to Recordings
        </Link>
      </div>
    );
  }

  if (recording.status !== 'done' || !recording.videoUrl) {
    return (
      <div className="min-h-full flex flex-col items-center justify-center gap-4">
        <p className="text-gray-400">This recording is still being processed.</p>
        <Link href="/recordings" className="text-accent hover:underline text-sm">
          Back to Recordings
        </Link>
      </div>
    );
  }

  const formattedDate = new Date(recording.createdAt).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });

  return (
    <div className="min-h-full bg-[#0a0a0a]">
      {/* Back nav */}
      <div className="px-6 py-4">
        <Link
          href="/recordings"
          className="inline-flex items-center gap-2 text-gray-400 hover:text-white transition-colors text-sm"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Recordings
        </Link>
      </div>

      {/* Video player */}
      <div className="max-w-5xl mx-auto px-4">
        <VideoPlayer src={recording.videoUrl} title={recording.filename} />

        {/* Title and metadata */}
        <div className="mt-4 pb-3">
          <h1 className="text-xl font-bold text-white">{recording.filename}</h1>
          <p className="text-sm text-gray-400 mt-1">{formattedDate}</p>
        </div>

        {/* Technical details */}
        <div className="flex items-center gap-6 py-3 border-t border-gray-800">
          {recording.fps && (
            <div className="flex items-center gap-2 text-sm text-gray-400">
              <Gauge className="w-4 h-4" />
              <span>{recording.fps} FPS</span>
            </div>
          )}
          {recording.numFrames && (
            <div className="flex items-center gap-2 text-sm text-gray-400">
              <Film className="w-4 h-4" />
              <span>{recording.numFrames.toLocaleString()} frames</span>
            </div>
          )}
          {recording.fps && recording.numFrames && (
            <div className="flex items-center gap-2 text-sm text-gray-400">
              <Clock className="w-4 h-4" />
              <span>{Math.round(recording.numFrames / recording.fps)}s duration</span>
            </div>
          )}
        </div>

        {/* Heatmaps */}
        {(recording.bounceHeatmapUrl || recording.playerHeatmapUrl) && (
          <div className="mt-6 mb-8">
            <h3 className="text-lg font-semibold text-white mb-4">Match Heatmaps</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {recording.playerHeatmapUrl && (
                <div className="bg-secondary rounded-lg p-4">
                  <h4 className="text-sm font-medium text-gray-300 mb-3">Player Position</h4>
                  <img
                    src={recording.playerHeatmapUrl}
                    alt="Player position heatmap"
                    className="w-full rounded border border-gray-700"
                  />
                </div>
              )}
              {recording.bounceHeatmapUrl && (
                <div className="bg-secondary rounded-lg p-4">
                  <h4 className="text-sm font-medium text-gray-300 mb-3">Ball Bounces</h4>
                  <img
                    src={recording.bounceHeatmapUrl}
                    alt="Ball bounce heatmap"
                    className="w-full rounded border border-gray-700"
                  />
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
