'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { Badge } from '@/components/ui/badge';

interface Recording {
  id: string;
  status: 'pending' | 'processing' | 'done' | 'failed';
  progress: number;
  error: string | null;
  videoUrl: string | null;
  createdAt: string;
  filename: string;
  fps: number | null;
  numFrames: number | null;
}

function StatusBadge({ status }: { status: Recording['status'] }) {
  switch (status) {
    case 'done':
      return <Badge className="bg-green-600 text-white border-transparent">Done</Badge>;
    case 'processing':
      return <Badge className="bg-yellow-600 text-white border-transparent">Processing</Badge>;
    case 'pending':
      return <Badge className="bg-blue-600 text-white border-transparent">Pending</Badge>;
    case 'failed':
      return <Badge className="bg-red-600 text-white border-transparent">Failed</Badge>;
  }
}

function formatDate(dateStr: string) {
  return new Date(dateStr).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
}

export default function RecordingsPage() {
  const [recordings, setRecordings] = useState<Recording[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchRecordings() {
      try {
        const res = await fetch('/api/recordings');
        if (!res.ok) {
          throw new Error('Failed to fetch recordings');
        }
        const data = await res.json();
        setRecordings(data.recordings);
      } catch (e) {
        setError((e as Error).message);
      } finally {
        setLoading(false);
      }
    }

    fetchRecordings();
  }, []);

  if (loading) {
    return (
      <div className="px-4 py-8">
        <h2 className="text-2xl font-bold text-white mb-4">Recordings</h2>
        <p className="text-gray-400">Loading your recordings...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="px-4 py-8">
        <h2 className="text-2xl font-bold text-white mb-4">Recordings</h2>
        <p className="text-red-400">Error: {error}</p>
      </div>
    );
  }

  if (recordings.length === 0) {
    return (
      <div className="px-4 py-8">
        <h2 className="text-2xl font-bold text-white mb-4">Recordings</h2>
        <p className="text-gray-400">Your processed match recordings will appear here.</p>
        <div className="mt-8 p-8 bg-secondary rounded-lg text-center">
          <h3 className="text-xl font-semibold text-white mb-2">No Recordings Yet</h3>
          <p className="text-gray-400 mb-4">
            Upload a tennis match video to get started with your analysis.
          </p>
          <Link
            href="/upload"
            className="inline-block px-6 py-3 bg-accent text-primary font-semibold rounded-lg hover:bg-opacity-90 transition-colors"
          >
            Upload Video
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="px-4 py-8">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-white">Recordings</h2>
        <Link
          href="/upload"
          className="px-4 py-2 bg-accent text-primary font-semibold rounded-lg hover:bg-opacity-90 transition-colors text-sm"
        >
          Upload Video
        </Link>
      </div>

      <div className="space-y-4">
        {recordings.map((rec) => (
          <div
            key={rec.id}
            className="bg-secondary rounded-lg p-4 flex items-center justify-between"
          >
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-3 mb-1">
                <p className="text-white font-medium truncate">{rec.filename}</p>
                <StatusBadge status={rec.status} />
              </div>
              <p className="text-sm text-gray-400">{formatDate(rec.createdAt)}</p>
              {rec.status === 'processing' && (
                <div className="mt-2 w-64">
                  <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-accent rounded-full transition-all duration-300"
                      style={{ width: `${Math.round(rec.progress * 100)}%` }}
                    />
                  </div>
                  <p className="text-xs text-gray-400 mt-1">
                    {Math.round(rec.progress * 100)}%
                  </p>
                </div>
              )}
              {rec.status === 'failed' && rec.error && (
                <p className="text-sm text-red-400 mt-1">{rec.error}</p>
              )}
            </div>

            <div className="ml-4 shrink-0">
              {rec.status === 'done' && (
                <Link
                  href={`/recordings/${rec.id}`}
                  className="px-4 py-2 bg-accent text-primary font-semibold rounded-lg hover:bg-opacity-90 transition-colors text-sm"
                >
                  Watch
                </Link>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
