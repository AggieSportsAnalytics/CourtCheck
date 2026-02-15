'use client';

import { useEffect, useMemo, useState } from 'react';

const HeatMaps = () => {
  const [activeSet, setActiveSet] = useState<'set1' | 'set2' | 'set3' | 'overall'>('set1');

  const [loading, setLoading] = useState(true);
  const [heatmaps, setHeatmaps] = useState<{
    matchId: string;
    createdAt: string;
    bounceHeatmapUrl: string | null;
    playerHeatmapUrl: string | null;
  } | null>(null);

  const defaultHeatmapUrls = useMemo(() => {
    const make = (label: string) => {
      const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="750" viewBox="0 0 1200 750">
  <rect width="1200" height="750" fill="#111827"/>
  <rect x="120" y="75" width="960" height="600" rx="24" fill="#0f172a" stroke="#334155" stroke-width="4"/>
  <rect x="210" y="150" width="780" height="450" fill="none" stroke="#e5e7eb" stroke-width="6"/>
  <line x1="600" y1="150" x2="600" y2="600" stroke="#e5e7eb" stroke-width="6"/>
  <line x1="210" y1="315" x2="990" y2="315" stroke="#e5e7eb" stroke-width="6"/>
  <line x1="405" y1="315" x2="405" y2="510" stroke="#e5e7eb" stroke-width="6"/>
  <line x1="795" y1="315" x2="795" y2="510" stroke="#e5e7eb" stroke-width="6"/>
  <line x1="210" y1="510" x2="990" y2="510" stroke="#e5e7eb" stroke-width="6"/>
  <text x="600" y="705" text-anchor="middle" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto" font-size="28" fill="#94a3b8">
    ${label}
  </text>
</svg>`;
      return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
    };

    return {
      ball: make('No ball heatmap yet — run an analysis to generate one'),
      player: make('No player heatmap yet — run an analysis to generate one'),
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    async function fetchLatestHeatmaps() {
      try {
        setLoading(true);
        const res = await fetch('/api/heatmaps/latest');
        if (!res.ok) {
          if (res.status === 404) {
            if (!cancelled) setHeatmaps(null);
            return;
          }
          throw new Error('Failed to fetch heatmaps');
        }
        const data = await res.json();
        if (!cancelled) setHeatmaps(data.heatmaps);
      } catch {
        if (!cancelled) setHeatmaps(null);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    fetchLatestHeatmaps();
    return () => {
      cancelled = true;
    };
  }, []);

  const shotsUrl = heatmaps?.bounceHeatmapUrl ?? null;
  const playerUrl = heatmaps?.playerHeatmapUrl ?? null;
  const showMissingMessage = !loading && !shotsUrl && !playerUrl;

  return (
    <div className="bg-secondary rounded-xl overflow-hidden">
      <div className="p-4">
        <h3 className="text-xl font-bold mb-4">Heat Maps!</h3>

        {/* Set Selection */}
        <div className="grid grid-cols-4 gap-1 mb-4">
          <button
            className={`py-2 rounded-md ${activeSet === 'set1' ? 'bg-gray-600' : 'bg-gray-800'}`}
            onClick={() => setActiveSet('set1')}
          aria-label="View Game 1"
          >
          Game 1
          </button>
          <button
            className={`py-2 rounded-md ${activeSet === 'set2' ? 'bg-gray-600' : 'bg-gray-800'}`}
            onClick={() => setActiveSet('set2')}
          aria-label="View Game 2"
          >
          Game 2
          </button>
          <button
            className={`py-2 rounded-md ${activeSet === 'set3' ? 'bg-gray-600' : 'bg-gray-800'}`}
            onClick={() => setActiveSet('set3')}
          aria-label="View Game 3"
          >
          Game 3
          </button>
          <button
            className={`py-2 rounded-md ${activeSet === 'overall' ? 'bg-gray-600' : 'bg-gray-800'}`}
            onClick={() => setActiveSet('overall')}
            aria-label="View overall statistics"
          >
            Overall
          </button>
        </div>

        {showMissingMessage && (
          <p className="text-sm text-gray-300">
            You need to run the model (upload and process a video) to see your heatmaps.
          </p>
        )}
      </div>

      {/* Heatmaps */}
      <div className="p-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <div className="text-sm text-gray-300 mb-2">Ball heatmap</div>
            <img
              src={shotsUrl ?? defaultHeatmapUrls.ball}
              alt="Ball bounce heatmap"
              className="w-full rounded-lg border border-gray-700"
            />
          </div>
          <div>
            <div className="text-sm text-gray-300 mb-2">Player heatmap</div>
            <img
              src={playerUrl ?? defaultHeatmapUrls.player}
              alt="Player position heatmap"
              className="w-full rounded-lg border border-gray-700"
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default HeatMaps;
