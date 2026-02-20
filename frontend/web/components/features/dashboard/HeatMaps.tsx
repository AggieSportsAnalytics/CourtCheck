'use client';

import { useEffect, useMemo, useState } from 'react';

const HeatMaps = () => {
  const [loading, setLoading] = useState(true);
  const [heatmaps, setHeatmaps] = useState<{
    matchId: string;
    createdAt: string;
    bounceHeatmapUrl: string | null;
    playerHeatmapUrl: string | null;
  } | null>(null);

  const defaultHeatmapSvg = useMemo(() => {
    const make = (label: string) => {
      const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="750" viewBox="0 0 1200 750">
  <rect width="1200" height="750" fill="#0a0a0d"/>
  <rect x="210" y="150" width="780" height="450" fill="none" stroke="#1a1a20" stroke-width="4"/>
  <line x1="600" y1="150" x2="600" y2="600" stroke="#1a1a20" stroke-width="4"/>
  <line x1="210" y1="315" x2="990" y2="315" stroke="#1a1a20" stroke-width="4"/>
  <line x1="405" y1="315" x2="405" y2="510" stroke="#1a1a20" stroke-width="4"/>
  <line x1="795" y1="315" x2="795" y2="510" stroke="#1a1a20" stroke-width="4"/>
  <line x1="210" y1="510" x2="990" y2="510" stroke="#1a1a20" stroke-width="4"/>
  <text x="600" y="400" text-anchor="middle" font-family="ui-sans-serif, system-ui" font-size="18" fill="#2a2a33">${label}</text>
</svg>`;
      return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
    };
    return {
      ball: make('No ball heatmap — analyze a match to generate one'),
      player: make('No player heatmap — analyze a match to generate one'),
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
    return () => { cancelled = true; };
  }, []);

  const shotsUrl = heatmaps?.bounceHeatmapUrl ?? null;
  const playerUrl = heatmaps?.playerHeatmapUrl ?? null;
  const hasHeatmaps = !loading && (shotsUrl || playerUrl);

  return (
    <div
      className="rounded-xl overflow-hidden"
      style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }}
    >
      <div className="p-5 pb-3 flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-white">Heatmaps</h3>
          <p className="text-xs mt-0.5" style={{ color: '#5A5A66' }}>
            {loading
              ? 'Loading…'
              : hasHeatmaps
              ? `Latest · ${new Date(heatmaps!.createdAt).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}`
              : 'Run an analysis to generate heatmaps'}
          </p>
        </div>
        {hasHeatmaps && (
          <span
            className="text-[10px] px-2 py-1 rounded-full font-semibold"
            style={{ background: 'rgba(180,240,0,0.1)', color: '#B4F000', border: '1px solid rgba(180,240,0,0.2)' }}
          >
            Latest
          </span>
        )}
      </div>

      <div className="p-5 pt-2">
        {loading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="aspect-video rounded-lg animate-pulse" style={{ background: 'rgba(255,255,255,0.04)' }} />
            <div className="aspect-video rounded-lg animate-pulse" style={{ background: 'rgba(255,255,255,0.04)' }} />
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <p className="text-[10px] font-semibold uppercase tracking-widest mb-2" style={{ color: '#3A3A44' }}>
                Ball Bounces
              </p>
              <img
                src={shotsUrl ?? defaultHeatmapSvg.ball}
                alt="Ball bounce heatmap"
                className="w-full rounded-lg object-cover"
                style={{ border: '1px solid rgba(255,255,255,0.07)' }}
              />
            </div>
            <div>
              <p className="text-[10px] font-semibold uppercase tracking-widest mb-2" style={{ color: '#3A3A44' }}>
                Player Positions
              </p>
              <img
                src={playerUrl ?? defaultHeatmapSvg.player}
                alt="Player position heatmap"
                className="w-full rounded-lg object-cover"
                style={{ border: '1px solid rgba(255,255,255,0.07)' }}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default HeatMaps;
