'use client';

import { useEffect, useMemo, useState } from 'react';
import Link from 'next/link';

const TABS = [
  { key: 'ball',   label: 'Ball Bounces' },
  { key: 'player', label: 'Shot Positions' },
] as const;

type TabKey = (typeof TABS)[number]['key'];

const STROKE_LEGEND = [
  { label: 'Forehand',  color: 'rgb(80,220,80)' },
  { label: 'Backhand',  color: 'rgb(80,100,220)' },
  { label: 'Serve',     color: 'rgb(255,210,0)' },
  { label: 'Slice',     color: 'rgb(160,60,220)' },
] as const;

interface HeatmapData {
  matchId: string;
  matchName: string;
  createdAt: string;
  bounceHeatmapUrl: string | null;
  playerHeatmapUrl: string | null;
  playerShotMapUrl: string | null;
  bounceCount: number | null;
  shotCount: number | null;
  rallyCount: number | null;
}

const HeatMaps = () => {
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<TabKey>('ball');
  const [heatmaps, setHeatmaps] = useState<HeatmapData | null>(null);

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
      ball:   make('No ball heatmap \u2014 analyze a match to generate one'),
      player: make('No shot positions \u2014 analyze a match to generate one'),
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

  const urlForTab: Record<TabKey, string | null> = {
    ball:   heatmaps?.bounceHeatmapUrl   ?? null,
    player: heatmaps?.playerHeatmapUrl   ?? null,
  };
  const fallbackForTab: Record<TabKey, string> = {
    ball:   defaultHeatmapSvg.ball,
    player: defaultHeatmapSvg.player,
  };

  const activeUrl = urlForTab[activeTab];
  const activeFallback = fallbackForTab[activeTab];
  const hasHeatmaps = !loading && (urlForTab.ball || urlForTab.player);

  return (
    <div
      className="rounded-xl overflow-hidden w-full flex flex-col"
      style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }}
    >
      {/* Header row */}
      <div className="p-5 pb-0 flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-white">Heatmaps</h3>
          <p className="text-xs mt-0.5" style={{ color: '#5A5A66' }}>
            {loading
              ? 'Loading\u2026'
              : hasHeatmaps
              ? `Latest \u00b7 ${new Date(heatmaps!.createdAt).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}`
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

      {/* Tab switcher */}
      <div className="px-5 pt-4 pb-3 flex gap-1">
        {TABS.map(({ key, label }) => {
          const isActive = activeTab === key;
          return (
            <button
              key={key}
              onClick={() => setActiveTab(key)}
              className="text-[11px] font-semibold px-3 py-1.5 rounded-md transition-all duration-200"
              style={{
                background: isActive ? 'rgba(255,255,255,0.08)' : 'transparent',
                color: isActive ? '#FAFAFA' : '#4A4A55',
                border: isActive ? '1px solid rgba(255,255,255,0.1)' : '1px solid transparent',
              }}
            >
              {label}
            </button>
          );
        })}
      </div>

      {/* Heatmap display */}
      <div className="px-5 flex-1 flex flex-col">
        {loading ? (
          <div
            className="rounded-lg animate-pulse"
            style={{ background: 'rgba(255,255,255,0.04)', paddingBottom: '47.5%' }}
          />
        ) : (
          <div
            className="rounded-lg overflow-hidden"
            style={{ border: '1px solid rgba(255,255,255,0.07)', background: '#050507' }}
          >
            <img
              src={activeUrl ?? activeFallback}
              alt={`${activeTab === 'ball' ? 'Ball bounce heatmap' : 'Player shot positions'}`}
              className="w-full block"
            />
          </div>
        )}

        {/* Stroke type legend — shown for all tabs */}
        {!loading && (
          <div className="flex flex-wrap gap-x-4 gap-y-1.5 pt-3 pb-1">
            {STROKE_LEGEND.map(({ label, color }) => (
              <div key={label} className="flex items-center gap-1.5">
                <span
                  className="block rounded-full shrink-0"
                  style={{ width: 8, height: 8, background: color }}
                />
                <span className="text-[10px]" style={{ color: '#4A4A55' }}>{label}</span>
              </div>
            ))}
            <div className="flex items-center gap-1.5">
              <span
                className="block shrink-0"
                style={{
                  width: 10, height: 10,
                  background: 'transparent',
                  position: 'relative',
                  display: 'inline-block',
                }}
              >
                {/* X marker for out-of-bounds */}
                <svg width="10" height="10" viewBox="0 0 10 10">
                  <line x1="1" y1="1" x2="9" y2="9" stroke="rgb(220,0,0)" strokeWidth="2" strokeLinecap="round"/>
                  <line x1="9" y1="1" x2="1" y2="9" stroke="rgb(220,0,0)" strokeWidth="2" strokeLinecap="round"/>
                </svg>
              </span>
              <span className="text-[10px]" style={{ color: '#4A4A55' }}>Out of bounds</span>
            </div>
          </div>
        )}
      </div>

      {/* Match context bar */}
      {!loading && hasHeatmaps && heatmaps && (
        <div className="px-5 pt-4 pb-5">
          <Link
            href={`/recordings/${heatmaps.matchId}`}
            className="block rounded-lg px-4 py-3 transition-colors duration-200"
            style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.05)' }}
          >
            <div className="flex items-center justify-between">
              <div className="min-w-0">
                <p className="text-xs font-semibold text-white truncate">
                  {heatmaps.matchName}
                </p>
                <p className="text-[10px] mt-0.5" style={{ color: '#4A4A55' }}>
                  {new Date(heatmaps.createdAt).toLocaleDateString('en-US', {
                    weekday: 'short',
                    month: 'short',
                    day: 'numeric',
                  })}
                </p>
              </div>
              <div className="flex items-center gap-4 shrink-0 ml-4">
                {heatmaps.shotCount != null && (
                  <div className="text-right">
                    <p className="text-xs font-bold text-white tabular-nums">{heatmaps.shotCount}</p>
                    <p className="text-[9px] uppercase tracking-wider" style={{ color: '#3A3A44' }}>shots</p>
                  </div>
                )}
                {heatmaps.bounceCount != null && (
                  <div className="text-right">
                    <p className="text-xs font-bold text-white tabular-nums">{heatmaps.bounceCount}</p>
                    <p className="text-[9px] uppercase tracking-wider" style={{ color: '#3A3A44' }}>bounces</p>
                  </div>
                )}
                {heatmaps.rallyCount != null && (
                  <div className="text-right">
                    <p className="text-xs font-bold text-white tabular-nums">{heatmaps.rallyCount}</p>
                    <p className="text-[9px] uppercase tracking-wider" style={{ color: '#3A3A44' }}>rallies</p>
                  </div>
                )}
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none" style={{ color: '#3A3A44' }}>
                  <path d="M6 4l4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </div>
            </div>
          </Link>
        </div>
      )}

      {/* Bottom padding when no context bar */}
      {(loading || !hasHeatmaps) && <div className="pb-5" />}
    </div>
  );
};

export default HeatMaps;
