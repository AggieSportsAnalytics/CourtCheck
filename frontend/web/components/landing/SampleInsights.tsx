'use client';

import { FadeIn } from './FadeIn';

function StatBadge({ value, label, accent = false }: { value: string; label: string; accent?: boolean }) {
  return (
    <div
      className="rounded-xl p-4"
      style={{
        background: accent ? 'rgba(180,240,0,0.05)' : 'rgba(255,255,255,0.03)',
        border: `1px solid ${accent ? 'rgba(180,240,0,0.15)' : 'rgba(255,255,255,0.06)'}`,
      }}
    >
      <p
        className="text-2xl font-black mb-0.5"
        style={accent ? {
          background: 'linear-gradient(135deg, #B4F000, #88CC00)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          backgroundClip: 'text',
        } : { color: '#FAFAFA' }}
      >
        {value}
      </p>
      <p className="text-xs" style={{ color: '#5A5A66' }}>{label}</p>
    </div>
  );
}

function MiniBar({ label, pct, color }: { label: string; pct: number; color: string }) {
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span style={{ color: '#8C8C99' }}>{label}</span>
        <span className="text-white font-semibold">{pct}%</span>
      </div>
      <div className="h-1.5 rounded-full" style={{ background: 'rgba(255,255,255,0.06)' }}>
        <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}

export function SampleInsights() {
  return (
    <section id="insights" className="py-28" style={{ background: '#07070A' }}>
      <div className="max-w-6xl mx-auto px-5 sm:px-8">
        <div className="w-full h-px mb-28" style={{ background: 'rgba(255,255,255,0.06)' }} />

        <FadeIn className="max-w-2xl mb-16">
          <p className="text-xs font-semibold uppercase tracking-widest mb-4" style={{ color: '#B4F000' }}>
            Sample Insights
          </p>
          <h2 className="text-4xl sm:text-5xl font-black tracking-tight text-white leading-tight mb-4">
            See what your data looks like.
          </h2>
          <p className="text-lg leading-relaxed" style={{ color: '#8C8C99' }}>
            Real output from a CourtCheck analysis — this is what you get after every session.
          </p>
        </FadeIn>

        <div className="grid lg:grid-cols-3 gap-5">
          {/* Card 1 — Match Overview */}
          <FadeIn delay={0}>
            <div
              className="rounded-2xl p-6 h-full"
              style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }}
            >
              <div className="flex items-center gap-2 mb-5">
                <div className="w-2 h-2 rounded-full" style={{ background: '#B4F000' }} />
                <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: '#8C8C99' }}>
                  Match Overview
                </p>
              </div>
              <div className="grid grid-cols-2 gap-3 mb-5">
                <StatBadge value="284" label="Total Shots" accent />
                <StatBadge value="47" label="Rallies" />
                <StatBadge value="68%" label="In-Bounds" accent />
                <StatBadge value="1m 52s" label="Avg. Rally" />
              </div>
              <div
                className="rounded-xl p-3 text-xs"
                style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)', color: '#5A5A66' }}
              >
                <span className="font-mono">Session 3.mp4 · 2m 47s duration</span>
              </div>
            </div>
          </FadeIn>

          {/* Card 2 — Stroke Breakdown */}
          <FadeIn delay={100}>
            <div
              className="rounded-2xl p-6 h-full"
              style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }}
            >
              <div className="flex items-center gap-2 mb-5">
                <div className="w-2 h-2 rounded-full" style={{ background: '#A78BFA' }} />
                <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: '#8C8C99' }}>
                  Stroke Breakdown
                </p>
              </div>

              {/* Stacked bar */}
              <div className="flex h-3 rounded-full overflow-hidden gap-px mb-5">
                <div className="h-full rounded-l-full" style={{ width: '45%', background: '#B4F000' }} />
                <div className="h-full" style={{ width: '38%', background: '#60A5FA' }} />
                <div className="h-full rounded-r-full" style={{ width: '17%', background: '#A78BFA' }} />
              </div>

              <div className="flex flex-col gap-3">
                <MiniBar label="Forehand" pct={45} color="#B4F000" />
                <MiniBar label="Backhand" pct={38} color="#60A5FA" />
                <MiniBar label="Serve / Smash" pct={17} color="#A78BFA" />
              </div>

              <p className="text-xs mt-5 pt-4" style={{ color: '#4A4A55', borderTop: '1px solid rgba(255,255,255,0.06)' }}>
                128 strokes classified across 47 rallies
              </p>
            </div>
          </FadeIn>

          {/* Card 3 — AI Scouting Report */}
          <FadeIn delay={200}>
            <div
              className="rounded-2xl p-6 h-full flex flex-col"
              style={{
                background: 'rgba(180,240,0,0.03)',
                border: '1px solid rgba(180,240,0,0.12)',
              }}
            >
              <div className="flex items-center gap-2 mb-5">
                <div className="w-2 h-2 rounded-full animate-pulse" style={{ background: '#B4F000' }} />
                <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: '#B4F000' }}>
                  AI Scouting Report
                </p>
                <span
                  className="ml-auto text-[10px] font-mono px-2 py-0.5 rounded-full"
                  style={{ background: 'rgba(180,240,0,0.1)', color: '#B4F000' }}
                >
                  GPT-4o mini
                </span>
              </div>

              <div className="flex-1 flex flex-col gap-4 text-sm leading-relaxed" style={{ color: '#8C8C99' }}>
                <div>
                  <p className="text-xs font-semibold text-white mb-1">Performance Summary</p>
                  <p>Strong baseline session. 47 rallies with consistent shot placement from the back of the court.</p>
                </div>
                <div>
                  <p className="text-xs font-semibold text-white mb-1">Strengths</p>
                  <p>Forehand cross-court accuracy is excellent — 68% of bounces landed in-bounds, well above average.</p>
                </div>
                <div>
                  <p className="text-xs font-semibold text-white mb-1">Areas to Improve</p>
                  <p>Second serve placement is wide 71% of the time. Targeting the T will open the court significantly.</p>
                </div>
              </div>
            </div>
          </FadeIn>
        </div>
      </div>
    </section>
  );
}
