'use client';

import Link from 'next/link';
import { useAuth } from '@/contexts/AuthContext';

function MockCourt() {
  return (
    <svg viewBox="0 0 120 220" fill="none" xmlns="http://www.w3.org/2000/svg" className="w-full h-full">
      {/* Court fill */}
      <rect x="10" y="10" width="100" height="200" fill="rgba(180,240,0,0.03)" stroke="rgba(180,240,0,0.35)" strokeWidth="1.2" />
      {/* Singles sidelines */}
      <rect x="22" y="10" width="76" height="200" stroke="rgba(180,240,0,0.2)" strokeWidth="0.6" fill="none" />
      {/* Net */}
      <line x1="10" y1="110" x2="110" y2="110" stroke="rgba(180,240,0,0.55)" strokeWidth="1.8" />
      {/* Service lines */}
      <line x1="22" y1="75" x2="98" y2="75" stroke="rgba(180,240,0,0.2)" strokeWidth="0.6" />
      <line x1="22" y1="145" x2="98" y2="145" stroke="rgba(180,240,0,0.2)" strokeWidth="0.6" />
      {/* Center service */}
      <line x1="60" y1="75" x2="60" y2="145" stroke="rgba(180,240,0,0.2)" strokeWidth="0.6" />
      {/* Ball traces */}
      <path d="M38,88 Q58,42 78,128" stroke="rgba(180,240,0,0.25)" strokeWidth="1" strokeDasharray="3,3" fill="none" />
      <path d="M78,128 Q90,160 45,172" stroke="rgba(180,240,0,0.18)" strokeWidth="1" strokeDasharray="3,3" fill="none" />
      {/* Bounce dots — in bounds */}
      <circle cx="38" cy="88" r="3.5" fill="#B4F000" fillOpacity="0.9" />
      <circle cx="78" cy="128" r="3.5" fill="#B4F000" fillOpacity="0.9" />
      <circle cx="55" cy="58" r="3" fill="#B4F000" fillOpacity="0.7" />
      <circle cx="85" cy="155" r="3" fill="#B4F000" fillOpacity="0.7" />
      <circle cx="45" cy="172" r="2.5" fill="#B4F000" fillOpacity="0.5" />
      {/* Out of bounds dot */}
      <circle cx="112" cy="95" r="2.5" fill="#FF4444" fillOpacity="0.7" />
      <circle cx="8" cy="140" r="2.5" fill="#FF4444" fillOpacity="0.5" />
    </svg>
  );
}

function MockDashboard() {
  return (
    <div
      className="relative w-full rounded-2xl overflow-hidden"
      style={{
        background: '#0C0C10',
        border: '1px solid rgba(255,255,255,0.08)',
        boxShadow: '0 0 60px rgba(180,240,0,0.06), 0 40px 80px rgba(0,0,0,0.6)',
      }}
    >
      {/* Window chrome */}
      <div
        className="flex items-center gap-2 px-4 py-3"
        style={{ borderBottom: '1px solid rgba(255,255,255,0.06)', background: '#0F0F14' }}
      >
        <div className="flex gap-1.5">
          <div className="w-3 h-3 rounded-full bg-[#FF5F57]" />
          <div className="w-3 h-3 rounded-full bg-[#FEBC2E]" />
          <div className="w-3 h-3 rounded-full bg-[#28C840]" />
        </div>
        <div
          className="mx-auto text-xs font-mono px-4 py-1 rounded-md"
          style={{ background: 'rgba(255,255,255,0.04)', color: '#6B6B78' }}
        >
          courtcheck · session_3.mp4
        </div>
        <div className="w-3 h-3 rounded-full" style={{ background: 'rgba(180,240,0,0.6)' }} />
      </div>

      {/* Content */}
      <div className="p-5 grid grid-cols-2 gap-5">
        {/* Court visualization */}
        <div
          className="rounded-xl p-3 aspect-[120/220]"
          style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)' }}
        >
          <MockCourt />
        </div>

        {/* Stats */}
        <div className="flex flex-col gap-4 justify-center">
          {/* Match numbers */}
          <div className="grid grid-cols-2 gap-2">
            {[
              { label: 'Shots', value: '284' },
              { label: 'Rallies', value: '47' },
              { label: 'In-bounds', value: '68%' },
              { label: 'Bounces', value: '113' },
            ].map(({ label, value }) => (
              <div
                key={label}
                className="rounded-lg p-2.5"
                style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.05)' }}
              >
                <p className="text-[9px] uppercase tracking-widest" style={{ color: '#5A5A66' }}>{label}</p>
                <p className="text-lg font-bold text-white mt-0.5">{value}</p>
              </div>
            ))}
          </div>

          {/* Stroke breakdown */}
          <div>
            <p className="text-[9px] uppercase tracking-widest mb-2" style={{ color: '#5A5A66' }}>Stroke Breakdown</p>
            {[
              { label: 'Forehand', pct: 45, color: '#B4F000' },
              { label: 'Backhand', pct: 38, color: '#60A5FA' },
              { label: 'Serve', pct: 17, color: '#A78BFA' },
            ].map(({ label, pct, color }) => (
              <div key={label} className="mb-1.5">
                <div className="flex justify-between text-[9px] mb-0.5">
                  <span style={{ color: '#8C8C99' }}>{label}</span>
                  <span className="text-white">{pct}%</span>
                </div>
                <div className="h-1.5 rounded-full" style={{ background: 'rgba(255,255,255,0.06)' }}>
                  <div className="h-full rounded-full" style={{ width: `${pct}%`, background: color }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* AI Report teaser */}
      <div
        className="mx-5 mb-5 rounded-xl p-4"
        style={{ background: 'rgba(180,240,0,0.04)', border: '1px solid rgba(180,240,0,0.12)' }}
      >
        <div className="flex items-center gap-2 mb-2">
          <div className="w-1.5 h-1.5 rounded-full" style={{ background: '#B4F000' }} />
          <p className="text-[10px] font-semibold tracking-widest uppercase" style={{ color: '#B4F000' }}>
            AI Scouting Report
          </p>
        </div>
        <p className="text-xs leading-relaxed" style={{ color: '#8C8C99' }}>
          &ldquo;Strong baseline consistency with excellent forehand cross-court accuracy. Consider targeting the T on your second serve — 71% of errors came from wide placement...&rdquo;
        </p>
      </div>
    </div>
  );
}

export function Hero() {
  const { user } = useAuth();
  return (
    <section
      className="relative min-h-screen flex flex-col justify-center overflow-hidden"
      style={{ background: '#07070A' }}
    >
      {/* Grid texture */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          backgroundImage:
            'linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px)',
          backgroundSize: '72px 72px',
        }}
      />
      {/* Bottom glow */}
      <div
        className="absolute bottom-0 left-1/2 -translate-x-1/2 pointer-events-none"
        style={{
          width: '900px',
          height: '500px',
          background: 'radial-gradient(ellipse, rgba(180,240,0,0.07) 0%, transparent 68%)',
        }}
      />

      <div className="relative max-w-6xl mx-auto px-5 sm:px-8 pt-28 pb-20 grid lg:grid-cols-2 gap-16 items-center">
        {/* Left — copy */}
        <div>
          {/* Badge */}
          <div
            className="inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-medium mb-8"
            style={{
              background: 'rgba(180,240,0,0.08)',
              border: '1px solid rgba(180,240,0,0.2)',
              color: '#B4F000',
            }}
          >
            <span className="w-1.5 h-1.5 rounded-full bg-[#B4F000] animate-pulse" />
            Computer Vision + AI · Tennis Analytics
          </div>

          {/* Headline */}
          <h1 className="text-5xl sm:text-6xl lg:text-7xl font-black tracking-tight leading-[1.05] text-white mb-6">
            See every shot.
            <br />
            <span
              style={{
                background: 'linear-gradient(135deg, #B4F000 0%, #88CC00 50%, #B4F000 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text',
              }}
            >
              Know every move.
            </span>
          </h1>

          {/* Subheadline */}
          <p className="text-lg leading-relaxed mb-10 max-w-md" style={{ color: '#8C8C99' }}>
            CourtCheck turns your raw match footage into deep performance insights — ball tracking, heatmaps, stroke classification, and AI-powered scouting reports.
          </p>

          {/* CTAs */}
          <div className="flex flex-wrap gap-3">
            <Link
              href={user ? '/' : '/auth/login'}
              className="inline-flex items-center gap-2 px-6 py-3 rounded-xl text-sm font-semibold transition-all duration-200"
              style={{ background: '#B4F000', color: '#07070A' }}
              onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = '#C7FF00'; (e.currentTarget as HTMLElement).style.transform = 'translateY(-1px)'; }}
              onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = '#B4F000'; (e.currentTarget as HTMLElement).style.transform = 'translateY(0)'; }}
            >
              {user ? 'Go to Dashboard' : 'Analyze Your Match'}
              <svg viewBox="0 0 16 16" fill="none" className="w-4 h-4">
                <path d="M3 8h10M9 4l4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </Link>
            <a
              href="#how-it-works"
              className="inline-flex items-center gap-2 px-6 py-3 rounded-xl text-sm font-semibold transition-all duration-200"
              style={{
                background: 'rgba(255,255,255,0.05)',
                border: '1px solid rgba(255,255,255,0.1)',
                color: '#FAFAFA',
              }}
              onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.08)'; }}
              onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.05)'; }}
            >
              See How It Works
            </a>
          </div>

          {/* Social proof */}
          <p className="mt-8 text-sm" style={{ color: '#4A4A55' }}>
            AI-powered tennis analytics for competitive players
          </p>
        </div>

        {/* Right — mock dashboard */}
        <div className="w-full max-w-lg mx-auto lg:mx-0">
          <MockDashboard />
        </div>
      </div>

      {/* Scroll indicator */}
      <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-1.5 opacity-40">
        <div className="w-px h-8" style={{ background: 'linear-gradient(to bottom, transparent, #B4F000)' }} />
        <svg viewBox="0 0 16 16" fill="none" className="w-3.5 h-3.5" style={{ color: '#B4F000' }}>
          <path d="M3 6l5 5 5-5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>
    </section>
  );
}
