'use client';

import { FadeIn } from './FadeIn';

const FEATURES = [
  {
    icon: (
      <svg viewBox="0 0 24 24" fill="none" className="w-6 h-6">
        <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="1.5" />
        <path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.93 4.93l2.12 2.12M16.95 16.95l2.12 2.12M4.93 19.07l2.12-2.12M16.95 7.05l2.12-2.12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      </svg>
    ),
    tag: 'Computer Vision',
    title: 'Ball Tracking',
    body: 'Frame-by-frame YOLO-based detection traces every ball trajectory — bounce positions, speed changes, and rally sequences captured with sub-pixel accuracy.',
    glow: 'rgba(180,240,0,0.12)',
    accent: '#B4F000',
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" fill="none" className="w-6 h-6">
        <rect x="3" y="3" width="18" height="18" rx="2" stroke="currentColor" strokeWidth="1.5" />
        <path d="M3 9h18M3 15h18M9 3v18M15 3v18" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      </svg>
    ),
    tag: 'Spatial Analysis',
    title: 'Court Heatmaps',
    body: 'Homography projection maps every ball landing and player position into court-space coordinates, generating density heatmaps that reveal patterns invisible to the naked eye.',
    glow: 'rgba(96,165,250,0.12)',
    accent: '#60A5FA',
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" fill="none" className="w-6 h-6">
        <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
    tag: 'Deep Learning',
    title: 'Stroke Classification',
    body: 'A temporal CNN classifies every detected shot as forehand, backhand, or serve using a rolling video window — giving you an exact stroke breakdown across the full match.',
    glow: 'rgba(167,139,250,0.12)',
    accent: '#A78BFA',
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" fill="none" className="w-6 h-6">
        <path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
    tag: 'GPT-4o mini',
    title: 'AI Scouting Report',
    body: "After every session, GPT-4o mini synthesizes your stats into a personalised coach-quality report — performance summary, identified strengths, and specific areas to improve.",
    glow: 'rgba(251,191,36,0.12)',
    accent: '#FBB724',
  },
];

export function Features() {
  return (
    <section id="features" className="py-28" style={{ background: '#07070A' }}>
      {/* Divider */}
      <div className="max-w-6xl mx-auto px-5 sm:px-8">
        <div className="w-full h-px mb-28" style={{ background: 'rgba(255,255,255,0.06)' }} />

        {/* Header */}
        <FadeIn className="max-w-2xl mb-16">
          <p className="text-xs font-semibold uppercase tracking-widest mb-4" style={{ color: '#B4F000' }}>
            Features
          </p>
          <h2 className="text-4xl sm:text-5xl font-black tracking-tight text-white leading-tight mb-4">
            Everything you need to play smarter.
          </h2>
          <p className="text-lg leading-relaxed" style={{ color: '#8C8C99' }}>
            A complete computer vision pipeline that turns a single video file into a full analytics report — no setup, no sensors.
          </p>
        </FadeIn>

        {/* Feature cards */}
        <div className="grid sm:grid-cols-2 gap-5">
          {FEATURES.map(({ icon, tag, title, body, glow, accent }, i) => (
            <FadeIn key={title} delay={i * 80}>
              <div
                className="relative h-full rounded-2xl p-7 overflow-hidden transition-all duration-300 group"
                style={{
                  background: 'rgba(255,255,255,0.02)',
                  border: '1px solid rgba(255,255,255,0.06)',
                }}
                onMouseEnter={(e) => {
                  const el = e.currentTarget as HTMLElement;
                  el.style.background = 'rgba(255,255,255,0.04)';
                  el.style.borderColor = 'rgba(255,255,255,0.1)';
                  el.style.transform = 'translateY(-2px)';
                }}
                onMouseLeave={(e) => {
                  const el = e.currentTarget as HTMLElement;
                  el.style.background = 'rgba(255,255,255,0.02)';
                  el.style.borderColor = 'rgba(255,255,255,0.06)';
                  el.style.transform = 'translateY(0)';
                }}
              >
                {/* Glow corner */}
                <div
                  className="absolute top-0 right-0 w-32 h-32 pointer-events-none rounded-bl-[100px]"
                  style={{ background: glow, filter: 'blur(24px)' }}
                />
                {/* Icon */}
                <div
                  className="w-11 h-11 rounded-xl flex items-center justify-center mb-5"
                  style={{ background: `${glow.replace('0.12', '0.15')}`, color: accent }}
                >
                  {icon}
                </div>
                {/* Tag */}
                <p className="text-[10px] font-semibold uppercase tracking-widest mb-2" style={{ color: accent }}>
                  {tag}
                </p>
                <h3 className="text-lg font-bold text-white mb-2">{title}</h3>
                <p className="text-sm leading-relaxed" style={{ color: '#6B6B78' }}>{body}</p>
              </div>
            </FadeIn>
          ))}
        </div>
      </div>
    </section>
  );
}
