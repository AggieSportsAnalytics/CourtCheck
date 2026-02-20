'use client';

import { FadeIn } from './FadeIn';

const PAIN_POINTS = [
  {
    icon: (
      <svg viewBox="0 0 24 24" fill="none" className="w-5 h-5">
        <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z" stroke="currentColor" strokeWidth="1.5" />
        <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="1.5" />
        <line x1="2" y1="2" x2="22" y2="22" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      </svg>
    ),
    title: 'Your patterns are invisible',
    body: 'You know roughly how a session felt. But without data, you have no idea where your ball actually lands, how your strokes split, or what tendencies you repeat.',
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" fill="none" className="w-5 h-5">
        <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        <circle cx="9" cy="7" r="4" stroke="currentColor" strokeWidth="1.5" />
        <path d="M23 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      </svg>
    ),
    title: 'Coaching is expensive and rare',
    body: "Quality coaching is $80–$150 an hour — and most players can only afford it occasionally. Between sessions, actionable feedback disappears completely.",
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" fill="none" className="w-5 h-5">
        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
    title: "You can't improve what you can't measure",
    body: "Progress stalls when you rely on memory and instinct alone. Every elite sport now uses data — tennis at the recreational and competitive club level still doesn't.",
  },
];

export function Problem() {
  return (
    <section className="py-28" style={{ background: '#07070A' }}>
      <div className="max-w-6xl mx-auto px-5 sm:px-8">
        {/* Header */}
        <FadeIn className="max-w-2xl mb-16">
          <p className="text-xs font-semibold uppercase tracking-widest mb-4" style={{ color: '#B4F000' }}>
            The Problem
          </p>
          <h2 className="text-4xl sm:text-5xl font-black tracking-tight text-white leading-tight mb-4">
            Tennis has always been played on instinct.
          </h2>
          <p className="text-lg leading-relaxed" style={{ color: '#8C8C99' }}>
            Every other major sport uses computer vision and analytics to train smarter. Tennis — at the club and collegiate level — is still flying blind.
          </p>
        </FadeIn>

        {/* Pain points */}
        <div className="grid md:grid-cols-3 gap-5">
          {PAIN_POINTS.map(({ icon, title, body }, i) => (
            <FadeIn key={title} delay={i * 100}>
              <div
                className="h-full rounded-2xl p-6 transition-all duration-300"
                style={{
                  background: 'rgba(255,255,255,0.02)',
                  border: '1px solid rgba(255,255,255,0.06)',
                }}
                onMouseEnter={(e) => {
                  (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.04)';
                  (e.currentTarget as HTMLElement).style.borderColor = 'rgba(255,255,255,0.1)';
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.02)';
                  (e.currentTarget as HTMLElement).style.borderColor = 'rgba(255,255,255,0.06)';
                }}
              >
                <div
                  className="w-10 h-10 rounded-xl flex items-center justify-center mb-5"
                  style={{ background: 'rgba(180,240,0,0.08)', color: '#B4F000' }}
                >
                  {icon}
                </div>
                <h3 className="text-base font-bold text-white mb-2">{title}</h3>
                <p className="text-sm leading-relaxed" style={{ color: '#6B6B78' }}>{body}</p>
              </div>
            </FadeIn>
          ))}
        </div>
      </div>
    </section>
  );
}
