'use client';

import Link from 'next/link';
import { FadeIn } from './FadeIn';

export function CTA() {
  return (
    <section className="py-28" style={{ background: '#07070A' }}>
      <div className="max-w-6xl mx-auto px-5 sm:px-8">
        <FadeIn>
          <div
            className="relative rounded-3xl overflow-hidden px-8 py-20 text-center"
            style={{
              background: 'radial-gradient(ellipse 120% 100% at 50% 100%, rgba(180,240,0,0.1) 0%, transparent 65%), rgba(255,255,255,0.02)',
              border: '1px solid rgba(180,240,0,0.15)',
            }}
          >
            {/* Corner glows */}
            <div
              className="absolute top-0 left-0 w-64 h-64 pointer-events-none"
              style={{ background: 'radial-gradient(circle, rgba(180,240,0,0.06) 0%, transparent 70%)', filter: 'blur(30px)' }}
            />
            <div
              className="absolute bottom-0 right-0 w-64 h-64 pointer-events-none"
              style={{ background: 'radial-gradient(circle, rgba(180,240,0,0.06) 0%, transparent 70%)', filter: 'blur(30px)' }}
            />

            <div className="relative max-w-2xl mx-auto">
              <p className="text-xs font-semibold uppercase tracking-widest mb-6" style={{ color: '#B4F000' }}>
                Ready to level up?
              </p>
              <h2 className="text-4xl sm:text-5xl lg:text-6xl font-black tracking-tight text-white leading-tight mb-6">
                Stop guessing.
                <br />
                <span
                  style={{
                    background: 'linear-gradient(135deg, #B4F000 0%, #88CC00 50%, #B4F000 100%)',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                    backgroundClip: 'text',
                  }}
                >
                  Start winning.
                </span>
              </h2>
              <p className="text-lg mb-10 leading-relaxed" style={{ color: '#8C8C99' }}>
                Upload your first match and get a full AI analysis — heatmaps, stroke breakdown, and a personalised scouting report in minutes.
              </p>
              <div className="flex flex-wrap items-center justify-center gap-4">
                <Link
                  href="/auth/login"
                  className="inline-flex items-center gap-2 px-8 py-3.5 rounded-xl text-sm font-bold transition-all duration-200"
                  style={{ background: '#B4F000', color: '#07070A' }}
                  onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = '#C7FF00'; (e.currentTarget as HTMLElement).style.transform = 'translateY(-2px)'; (e.currentTarget as HTMLElement).style.boxShadow = '0 8px 30px rgba(180,240,0,0.3)'; }}
                  onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = '#B4F000'; (e.currentTarget as HTMLElement).style.transform = 'translateY(0)'; (e.currentTarget as HTMLElement).style.boxShadow = 'none'; }}
                >
                  Analyze Your First Match
                  <svg viewBox="0 0 16 16" fill="none" className="w-4 h-4">
                    <path d="M3 8h10M9 4l4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </Link>
                <Link
                  href="/auth/login"
                  className="inline-flex items-center gap-2 px-8 py-3.5 rounded-xl text-sm font-semibold transition-all duration-200"
                  style={{
                    background: 'rgba(255,255,255,0.05)',
                    border: '1px solid rgba(255,255,255,0.1)',
                    color: '#FAFAFA',
                  }}
                  onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.08)'; }}
                  onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.05)'; }}
                >
                  Sign In
                </Link>
              </div>
            </div>
          </div>
        </FadeIn>
      </div>
    </section>
  );
}
