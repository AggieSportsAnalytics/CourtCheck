'use client';

import { FadeIn } from './FadeIn';

const STEPS = [
  {
    number: '01',
    title: 'Record & Upload',
    body: 'Film your match from behind the court using any camera or phone. Upload the video directly to CourtCheck — we accept any standard MP4 file.',
    detail: 'Standard tripod setup · Any camera · MP4 format',
  },
  {
    number: '02',
    title: 'AI Analyzes Everything',
    body: 'Our GPU-accelerated pipeline runs ball detection, court mapping, player tracking, and stroke classification across every frame of your footage.',
    detail: 'YOLO · CatBoost · Temporal CNN · Homography mapping',
  },
  {
    number: '03',
    title: 'Get Your Insights',
    body: 'Receive heatmaps, stroke breakdowns, shot quality metrics, and a personalized AI scouting report — ready minutes after upload.',
    detail: 'Court heatmaps · Stroke charts · AI scouting report',
  },
];

export function HowItWorks() {
  return (
    <section id="how-it-works" className="py-28" style={{ background: '#07070A' }}>
      <div className="max-w-6xl mx-auto px-5 sm:px-8">
        <div className="w-full h-px mb-28" style={{ background: 'rgba(255,255,255,0.06)' }} />

        <FadeIn className="max-w-2xl mb-20">
          <p className="text-xs font-semibold uppercase tracking-widest mb-4" style={{ color: '#B4F000' }}>
            How It Works
          </p>
          <h2 className="text-4xl sm:text-5xl font-black tracking-tight text-white leading-tight mb-4">
            From raw footage to insights in minutes.
          </h2>
          <p className="text-lg leading-relaxed" style={{ color: '#8C8C99' }}>
            No sensors, no special equipment, no setup. Just upload your video.
          </p>
        </FadeIn>

        {/* Steps */}
        <div className="grid md:grid-cols-3 gap-8 relative">
          {/* Connecting line (desktop) */}
          <div
            className="hidden md:block absolute top-8 left-[16%] right-[16%] h-px pointer-events-none"
            style={{ background: 'linear-gradient(90deg, transparent, rgba(180,240,0,0.2), rgba(180,240,0,0.2), transparent)' }}
          />

          {STEPS.map(({ number, title, body, detail }, i) => (
            <FadeIn key={number} delay={i * 120}>
              <div className="relative flex flex-col">
                {/* Step number */}
                <div
                  className="w-16 h-16 rounded-2xl flex items-center justify-center mb-6 relative z-10"
                  style={{
                    background: 'rgba(180,240,0,0.06)',
                    border: '1px solid rgba(180,240,0,0.2)',
                  }}
                >
                  <span
                    className="text-xl font-black"
                    style={{
                      background: 'linear-gradient(135deg, #B4F000, #88CC00)',
                      WebkitBackgroundClip: 'text',
                      WebkitTextFillColor: 'transparent',
                      backgroundClip: 'text',
                    }}
                  >
                    {number}
                  </span>
                </div>

                <h3 className="text-lg font-bold text-white mb-3">{title}</h3>
                <p className="text-sm leading-relaxed mb-4" style={{ color: '#6B6B78' }}>{body}</p>

                {/* Tech detail chip */}
                <div
                  className="inline-flex mt-auto rounded-lg px-3 py-1.5 text-[11px] font-mono"
                  style={{
                    background: 'rgba(255,255,255,0.03)',
                    border: '1px solid rgba(255,255,255,0.07)',
                    color: '#5A5A66',
                  }}
                >
                  {detail}
                </div>
              </div>
            </FadeIn>
          ))}
        </div>
      </div>
    </section>
  );
}
