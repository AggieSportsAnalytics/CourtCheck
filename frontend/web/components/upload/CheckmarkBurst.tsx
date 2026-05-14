'use client';

import Confetti from './Confetti';

/**
 * Done-state hero: 340px square with the locked CourtCheck checkmark WebM
 * playing once over a confetti burst. The video carries the ball+checkmark
 * portion of the brand animation. No loop — plays once and rests.
 *
 * The scoped <style> block defines `cc-logo-pop` (a spring scale-in) without
 * touching globals.css.
 */
export default function CheckmarkBurst({ size = 240 }: { size?: number }) {
  return (
    <div
      className="relative mx-auto flex items-center justify-center"
      style={{ width: size, maxWidth: '90%' }}
    >
      <style>{`
        @keyframes cc-logo-pop {
          0%   { transform: scale(0.92); opacity: 0; }
          60%  { transform: scale(1.04); opacity: 1; }
          100% { transform: scale(1.00); opacity: 1; }
        }
        .cc-logo-pop { animation: cc-logo-pop 460ms var(--ease-spring) both; }
        @media (prefers-reduced-motion: reduce) {
          .cc-logo-pop { animation: none; }
        }
      `}</style>
      <Confetti />
      <div className="cc-logo-pop relative z-20 flex w-full items-center justify-center">
        <video
          className="block h-auto w-full object-contain dark:invert dark:hue-rotate-180"
          autoPlay
          muted
          playsInline
          preload="auto"
          aria-hidden="true"
          style={{ filter: 'drop-shadow(0 8px 20px rgba(191, 200, 70, 0.18))' }}
        >
          <source src="/CourtCheckCheckmark.webm" type="video/webm" />
        </video>
      </div>
    </div>
  );
}
