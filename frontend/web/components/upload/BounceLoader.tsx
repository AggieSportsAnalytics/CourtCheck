'use client';

/**
 * Brand-locked waiting signal. Plays /Bounce_Animated.webm with transparent alpha.
 * In dark mode the WebM is inverted+hue-rotated so the ink-on-cream artwork
 * becomes cream-on-ink. Reduced-motion users see the static poster only.
 *
 * Size sets the WIDTH only — the container hugs the video's natural aspect so
 * there's no whitespace above/below the artwork. (The bounce loop is wider
 * than tall; a 1:1 square left huge top/bottom dead-air.)
 */
export default function BounceLoader({ size = 380 }: { size?: number }) {
  return (
    <div
      className="mx-auto flex items-center justify-center"
      style={{ width: size, maxWidth: '90%' }}
      aria-hidden="true"
    >
      <video
        className="block h-auto w-full motion-reduce:hidden dark:invert dark:hue-rotate-180"
        autoPlay
        muted
        loop
        playsInline
        preload="auto"
        poster="/Bounce_Still.png"
      >
        <source src="/Bounce_Animated.webm" type="video/webm" />
        <source src="/Bounce_Animated.mp4" type="video/mp4" />
      </video>
      {/* Reduced-motion fallback */}
      <img
        src="/Bounce_Still.png"
        alt=""
        className="hidden h-auto w-full motion-reduce:block dark:invert dark:hue-rotate-180"
      />
    </div>
  );
}
