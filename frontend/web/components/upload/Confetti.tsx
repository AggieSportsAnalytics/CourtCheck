'use client';

import { useEffect, useRef } from 'react';

/**
 * Canvas confetti tuned to the brand. 160 particles burst outward from center,
 * gravity pulls them down, then fade. Brand palette only — no red, no rainbow.
 * Light + dark variants. Disabled under prefers-reduced-motion.
 *
 * Plays once on mount. Mount this component inside the `done` state pane and
 * unmount it when the user resets — every mount fires a fresh burst.
 */
const PALETTE_LIGHT = ['#BFC846', '#E1E7A6', '#B05B36', '#D08866', '#2E5341', '#7B4E6E'];
const PALETTE_DARK = ['#DDE970', '#BFC846', '#E07A52', '#F0A07F', '#7B4E6E', '#6FA88B'];

const LIFE_MS = 2400;
const BURST_MS = 1700;
const PARTICLE_COUNT = 160;

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
  rot: number;
  vrot: number;
  color: string;
  shape: 'rect' | 'circle';
}

export default function Confetti() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (prefersReduced) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const isDark =
      document.documentElement.classList.contains('dark') ||
      document.body.classList.contains('dark');
    const colors = isDark ? PALETTE_DARK : PALETTE_LIGHT;

    const cx = rect.width / 2;
    const cy = rect.height / 2;
    const particles: Particle[] = [];
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const angle = Math.random() * Math.PI * 2;
      const speed = 220 + Math.random() * 340;
      particles.push({
        x: cx,
        y: cy,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed - 80,
        size: 3 + Math.random() * 6,
        rot: Math.random() * Math.PI * 2,
        vrot: (Math.random() - 0.5) * 10,
        color: colors[Math.floor(Math.random() * colors.length)],
        shape: Math.random() < 0.5 ? 'rect' : 'circle',
      });
    }

    const startTime = performance.now();
    const gravity = 520;
    const drag = 0.96;
    const dt = 1 / 60;
    let rafId: number | null = null;

    const step = (now: number) => {
      const elapsed = now - startTime;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      let alpha = 1;
      if (elapsed > BURST_MS) {
        alpha = Math.max(0, 1 - (elapsed - BURST_MS) / (LIFE_MS - BURST_MS));
      }

      for (const p of particles) {
        p.vx *= drag;
        p.vy = p.vy * drag + gravity * dt;
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        p.rot += p.vrot * dt;

        ctx.save();
        ctx.globalAlpha = alpha;
        ctx.translate(p.x, p.y);
        ctx.rotate(p.rot);
        ctx.fillStyle = p.color;
        if (p.shape === 'rect') {
          ctx.fillRect(-p.size / 2, -p.size / 2, p.size, p.size * 0.6);
        } else {
          ctx.beginPath();
          ctx.arc(0, 0, p.size / 2, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.restore();
      }

      if (elapsed < LIFE_MS) {
        rafId = requestAnimationFrame(step);
      } else {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        rafId = null;
      }
    };

    rafId = requestAnimationFrame(step);
    return () => {
      if (rafId !== null) cancelAnimationFrame(rafId);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className="pointer-events-none absolute inset-0 z-10 h-full w-full motion-reduce:hidden"
    />
  );
}
