'use client';

import { useEffect, useState } from 'react';
import { getDemoFlag, setDemoFlag } from '@/lib/demo/demoData';

/**
 * Small persistent demo-mode switch. Bottom-left, low-key so it stays out of
 * the way during a screen recording (crop it or it sits in the corner). Flips
 * localStorage and reloads so the dashboard / recordings re-run their mount
 * load and swap in the fabricated season of data.
 *
 * Mounted inside AppLayout, so it only renders for authenticated users (demo
 * data is gated behind auth anyway).
 */
export default function DemoToggle() {
  const [on, setOn] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    // Sync a `?demo=1` visit into the persistent flag so it survives
    // navigation, then reflect current state.
    const params = new URLSearchParams(window.location.search);
    const param = params.get('demo');
    if ((param === '1' || param === 'true') && !getDemoFlag()) {
      setDemoFlag(true);
    }
    setOn(getDemoFlag());
    setMounted(true);
  }, []);

  if (!mounted) return null;

  const toggle = () => {
    const next = !on;
    setDemoFlag(next);
    setOn(next);
    // Hard reload: dashboard/recordings decide demo vs live at mount.
    window.location.reload();
  };

  return (
    <button
      type="button"
      role="switch"
      aria-checked={on}
      aria-label="Toggle demo mode"
      onClick={toggle}
      title={on ? 'Demo mode ON — click to show live data' : 'Demo mode OFF — click for demo data'}
      style={{
        position: 'fixed',
        left: 14,
        bottom: 14,
        zIndex: 90,
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        padding: '6px 10px 6px 8px',
        borderRadius: 999,
        border: '1px solid var(--color-line)',
        background: 'var(--color-paper)',
        color: 'var(--color-ink-soft)',
        font: '500 0.7rem/1 var(--font-mono, ui-monospace, monospace)',
        letterSpacing: '0.08em',
        textTransform: 'uppercase',
        boxShadow: 'var(--shadow-card)',
        opacity: on ? 0.92 : 0.45,
        transition: 'opacity 160ms ease, background 160ms ease',
        cursor: 'pointer',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.opacity = '1';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.opacity = on ? '0.92' : '0.45';
      }}
    >
      <span
        aria-hidden
        style={{
          width: 30,
          height: 16,
          borderRadius: 999,
          background: on ? 'var(--color-court)' : 'var(--color-line)',
          position: 'relative',
          flexShrink: 0,
          transition: 'background 160ms ease',
        }}
      >
        <span
          style={{
            position: 'absolute',
            top: 2,
            left: on ? 16 : 2,
            width: 12,
            height: 12,
            borderRadius: '50%',
            background: 'var(--color-cream)',
            transition: 'left 160ms ease',
          }}
        />
      </span>
      Demo
    </button>
  );
}
