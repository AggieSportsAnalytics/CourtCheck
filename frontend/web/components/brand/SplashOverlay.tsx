'use client'

import { useEffect, useRef, useState } from 'react'

type Props = {
  /**
   * sessionStorage key. Splash plays once per browser session per key.
   * Default 'ccLandingSplashSeen'. Dashboard uses 'ccDashSplashSeen'.
   */
  storageKey?: string
}

export default function SplashOverlay({ storageKey = 'ccLandingSplashSeen' }: Props) {
  const [phase, setPhase] = useState<'visible' | 'hiding' | 'hidden'>('hidden')
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const dismissedRef = useRef(false)

  useEffect(() => {
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    let seen = false
    try {
      seen = sessionStorage.getItem(storageKey) === '1'
    } catch {}
    if (reduce || seen) {
      // Defensive: in case a prior render left the lock class on body.
      document.body.classList.remove('splash-locked')
      return
    }

    try {
      sessionStorage.setItem(storageKey, '1')
    } catch {}

    setPhase('visible')
    document.body.classList.add('splash-locked')

    const v = videoRef.current
    if (v) {
      v.play().catch(() => {})
    }

    // Always release the body lock on unmount (covers HMR + route changes
    // that fire before the dismiss timer.)
    return () => {
      document.body.classList.remove('splash-locked')
    }
  }, [storageKey])

  useEffect(() => {
    if (phase !== 'visible') return

    function dismiss() {
      if (dismissedRef.current) return
      dismissedRef.current = true
      setPhase('hiding')
      setTimeout(() => {
        setPhase('hidden')
        document.body.classList.remove('splash-locked')
      }, 640)
    }

    const v = videoRef.current
    v?.addEventListener('ended', dismiss)
    const t1 = setTimeout(dismiss, 4500)
    const t2 = setTimeout(() => {
      if (!dismissedRef.current) {
        setPhase('hidden')
        document.body.classList.remove('splash-locked')
        dismissedRef.current = true
      }
    }, 6000)

    return () => {
      v?.removeEventListener('ended', dismiss)
      clearTimeout(t1)
      clearTimeout(t2)
    }
  }, [phase])

  if (phase === 'hidden') return null

  return (
    <div
      role="presentation"
      aria-hidden="true"
      onClick={() => {
        if (dismissedRef.current) return
        dismissedRef.current = true
        setPhase('hiding')
        setTimeout(() => {
          setPhase('hidden')
          document.body.classList.remove('splash-locked')
        }, 640)
      }}
      className="fixed inset-0 z-[9999] flex items-center justify-center bg-cream cursor-pointer"
      style={{
        opacity: phase === 'hiding' ? 0 : 1,
        transition: 'opacity 620ms cubic-bezier(0.4, 0, 0.2, 1)',
      }}
    >
      <video
        ref={videoRef}
        muted
        playsInline
        autoPlay
        preload="auto"
        className="block object-contain dark:invert dark:hue-rotate-180"
        style={{
          width: 'min(72vw, 760px)',
          height: 'auto',
          maxHeight: '78vh',
          background: 'transparent',
        }}
      >
        <source src="/CourtCheckAnimation.webm" type="video/webm" />
        <source src="/CourtCheckAnimation.mp4" type="video/mp4" />
      </video>
    </div>
  )
}
