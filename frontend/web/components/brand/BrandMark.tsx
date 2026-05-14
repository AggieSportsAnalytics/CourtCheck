'use client'

import { useRef } from 'react'
import Link from 'next/link'

import { cn } from '@/lib/utils'

type Size = 'sm' | 'md' | 'lg'

// Natural aspect ratio of the wordmark PNG is ~1.87:1 (613x327).
// Mock uses height-based sizing: 52px in nav, 40px in footer, up to 64px in sidebar.
const SIZE_TOKENS: Record<Size, { height: number }> = {
  sm: { height: 40 },
  md: { height: 52 },
  lg: { height: 64 },
}

function BrandMark({
  href = '/',
  size = 'md',
  className,
  withHoverVideo = true,
  heightPx,
}: {
  href?: string | null
  size?: Size
  className?: string
  withHoverVideo?: boolean
  /** Override height in pixels. */
  heightPx?: number
}) {
  const height = heightPx ?? SIZE_TOKENS[size].height
  const videoRef = useRef<HTMLVideoElement | null>(null)

  const handleEnter = () => {
    const v = videoRef.current
    if (!v) return
    v.currentTime = 0
    v.play().catch(() => {})
  }
  const handleLeave = () => {
    const v = videoRef.current
    if (!v) return
    v.pause()
    v.currentTime = 0
  }

  const content = (
    <span
      className={cn('brand-mark relative inline-flex items-center', className)}
      onMouseEnter={withHoverVideo ? handleEnter : undefined}
      onMouseLeave={withHoverVideo ? handleLeave : undefined}
      style={{ height }}
    >
      <img
        src="/CourtCheckLogoLight.png"
        alt="CourtCheck"
        className="bm-img block w-auto dark:hidden"
        style={{ height: '100%' }}
      />
      <img
        src="/CourtCheckLogoDark.png"
        alt="CourtCheck"
        className="bm-img hidden w-auto dark:block"
        style={{ height: '100%' }}
      />
      {withHoverVideo && (
        <video
          ref={videoRef}
          className="bm-video absolute inset-0 w-auto dark:invert dark:hue-rotate-180"
          style={{ height: '100%' }}
          muted
          playsInline
          preload="none"
          aria-hidden="true"
        >
          <source src="/CourtCheckAnimation.webm" type="video/webm" />
          <source src="/CourtCheckAnimation.mp4" type="video/mp4" />
        </video>
      )}
    </span>
  )

  if (href === null) return content

  return (
    <Link href={href} aria-label="CourtCheck home" className="inline-flex">
      {content}
    </Link>
  )
}

export { BrandMark }
