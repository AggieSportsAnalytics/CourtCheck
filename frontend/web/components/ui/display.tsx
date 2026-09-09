import * as React from 'react'

import { cn } from '@/lib/utils'

type DisplaySize = 'hero' | 'xl' | 'lg' | 'md' | 'sm'

const sizeClasses: Record<DisplaySize, string> = {
  hero: 'text-[clamp(56px,9vw,142px)] leading-[0.98] tracking-[-0.025em]',
  xl: 'text-[clamp(48px,7.5vw,108px)] leading-[1.0] tracking-[-0.025em]',
  lg: 'text-[clamp(40px,5.6vw,76px)] leading-[1.0] tracking-[-0.022em]',
  md: 'text-[clamp(32px,3.8vw,52px)] leading-[1.15] tracking-[-0.014em]',
  sm: 'text-[2rem] leading-[1.15] tracking-[-0.012em]',
}

const opszMap: Record<DisplaySize, number> = {
  hero: 144,
  xl: 144,
  lg: 96,
  md: 72,
  sm: 60,
}

function Display({
  className,
  size = 'lg',
  as: As = 'h2',
  style,
  ...props
}: React.HTMLAttributes<HTMLElement> & {
  size?: DisplaySize
  as?: React.ElementType
}) {
  return (
    <As
      data-slot="display"
      className={cn('font-display font-medium', sizeClasses[size], className)}
      style={{
        fontVariationSettings: `"opsz" ${opszMap[size]}`,
        ...style,
      }}
      {...props}
    />
  )
}

function Num({
  className,
  size = 'md',
  style,
  ...props
}: React.ComponentProps<'span'> & {
  size?: 'sm' | 'md' | 'lg' | 'xl'
}) {
  const sizeClass = {
    sm: 'text-[1.05rem]',
    md: 'text-[1.6rem] tracking-[-0.012em]',
    lg: 'text-[2.6rem] tracking-[-0.018em] leading-[1]',
    xl: 'text-[clamp(40px,5vw,72px)] tracking-[-0.022em] leading-[1]',
  }[size]

  return (
    <span
      data-slot="num"
      data-num=""
      className={cn(
        'font-display font-medium',
        '[font-feature-settings:"tnum"]',
        sizeClass,
        className,
      )}
      style={{
        fontVariationSettings: '"opsz" 72',
        ...style,
      }}
      {...props}
    />
  )
}

function Prose({ className, ...props }: React.ComponentProps<'p'>) {
  return (
    <p
      data-slot="prose"
      className={cn('font-display text-[1.12rem] leading-[1.6] text-ink', className)}
      style={{ fontVariationSettings: '"opsz" 16' }}
      {...props}
    />
  )
}

export { Display, Num, Prose }
