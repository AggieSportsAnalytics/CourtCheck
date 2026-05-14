import * as React from 'react'

import { cn } from '@/lib/utils'

function Eyebrow({
  className,
  showDot = true,
  tone = 'court',
  ...props
}: React.ComponentProps<'span'> & {
  showDot?: boolean
  tone?: 'court' | 'clay' | 'ink' | 'mute'
}) {
  const toneClass = {
    court: 'text-court',
    clay: 'text-clay',
    ink: 'text-ink',
    mute: 'text-ink-mute',
  }[tone]

  return (
    <span
      data-slot="eyebrow"
      className={cn(
        'inline-flex items-center gap-2',
        'font-mono uppercase',
        'text-[0.72rem] tracking-[0.18em]',
        toneClass,
        showDot &&
          "before:content-[''] before:size-[6px] before:rounded-full before:bg-clay",
        className,
      )}
      {...props}
    />
  )
}

export { Eyebrow }
