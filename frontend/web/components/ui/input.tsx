import * as React from 'react'

import { cn } from '@/lib/utils'

function Input({ className, type, ...props }: React.ComponentProps<'input'>) {
  return (
    <input
      type={type}
      data-slot="input"
      className={cn(
        'w-full min-w-0 h-11 px-4 py-2',
        'bg-paper text-ink',
        'border border-line rounded-[12px]',
        'font-sans text-[0.95rem] leading-[1.5]',
        'placeholder:text-ink-mute',
        'transition-[border-color,box-shadow,background-color] duration-[160ms] ease-[cubic-bezier(0.2,0.8,0.2,1)]',
        'outline-none',
        'hover:border-ink-mute',
        'focus-visible:border-court focus-visible:ring-2 focus-visible:ring-court/20',
        'aria-invalid:border-clay aria-invalid:ring-2 aria-invalid:ring-clay/20',
        'disabled:pointer-events-none disabled:cursor-not-allowed disabled:opacity-50',
        'file:inline-flex file:h-7 file:border-0 file:bg-transparent file:text-sm file:font-medium file:text-ink',
        className,
      )}
      {...props}
    />
  )
}

export { Input }
