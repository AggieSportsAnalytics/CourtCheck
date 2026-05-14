import * as React from 'react'
import { Slot } from '@radix-ui/react-slot'
import { cva, type VariantProps } from 'class-variance-authority'

import { cn } from '@/lib/utils'

const buttonVariants = cva(
  [
    'inline-flex items-center justify-center gap-2 whitespace-nowrap',
    'font-sans font-medium tracking-tight',
    'rounded-full',
    'transition-[transform,background-color,border-color,color]',
    'duration-[160ms] ease-[cubic-bezier(0.34,1.56,0.64,1)]',
    'disabled:pointer-events-none disabled:opacity-50',
    "[&_svg]:pointer-events-none [&_svg:not([class*='size-'])]:size-[18px] shrink-0 [&_svg]:shrink-0",
    'outline-none',
    'focus-visible:ring-2 focus-visible:ring-ring/40 focus-visible:ring-offset-2 focus-visible:ring-offset-cream',
    'dark:focus-visible:ring-offset-cream',
  ].join(' '),
  {
    variants: {
      variant: {
        primary:
          'bg-court text-cream hover:-translate-y-px hover:bg-court-deep dark:bg-court-deep dark:hover:bg-court',
        ghost:
          'bg-transparent text-ink border border-line hover:border-ink',
        accent:
          'bg-clay text-cream hover:-translate-y-px hover:bg-clay-soft',
        ink: 'bg-ink text-cream hover:-translate-y-px hover:opacity-95 dark:bg-court-deep',
        link: 'text-court underline-offset-4 hover:underline rounded-none px-0 py-0 h-auto',
        subtle:
          'bg-shade text-ink hover:bg-line-soft dark:bg-surface dark:hover:bg-line',
      },
      size: {
        default: 'h-11 px-[26px] text-[1rem]',
        sm: 'h-9 px-[18px] text-[0.92rem]',
        lg: 'h-12 px-8 text-[1.05rem]',
        icon: 'size-10',
        'icon-sm': 'size-9',
        'icon-lg': 'size-11',
      },
    },
    defaultVariants: {
      variant: 'primary',
      size: 'default',
    },
  },
)

function Button({
  className,
  variant,
  size,
  asChild = false,
  ...props
}: React.ComponentProps<'button'> &
  VariantProps<typeof buttonVariants> & {
    asChild?: boolean
  }) {
  const Comp = asChild ? Slot : 'button'

  return (
    <Comp
      data-slot="button"
      className={cn(buttonVariants({ variant, size, className }))}
      {...props}
    />
  )
}

export { Button, buttonVariants }
