import { ReactNode } from 'react'

import { cn } from '@/lib/utils'

function AuthCard({
  children,
  className,
}: {
  children: ReactNode
  className?: string
}) {
  return (
    <div
      className={cn(
        'w-full bg-paper text-ink',
        'border border-line rounded-[var(--radius-lg)]',
        'p-8',
        'shadow-[var(--shadow-card)]',
        className,
      )}
    >
      {children}
    </div>
  )
}

function AuthEyebrow({ children }: { children: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-2 font-mono uppercase text-[0.7rem] tracking-[0.18em] text-court before:content-[''] before:size-[6px] before:rounded-full before:bg-clay">
      {children}
    </span>
  )
}

function AuthTitle({
  children,
  className,
}: {
  children: ReactNode
  className?: string
}) {
  return (
    <h1
      className={cn(
        'font-display font-medium leading-[1.05] tracking-[-0.022em]',
        'text-[clamp(32px,4vw,44px)]',
        'mt-3.5 mb-2',
        className,
      )}
      style={{ fontVariationSettings: '"opsz" 72' }}
    >
      {children}
    </h1>
  )
}

function AuthSub({
  children,
  className,
}: {
  children: ReactNode
  className?: string
}) {
  return (
    <p
      className={cn(
        'text-ink-soft text-[0.98rem] leading-[1.5] mb-7',
        className,
      )}
    >
      {children}
    </p>
  )
}

function AuthFoot({ children }: { children: ReactNode }) {
  return (
    <div className="text-center text-[0.92rem] text-ink-soft pt-4 mt-5 border-t border-line-soft">
      {children}
    </div>
  )
}

function AuthInlineLink({
  children,
  ...props
}: React.ComponentProps<'a'>) {
  return (
    <a
      {...props}
      className="text-court font-medium border-b border-current dark:text-court-light"
    >
      {children}
    </a>
  )
}

export { AuthCard, AuthEyebrow, AuthTitle, AuthSub, AuthFoot, AuthInlineLink }
