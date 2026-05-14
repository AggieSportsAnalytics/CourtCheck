'use client'

import { cn } from '@/lib/utils'

function GoogleButton({
  label,
  onClick,
  disabled,
  className,
}: {
  label: string
  onClick?: () => void
  disabled?: boolean
  className?: string
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={cn(
        'w-full inline-flex items-center justify-center gap-3',
        'px-5 py-3 rounded-full',
        'bg-surface text-ink border border-line',
        'font-sans text-[0.95rem] font-medium',
        'transition-[border-color,background-color] duration-[160ms] ease-[cubic-bezier(0.2,0.8,0.2,1)]',
        'hover:border-ink',
        'disabled:pointer-events-none disabled:opacity-50',
        'cursor-pointer',
        className,
      )}
    >
      <span className="inline-flex size-[18px] items-center justify-center">
        <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden>
          <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
          <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
          <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" />
          <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
        </svg>
      </span>
      {label}
    </button>
  )
}

function OrDivider() {
  return (
    <div
      className="flex items-center gap-3.5 my-3 mb-5 font-mono uppercase text-[0.7rem] tracking-[0.18em] text-ink-mute before:content-[''] before:flex-1 before:h-px before:bg-line after:content-[''] after:flex-1 after:h-px after:bg-line"
    >
      or
    </div>
  )
}

export { GoogleButton, OrDivider }
