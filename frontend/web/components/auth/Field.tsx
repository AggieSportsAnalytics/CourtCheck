import * as React from 'react'

import { cn } from '@/lib/utils'

function Field({
  children,
  className,
}: {
  children: React.ReactNode
  className?: string
}) {
  return <div className={cn('grid gap-1.5', className)}>{children}</div>
}

function FieldLabel({
  children,
  htmlFor,
  className,
}: {
  children: React.ReactNode
  htmlFor?: string
  className?: string
}) {
  return (
    <label
      htmlFor={htmlFor}
      className={cn(
        'font-mono uppercase text-[0.66rem] tracking-[0.14em] text-ink-mute',
        className,
      )}
    >
      {children}
    </label>
  )
}

function FieldControl({
  children,
  className,
}: {
  children: React.ReactNode
  className?: string
}) {
  return (
    <div
      className={cn(
        '[&>input]:w-full [&>select]:w-full',
        '[&>input]:px-3.5 [&>input]:py-3 [&>select]:px-3.5 [&>select]:py-3',
        '[&>input]:bg-shade [&>select]:bg-shade dark:[&>input]:bg-surface dark:[&>select]:bg-surface',
        '[&>input]:border [&>input]:border-transparent [&>select]:border [&>select]:border-transparent',
        '[&>input]:rounded-[10px] [&>select]:rounded-[10px]',
        '[&>input]:font-sans [&>select]:font-sans',
        '[&>input]:text-[0.95rem] [&>select]:text-[0.95rem]',
        '[&>input]:text-ink [&>select]:text-ink',
        '[&>input]:placeholder:text-ink-mute',
        '[&>input]:transition-[border-color,background-color] [&>select]:transition-[border-color,background-color]',
        '[&>input]:duration-[160ms] [&>select]:duration-[160ms]',
        '[&>input]:ease-[cubic-bezier(0.2,0.8,0.2,1)] [&>select]:ease-[cubic-bezier(0.2,0.8,0.2,1)]',
        '[&>input]:outline-none [&>select]:outline-none',
        '[&>input:focus]:border-ink [&>input:focus]:bg-paper',
        '[&>select:focus]:border-ink [&>select:focus]:bg-paper',
        'dark:[&>input:focus]:bg-surface dark:[&>select:focus]:bg-surface',
        'aria-[invalid=true]:[&>input]:border-clay',
        className,
      )}
    >
      {children}
    </div>
  )
}

function FieldError({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-clay text-[0.82rem] leading-[1.4] mt-1 inline-flex items-center gap-1.5">
      <svg
        viewBox="0 0 24 24"
        width="14"
        height="14"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.75"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden
      >
        <circle cx="12" cy="12" r="10" />
        <line x1="12" y1="8" x2="12" y2="12" />
        <line x1="12" y1="16" x2="12.01" y2="16" />
      </svg>
      <span>{children}</span>
    </p>
  )
}

function FieldRow({
  children,
  className,
}: {
  children: React.ReactNode
  className?: string
}) {
  return (
    <div
      className={cn(
        'grid grid-cols-1 sm:grid-cols-2 gap-3.5',
        className,
      )}
    >
      {children}
    </div>
  )
}

export { Field, FieldLabel, FieldControl, FieldError, FieldRow }
