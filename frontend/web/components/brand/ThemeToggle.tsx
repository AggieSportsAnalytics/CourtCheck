'use client'

import { useEffect, useState } from 'react'
import { Moon, Sun } from 'lucide-react'

import { cn } from '@/lib/utils'

function ThemeToggle({ className }: { className?: string }) {
  const [dark, setDark] = useState(false)

  useEffect(() => {
    const stored = localStorage.getItem('cc-theme')
    const initial = stored === 'dark'
    setDark(initial)
    document.documentElement.classList.toggle('dark', initial)
  }, [])

  function toggle() {
    const next = !dark
    setDark(next)
    document.documentElement.classList.toggle('dark', next)
    localStorage.setItem('cc-theme', next ? 'dark' : 'light')
  }

  return (
    <button
      type="button"
      onClick={toggle}
      aria-label="Toggle theme"
      className={cn(
        'size-9 rounded-full',
        'bg-paper text-ink border border-line',
        'inline-flex items-center justify-center',
        'transition-[background-color,border-color,transform] duration-[240ms] ease-[cubic-bezier(0.2,0.8,0.2,1)]',
        'hover:border-ink hover:rotate-[20deg]',
        'cursor-pointer',
        className,
      )}
    >
      {dark ? <Sun className="size-4" /> : <Moon className="size-4" />}
    </button>
  )
}

export { ThemeToggle }
