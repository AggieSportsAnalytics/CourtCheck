import { ReactNode } from 'react'

import { BrandMark } from '@/components/brand/BrandMark'
import { ThemeToggle } from '@/components/brand/ThemeToggle'

function AuthShell({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="px-7 sm:px-14 py-7 flex items-center justify-between">
        <BrandMark />
        <ThemeToggle />
      </header>

      <main className="flex-1 flex flex-col items-center justify-center px-7 pb-16 pt-2">
        <div className="w-full max-w-[460px] motion-safe:animate-[rise_480ms_cubic-bezier(0.2,0.8,0.2,1)]">
          {children}
        </div>
      </main>

      <footer className="px-7 sm:px-14 py-7 flex flex-col sm:flex-row gap-1.5 sm:gap-0 sm:justify-between font-mono uppercase text-[0.7rem] tracking-[0.14em] text-ink-mute">
        <span>© 2026 CourtCheck</span>
        <span>v0.1</span>
      </footer>

      <style>{`
        @keyframes rise {
          from { opacity: 0; transform: translateY(12px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  )
}

export { AuthShell }
