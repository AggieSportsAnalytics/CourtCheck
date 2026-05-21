'use client'

import { useEffect, useState } from 'react'

import { BrandMark } from '@/components/brand/BrandMark'
import { ThemeToggle } from '@/components/brand/ThemeToggle'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

type TemplatePlayer = {
  name: string
  position: string | null
  year: string | null
  photo_url: string | null
  handedness: string | null
}

type Template = {
  key: string
  name: string
  description: string
  players: TemplatePlayer[]
}

export default function OnboardingPage() {
  const [templates, setTemplates] = useState<Template[]>([])
  const [picking, setPicking] = useState<string | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    fetch('/api/onboarding', { cache: 'no-store' })
      .then((r) => r.json())
      .then((d) => setTemplates(d.templates ?? []))
      .catch(() => setError("Couldn't load templates."))
  }, [])

  async function pick(template: string | null) {
    setError('')
    setPicking(template ?? 'empty')
    try {
      const res = await fetch('/api/onboarding', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ template }),
      })
      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.error || 'Failed to save selection')
      }
      // Full reload so middleware reads the updated user_metadata.onboarded
      // on the next request rather than the stale session it currently has.
      window.location.assign('/')
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Something went wrong.'
      setError(message)
      setPicking(null)
    }
  }

  const ucDavis = templates.find((t) => t.key === 'uc-davis')

  return (
    <div className="min-h-screen flex flex-col">
      <header className="px-7 sm:px-14 py-7 flex items-center justify-between">
        <BrandMark />
        <ThemeToggle />
      </header>

      <main className="flex-1 flex flex-col items-center justify-center px-7 pb-16 pt-2">
        <div className="w-full max-w-[760px] motion-safe:animate-[rise_480ms_cubic-bezier(0.2,0.8,0.2,1)]">
          <div className="flex justify-center mb-5">
            <BrandMark size="md" href={null} />
          </div>

          <h1
            className="font-display font-medium text-[1.65rem] tracking-[-0.014em] text-center"
            style={{ fontVariationSettings: '"opsz" 72' }}
          >
            Pick your <em>starting roster.</em>
          </h1>
          <p className="text-ink-soft text-[0.95rem] text-center mt-2 mb-7">
            You can edit, delete, and add players any time after.
          </p>

          <div className="grid gap-4 md:grid-cols-2 mb-4">
            <RosterCard
              title="UC Davis Tennis"
              subtitle={ucDavis ? `${ucDavis.players.length} players` : 'Loading...'}
              body="Start with the UC Davis women's tennis roster. Every player becomes yours to rename, edit, or remove."
              preview={ucDavis?.players.slice(0, 6).map((p) => p.name) ?? []}
              cta="Use UC Davis"
              busy={picking === 'uc-davis'}
              disabled={!ucDavis || !!picking}
              onClick={() => pick('uc-davis')}
            />
            <RosterCard
              title="Empty roster"
              subtitle="Start from scratch"
              body="Build your own roster from zero. Add players one at a time as you upload matches."
              preview={[]}
              cta="Start empty"
              busy={picking === 'empty'}
              disabled={!!picking}
              onClick={() => pick(null)}
            />
          </div>

          {error && (
            <div
              role="alert"
              className="rounded-[10px] border border-clay bg-[color-mix(in_srgb,var(--color-clay)_8%,transparent)] text-clay text-[0.88rem] leading-[1.45] px-3.5 py-2.5"
            >
              {error}
            </div>
          )}
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

function RosterCard({
  title,
  subtitle,
  body,
  preview,
  cta,
  busy,
  disabled,
  onClick,
}: {
  title: string
  subtitle: string
  body: string
  preview: string[]
  cta: string
  busy: boolean
  disabled: boolean
  onClick: () => void
}) {
  return (
    <div
      className={cn(
        'rounded-[14px] border border-line bg-paper p-5',
        'flex flex-col gap-3 shadow-[var(--shadow-card)]',
      )}
    >
      <div>
        <div className="font-display font-medium text-[1.18rem] tracking-[-0.012em]">{title}</div>
        <div className="text-ink-mute text-[0.85rem] mt-0.5">{subtitle}</div>
      </div>
      <p className="text-ink-soft text-[0.92rem] leading-[1.5]">{body}</p>
      {preview.length > 0 && (
        <ul className="text-[0.82rem] text-ink-mute leading-[1.55] grid grid-cols-2 gap-x-3">
          {preview.map((n) => (
            <li key={n}>· {n}</li>
          ))}
        </ul>
      )}
      <Button
        type="button"
        variant="primary"
        size="lg"
        disabled={disabled}
        onClick={onClick}
        className="mt-auto"
      >
        {busy ? 'Setting up...' : cta}
      </Button>
    </div>
  )
}
