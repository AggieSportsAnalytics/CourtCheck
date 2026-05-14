'use client'

import { useState } from 'react'
import Link from 'next/link'
import { useAuth } from '@/contexts/AuthContext'
import { createBrowserClient } from '@supabase/ssr'
import { ThemeToggle } from '@/components/brand/ThemeToggle'

type SaveMsg = { kind: 'ok' | 'err'; text: string }

export default function SettingsPage() {
  const { user, signOut } = useAuth()

  const supabase = createBrowserClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
  )

  const currentName = user?.user_metadata?.name || user?.email?.split('@')[0] || ''
  const email = user?.email || ''
  const role = user?.user_metadata?.role || 'Coach'

  const [displayName, setDisplayName] = useState(currentName)
  const [saving, setSaving] = useState(false)
  const [saveMsg, setSaveMsg] = useState<SaveMsg | null>(null)
  const [confirmSignOut, setConfirmSignOut] = useState(false)

  async function handleSaveName() {
    const next = displayName.trim()
    if (!next || next === currentName) return
    setSaving(true)
    setSaveMsg(null)
    try {
      const { error } = await supabase.auth.updateUser({ data: { name: next } })
      if (error) throw error
      setSaveMsg({ kind: 'ok', text: 'Display name updated.' })
    } catch (e) {
      setSaveMsg({ kind: 'err', text: (e as Error).message })
    } finally {
      setSaving(false)
    }
  }

  const initials = currentName
    .split(' ')
    .filter(Boolean)
    .slice(0, 2)
    .map((p: string) => p[0]?.toUpperCase())
    .join('')

  return (
    <div className="container mx-auto max-w-[760px] px-6 md:px-14 py-10">
      {/* Header */}
      <section className="pt-6 pb-7">
        <span
          aria-hidden
          className="inline-flex items-center gap-2 font-mono uppercase tracking-[0.18em] text-[0.72rem] text-court dark:text-court-light"
        >
          <span className="w-1.5 h-1.5 rounded-full bg-clay dark:bg-clay-soft" />
          Account
        </span>
        <h1
          className="text-ink mt-4"
          style={{
            fontFamily: 'var(--font-display)',
            fontWeight: 500,
            letterSpacing: '-0.022em',
            lineHeight: 1.15,
            fontSize: 'clamp(40px, 4.8vw, 64px)',
            paddingTop: '0.08em',
          }}
        >
          Your <em>settings</em>.
        </h1>
        <p className="text-ink-soft text-base mt-3">
          Manage your account, display, and signed-in sessions.
        </p>
      </section>

      {/* Profile section */}
      <section className="cc-card p-7 mb-6">
        <header className="flex items-center justify-between gap-4 mb-5">
          <h2 className="font-mono uppercase tracking-[0.16em] text-[0.7rem] text-court dark:text-court-light">
            Profile
          </h2>
          <Link
            href="/profile"
            className="font-mono uppercase tracking-[0.14em] text-[0.66rem] text-ink-mute hover:text-ink transition-colors"
          >
            View profile →
          </Link>
        </header>

        {/* Identity row */}
        <div className="flex items-center gap-4 pb-5 border-b border-line-soft">
          <div
            className="w-14 h-14 rounded-full flex items-center justify-center text-cream flex-shrink-0"
            style={{
              background: 'var(--color-court)',
              fontFamily: 'var(--font-display)',
              fontWeight: 500,
              fontSize: '1.4rem',
            }}
          >
            {initials || 'P'}
          </div>
          <div className="min-w-0">
            <p
              className="text-ink"
              style={{
                fontFamily: 'var(--font-display)',
                fontWeight: 500,
                fontSize: '1.25rem',
                letterSpacing: '-0.012em',
                lineHeight: 1.25,
                paddingTop: '0.06em',
              }}
            >
              {currentName}
            </p>
            <p className="text-ink-mute font-mono text-[0.72rem] tracking-[0.04em] mt-0.5">
              {email}
            </p>
            <p className="text-court dark:text-court-light font-mono uppercase tracking-[0.14em] text-[0.62rem] mt-1">
              {role}
            </p>
          </div>
        </div>

        {/* Display name */}
        <div className="pt-5 pb-5 border-b border-line-soft">
          <label
            htmlFor="displayName"
            className="block font-mono uppercase tracking-[0.14em] text-[0.66rem] text-ink-mute mb-2"
          >
            Display name
          </label>
          <div className="flex gap-2">
            <input
              id="displayName"
              type="text"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              placeholder="Your name"
              className="flex-1 rounded-md px-3 py-2.5 text-sm bg-cream text-ink placeholder:text-ink-mute border border-line focus:border-court focus:outline-none focus:ring-1 focus:ring-court transition-colors"
            />
            <button
              type="button"
              onClick={handleSaveName}
              disabled={saving || !displayName.trim() || displayName.trim() === currentName}
              className="px-4 py-2.5 rounded-md text-sm font-medium bg-court text-cream hover:-translate-y-px transition-transform disabled:opacity-50 disabled:translate-y-0 disabled:cursor-not-allowed dark:bg-court-deep dark:hover:bg-court"
            >
              {saving ? 'Saving.' : 'Save'}
            </button>
          </div>
          {saveMsg && (
            <p
              className={`text-xs mt-2 ${
                saveMsg.kind === 'ok'
                  ? 'text-court dark:text-court-light'
                  : 'text-clay dark:text-clay-soft'
              }`}
            >
              {saveMsg.text}
            </p>
          )}
        </div>

        {/* Email (read-only) */}
        <div className="pt-5">
          <label
            htmlFor="email"
            className="block font-mono uppercase tracking-[0.14em] text-[0.66rem] text-ink-mute mb-2"
          >
            Email <span className="text-ink-mute/70">(read only)</span>
          </label>
          <input
            id="email"
            type="email"
            value={email}
            readOnly
            className="w-full rounded-md px-3 py-2.5 text-sm bg-shade text-ink-soft border border-line cursor-not-allowed"
          />
          <p className="text-ink-mute text-xs mt-2">
            Need to change this? <em>Contact support</em>.
          </p>
        </div>
      </section>

      {/* Appearance */}
      <section className="cc-card p-7 mb-6">
        <h2 className="font-mono uppercase tracking-[0.16em] text-[0.7rem] text-court dark:text-court-light mb-5">
          Appearance
        </h2>
        <div className="flex items-center justify-between gap-4">
          <div className="min-w-0">
            <p className="text-ink text-base font-medium">Theme</p>
            <p className="text-ink-soft text-sm mt-0.5">
              Cream paper by day, stadium at night. Tap to flip.
            </p>
          </div>
          <ThemeToggle />
        </div>
      </section>

      {/* Account */}
      <section className="cc-card p-7">
        <h2 className="font-mono uppercase tracking-[0.16em] text-[0.7rem] text-court dark:text-court-light mb-5">
          Account
        </h2>

        {!confirmSignOut ? (
          <div className="flex items-center justify-between gap-4">
            <div className="min-w-0">
              <p className="text-ink text-base font-medium">Sign out</p>
              <p className="text-ink-soft text-sm mt-0.5">
                End your session on this device.
              </p>
            </div>
            <button
              type="button"
              onClick={() => setConfirmSignOut(true)}
              className="px-4 py-2.5 rounded-md text-sm font-medium border border-clay text-clay hover:bg-clay hover:text-cream transition-colors dark:border-clay-soft dark:text-clay-soft dark:hover:bg-clay-soft dark:hover:text-cream"
            >
              Sign out
            </button>
          </div>
        ) : (
          <div>
            <p className="text-ink text-base mb-4">
              Sign out of CourtCheck on this device?
            </p>
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                onClick={() => signOut()}
                className="px-4 py-2.5 rounded-md text-sm font-medium bg-clay text-cream hover:-translate-y-px transition-transform"
              >
                Yes, sign me out
              </button>
              <button
                type="button"
                onClick={() => setConfirmSignOut(false)}
                className="px-4 py-2.5 rounded-md text-sm font-medium border border-line text-ink-soft hover:border-ink hover:text-ink transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        )}
      </section>
    </div>
  )
}
