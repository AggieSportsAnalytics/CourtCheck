'use client'

import Link from 'next/link'
import { useState } from 'react'
import { ArrowRight } from 'lucide-react'

import { supabase } from '@/lib/supabase/client'
import { AuthShell } from '@/components/auth/AuthShell'
import {
  AuthCard,
  AuthSub,
  AuthTitle,
} from '@/components/auth/AuthCard'
import {
  Field,
  FieldControl,
  FieldError,
  FieldLabel,
} from '@/components/auth/Field'
import { BrandMark } from '@/components/brand/BrandMark'
import { Button } from '@/components/ui/button'

export default function UpdatePasswordPage() {
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [pwError, setPwError] = useState('')
  const [confirmError, setConfirmError] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [done, setDone] = useState(false)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setPwError('')
    setConfirmError('')
    setError('')

    if (password.length < 10) {
      setPwError('At least 10 characters.')
      return
    }
    if (password !== confirmPassword) {
      setConfirmError("Passwords don't match.")
      return
    }

    setLoading(true)
    try {
      const { error: updateError } = await supabase.auth.updateUser({
        password,
      })
      if (updateError) throw updateError
      // Revoke all other sessions for safety.
      await supabase.auth.signOut({ scope: 'others' })
      setDone(true)
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Failed to update password.'
      setError(message)
    } finally {
      setLoading(false)
    }
  }

  if (done) {
    return (
      <AuthShell>
        <AuthCard>
          <div className="flex justify-center mb-6">
            <BrandMark size="md" href={null} />
          </div>
          <div className="text-center pt-1 pb-1">
            <div className="size-16 mx-auto mb-5 rounded-full inline-flex items-center justify-center bg-[color-mix(in_srgb,var(--color-court)_12%,transparent)] text-court dark:bg-[color-mix(in_srgb,var(--color-court-light)_18%,transparent)] dark:text-court-light">
              <svg
                viewBox="0 0 24 24"
                width="26"
                height="26"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.75"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="m4 12 6 6L20 6" />
              </svg>
            </div>
            <h3
              className="font-display font-medium text-[1.5rem] tracking-[-0.014em] mb-2"
              style={{ fontVariationSettings: '"opsz" 72' }}
            >
              Password <em>updated.</em>
            </h3>
            <p className="text-ink-soft text-[0.95rem] leading-[1.5] max-w-[32ch] mx-auto mb-6">
              Other sessions have been signed out for safety.
            </p>
            <Link
              href="/auth/login"
              className="inline-flex items-center gap-1.5 text-court font-medium border-b border-current dark:text-court-light"
            >
              Sign in with your new password
              <ArrowRight className="size-3.5" />
            </Link>
          </div>
        </AuthCard>
      </AuthShell>
    )
  }

  return (
    <AuthShell>
      <AuthCard>
        <div className="flex justify-center mb-6">
          <BrandMark size="md" href={null} />
        </div>

        <AuthTitle className="text-center">
          Set a new <em>password.</em>
        </AuthTitle>
        <AuthSub className="text-center">
          At least 10 characters. Longer is stronger.
        </AuthSub>

        <form onSubmit={handleSubmit} className="grid gap-3.5 mb-4">
          <Field>
            <FieldLabel htmlFor="password">New password</FieldLabel>
            <FieldControl>
              <input
                id="password"
                type="password"
                placeholder="••••••••••"
                autoComplete="new-password"
                required
                aria-invalid={pwError ? true : undefined}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </FieldControl>
            {pwError && <FieldError>{pwError}</FieldError>}
          </Field>

          <Field>
            <FieldLabel htmlFor="confirmPassword">
              Confirm new password
            </FieldLabel>
            <FieldControl>
              <input
                id="confirmPassword"
                type="password"
                placeholder="••••••••••"
                autoComplete="new-password"
                required
                aria-invalid={confirmError ? true : undefined}
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
              />
            </FieldControl>
            {confirmError && <FieldError>{confirmError}</FieldError>}
          </Field>

          {error && (
            <div
              role="alert"
              className="rounded-[10px] border border-clay bg-[color-mix(in_srgb,var(--color-clay)_8%,transparent)] text-clay text-[0.88rem] leading-[1.45] px-3.5 py-2.5"
            >
              {error}
            </div>
          )}

          <Button
            type="submit"
            variant="primary"
            size="lg"
            disabled={loading}
            className="w-full mt-1"
          >
            {loading ? (
              <>
                <Spinner />
                Updating...
              </>
            ) : (
              'Update password'
            )}
          </Button>
        </form>
      </AuthCard>
    </AuthShell>
  )
}

function Spinner() {
  return (
    <svg
      className="size-4 animate-spin motion-reduce:animate-none"
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden
    >
      <circle
        cx="12"
        cy="12"
        r="9"
        stroke="currentColor"
        strokeWidth="2.5"
        opacity="0.25"
      />
      <path
        d="M21 12a9 9 0 0 0-9-9"
        stroke="currentColor"
        strokeWidth="2.5"
        strokeLinecap="round"
      />
    </svg>
  )
}
