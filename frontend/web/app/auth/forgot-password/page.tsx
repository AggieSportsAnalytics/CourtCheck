'use client'

import Link from 'next/link'
import { useState } from 'react'
import { ArrowLeft } from 'lucide-react'

import { supabase } from '@/lib/supabase/client'
import { AuthShell } from '@/components/auth/AuthShell'
import {
  AuthCard,
  AuthFoot,
  AuthInlineLink,
  AuthSub,
  AuthTitle,
} from '@/components/auth/AuthCard'
import { Field, FieldControl, FieldLabel } from '@/components/auth/Field'
import { BrandMark } from '@/components/brand/BrandMark'
import { Button } from '@/components/ui/button'

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState('')
  const [loading, setLoading] = useState(false)
  const [sent, setSent] = useState(false)
  const [error, setError] = useState('')

  async function sendLink() {
    const { error: resetError } = await supabase.auth.resetPasswordForEmail(
      email,
      {
        redirectTo: `${window.location.origin}/auth/reset-password`,
      },
    )
    if (resetError) throw resetError
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      await sendLink()
      setSent(true)
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Failed to send reset link.'
      setError(message)
    } finally {
      setLoading(false)
    }
  }

  async function handleResend() {
    if (!email) return
    setError('')
    try {
      await sendLink()
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Failed to resend.'
      setError(message)
    }
  }

  return (
    <AuthShell>
      <AuthCard>
        <div className="flex justify-center mb-6">
          <BrandMark size="md" href={null} />
        </div>

        {sent ? (
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
              Check your <em>email.</em>
            </h3>
            <p className="text-ink-soft text-[0.95rem] leading-[1.5] max-w-[32ch] mx-auto mb-5">
              We sent a reset link to{' '}
              <span className="text-ink font-medium">{email}</span>. The link
              expires in 15 minutes.
            </p>
            <p className="text-[0.85rem] text-ink-mute">
              Didn&apos;t get it?{' '}
              <button
                type="button"
                onClick={handleResend}
                className="text-court font-medium border-b border-current dark:text-court-light cursor-pointer"
              >
                Resend
              </button>
            </p>
            <div className="mt-7">
              <Link
                href="/auth/login"
                className="inline-flex items-center gap-1.5 text-[0.88rem] text-ink-mute hover:text-ink transition-colors"
              >
                <ArrowLeft className="size-3.5" />
                Back to sign in
              </Link>
            </div>
          </div>
        ) : (
          <>
            <AuthTitle className="text-center">
              Reset your <em>password.</em>
            </AuthTitle>
            <AuthSub className="text-center">
              Enter the email tied to your account. We&apos;ll send a secure
              reset link.
            </AuthSub>

            <form onSubmit={handleSubmit} className="grid gap-3.5 mb-4">
              <Field>
                <FieldLabel htmlFor="email">Email</FieldLabel>
                <FieldControl>
                  <input
                    id="email"
                    type="email"
                    placeholder="coach@school.edu"
                    autoComplete="email"
                    required
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                  />
                </FieldControl>
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
                    Sending...
                  </>
                ) : (
                  'Send reset link'
                )}
              </Button>
            </form>

            <AuthFoot>
              Remembered it?{' '}
              <Link href="/auth/login" legacyBehavior passHref>
                <AuthInlineLink>
                  <em>Sign in.</em>
                </AuthInlineLink>
              </Link>
            </AuthFoot>
          </>
        )}
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
