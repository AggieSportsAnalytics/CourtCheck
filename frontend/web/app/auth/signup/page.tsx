'use client'

import Link from 'next/link'
import { useState } from 'react'

import { useAuth } from '@/contexts/AuthContext'
import { supabase } from '@/lib/supabase/client'
import { AuthShell } from '@/components/auth/AuthShell'
import {
  AuthCard,
  AuthFoot,
  AuthInlineLink,
  AuthSub,
  AuthTitle,
} from '@/components/auth/AuthCard'
import {
  Field,
  FieldControl,
  FieldError,
  FieldLabel,
} from '@/components/auth/Field'
import { GoogleButton, OrDivider } from '@/components/auth/GoogleButton'
import { BrandMark } from '@/components/brand/BrandMark'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

type Role = 'Coach' | 'Player'
const ROLES: readonly Role[] = ['Coach', 'Player'] as const

export default function SignupPage() {
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [role, setRole] = useState<Role>('Coach')
  const [password, setPassword] = useState('')
  const [pwError, setPwError] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [emailSent, setEmailSent] = useState(false)
  const { signUp } = useAuth()

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setPwError('')
    setError('')

    if (password.length < 10) {
      setPwError('At least 10 characters.')
      return
    }

    setLoading(true)

    try {
      await signUp(email, password, {
        name: name.trim(),
        role,
      })
      setEmailSent(true)
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Failed to create account.'
      setError(message)
      setLoading(false)
    }
  }

  async function handleGoogle() {
    setError('')
    try {
      const { error: oauthError } = await supabase.auth.signInWithOAuth({
        provider: 'google',
        options: {
          redirectTo: `${window.location.origin}/auth/confirm`,
        },
      })
      if (oauthError) throw oauthError
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Couldn't reach Google."
      setError(message)
    }
  }

  if (emailSent) {
    return (
      <AuthShell>
        <AuthCard>
          <div className="flex justify-center mb-6">
            <BrandMark size="md" href={null} />
          </div>
          <div className="text-center pt-2 pb-1">
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
                <rect x="2" y="4" width="20" height="16" rx="2" />
                <path d="m22 7-10 7L2 7" />
              </svg>
            </div>
            <h3
              className="font-display font-medium text-[1.5rem] tracking-[-0.014em] mb-2"
              style={{ fontVariationSettings: '"opsz" 72' }}
            >
              Check your <em>email.</em>
            </h3>
            <p className="text-ink-soft text-[0.95rem] leading-[1.5] max-w-[32ch] mx-auto mb-5">
              We sent a confirmation link to{' '}
              <span className="text-ink font-medium">{email}</span>. Click it to
              activate your account.
            </p>
            <p className="text-[0.85rem] text-ink-mute">
              Already confirmed?{' '}
              <Link href="/auth/login" legacyBehavior passHref>
                <AuthInlineLink>Sign in</AuthInlineLink>
              </Link>
            </p>
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
          Create your <em>account.</em>
        </AuthTitle>
        <AuthSub className="text-center">
          Upload your first recording in under a minute.
        </AuthSub>

        <GoogleButton label="Sign up with Google" onClick={handleGoogle} />

        <OrDivider />

        <form onSubmit={handleSubmit} className="grid gap-3.5 mb-4">
          <Field>
            <FieldLabel>I am a</FieldLabel>
            <div className="grid grid-cols-2 gap-3" role="radiogroup" aria-label="Role">
              {ROLES.map((r) => {
                const selected = role === r
                return (
                  <button
                    key={r}
                    type="button"
                    role="radio"
                    aria-checked={selected}
                    onClick={() => setRole(r)}
                    className={cn(
                      'px-4 py-3 rounded-[10px] text-center',
                      'border transition-[border-color,background-color,color] duration-[160ms] ease-[cubic-bezier(0.2,0.8,0.2,1)]',
                      'cursor-pointer outline-none',
                      'font-medium text-[0.98rem]',
                      'focus-visible:ring-2 focus-visible:ring-court/40 focus-visible:ring-offset-2 focus-visible:ring-offset-paper',
                      selected
                        ? 'border-court bg-[color-mix(in_srgb,var(--color-court)_10%,transparent)] text-ink'
                        : 'border-line bg-shade text-ink-soft hover:border-ink dark:bg-surface',
                    )}
                  >
                    {r}
                  </button>
                )
              })}
            </div>
          </Field>

          <Field>
            <FieldLabel htmlFor="name">Full name</FieldLabel>
            <FieldControl>
              <input
                id="name"
                type="text"
                placeholder="Brian Le"
                autoComplete="name"
                required
                value={name}
                onChange={(e) => setName(e.target.value)}
              />
            </FieldControl>
          </Field>

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

          <Field>
            <FieldLabel htmlFor="password">Password</FieldLabel>
            <FieldControl>
              <input
                id="password"
                type="password"
                placeholder="At least 10 characters"
                autoComplete="new-password"
                required
                aria-invalid={pwError ? true : undefined}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </FieldControl>
            {pwError && <FieldError>{pwError}</FieldError>}
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
                Creating...
              </>
            ) : (
              'Sign up'
            )}
          </Button>
        </form>

        <AuthFoot>
          Already have an account?{' '}
          <Link href="/auth/login" legacyBehavior passHref>
            <AuthInlineLink>
              <em>Sign in.</em>
            </AuthInlineLink>
          </Link>
        </AuthFoot>
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

