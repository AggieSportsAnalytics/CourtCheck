'use client'

import Link from 'next/link'
import { useRouter, useSearchParams } from 'next/navigation'
import { Suspense, useState } from 'react'

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
import { Field, FieldControl, FieldLabel } from '@/components/auth/Field'
import { GoogleButton, OrDivider } from '@/components/auth/GoogleButton'
import { BrandMark } from '@/components/brand/BrandMark'
import { Button } from '@/components/ui/button'

export default function LoginPage() {
  return <Suspense fallback={<AuthShell><AuthCard>Loading sign in.</AuthCard></AuthShell>}><LoginForm /></Suspense>
}

function LoginForm() {
  const searchParams = useSearchParams()
  const router = useRouter()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(() => searchParams.get('error') ?? '')
  const { signIn } = useAuth()

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      await signIn(email, password)
      router.push('/')
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Failed to sign in.'
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

  return (
    <AuthShell>
      <AuthCard>
        <div className="flex justify-center mb-6">
          <BrandMark size="md" href={null} />
        </div>

        <AuthTitle className="text-center">
          Welcome back.
        </AuthTitle>
        <AuthSub className="text-center">
          Sign in to open your roster and recordings.
        </AuthSub>

        <GoogleButton label="Continue with Google" onClick={handleGoogle} />

        <OrDivider />

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
          <Field>
            <FieldLabel htmlFor="password">Password</FieldLabel>
            <FieldControl>
              <input
                id="password"
                type="password"
                placeholder="••••••••"
                autoComplete="current-password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </FieldControl>
          </Field>

          <div className="flex items-center justify-end text-[0.95rem] mt-1 mb-2">
            <Link
              href="/auth/forgot-password"
              className="text-court font-medium hover:underline"
            >
              Forgot password?
            </Link>
          </div>

          {error && (
            <div
              role="alert"
              className="rounded-[10px] border border-clay bg-[color-mix(in_srgb,var(--color-clay)_8%,transparent)] text-clay text-[0.95rem] leading-[1.45] px-3.5 py-2.5"
            >
              {error}
            </div>
          )}

          <Button
            type="submit"
            variant="primary"
            size="lg"
            disabled={loading}
            className="w-full"
          >
            {loading ? (
              <>
                <Spinner />
                Signing in...
              </>
            ) : (
              'Sign in'
            )}
          </Button>
        </form>

        <AuthFoot>
          New to CourtCheck?{' '}
          <Link href="/auth/signup" legacyBehavior passHref>
            <AuthInlineLink>
              Create an account.
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
