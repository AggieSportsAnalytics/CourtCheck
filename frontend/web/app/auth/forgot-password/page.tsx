'use client';

import { useState } from 'react';
import Link from 'next/link';
import { supabase } from '@/lib/supabase/client';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import Logo from '@/components/layout/Logo';

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [emailSent, setEmailSent] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const { error } = await supabase.auth.resetPasswordForEmail(email, {
        redirectTo: `${window.location.origin}/auth/reset-password`,
      });
      if (error) throw error;
      // Always show success — never reveal whether email exists (enumeration protection)
      setEmailSent(true);
    } catch (err: any) {
      setError(err.message || 'Failed to send reset email');
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      {/* Faint court-grid background — suggests tennis court lines */}
      <style>{`
        .court-bg {
          position: fixed;
          inset: 0;
          pointer-events: none;
          z-index: 0;
          background-image:
            linear-gradient(rgba(180,240,0,0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(180,240,0,0.03) 1px, transparent 1px);
          background-size: 48px 48px;
          mask-image: radial-gradient(ellipse 70% 70% at 50% 50%, black 0%, transparent 100%);
        }
        @keyframes envelope-drop {
          0%   { opacity: 0; transform: translateY(-12px) scale(0.92); }
          100% { opacity: 1; transform: translateY(0) scale(1); }
        }
        @keyframes envelope-pulse {
          0%, 100% { box-shadow: 0 0 0 0 rgba(180,240,0,0.15); }
          50%       { box-shadow: 0 0 0 12px rgba(180,240,0,0); }
        }
        .envelope-icon {
          animation: envelope-drop 0.45s cubic-bezier(0.34,1.56,0.64,1) forwards,
                     envelope-pulse 2s ease-in-out 0.5s infinite;
        }
      `}</style>

      <div className="court-bg" />

      <div className="relative z-10 min-h-screen flex items-center justify-center bg-primary p-4">
        {emailSent ? (
          <Card className="w-full max-w-md bg-secondary border-gray-700">
            <CardHeader className="space-y-4 pb-2">
              <div className="flex justify-center">
                <Logo />
              </div>

              {/* Animated envelope */}
              <div className="flex justify-center pt-2">
                <div
                  className="envelope-icon w-16 h-16 rounded-2xl flex items-center justify-center"
                  style={{ background: 'rgba(180,240,0,0.08)', border: '1px solid rgba(180,240,0,0.2)' }}
                >
                  <svg viewBox="0 0 24 24" className="w-8 h-8" fill="none" stroke="#B4F000" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
                    <rect x="2" y="4" width="20" height="16" rx="2" />
                    <path d="M2 7l10 7 10-7" />
                  </svg>
                </div>
              </div>

              <CardTitle className="text-2xl text-center text-white">Check your email</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4 text-center pb-8">
              <p className="text-gray-300">
                We sent a password reset link to{' '}
                <span className="text-white font-semibold">{email}</span>.
              </p>
              <p className="text-sm text-gray-400">
                Click the link in the email to reset your password.
                Check your spam folder if you don&apos;t see it within a minute.
              </p>
              <div className="pt-2">
                <Link href="/auth/login" className="text-sm text-accent hover:underline inline-flex items-center gap-1">
                  <svg viewBox="0 0 16 16" className="w-3 h-3" fill="currentColor">
                    <path fillRule="evenodd" d="M14 8a.75.75 0 01-.75.75H4.56l3.22 3.22a.75.75 0 11-1.06 1.06l-4.5-4.5a.75.75 0 010-1.06l4.5-4.5a.75.75 0 011.06 1.06L4.56 7.25h8.69A.75.75 0 0114 8z" />
                  </svg>
                  Back to sign in
                </Link>
              </div>
            </CardContent>
          </Card>
        ) : (
          <Card className="w-full max-w-md bg-secondary border-gray-700">
            <CardHeader className="space-y-4">
              <div className="flex justify-center">
                <Logo />
              </div>
              <div className="space-y-1 text-center">
                <CardTitle className="text-2xl text-white">Reset your password</CardTitle>
                <p className="text-sm text-gray-400">
                  Enter your email and we&apos;ll send you a reset link.
                </p>
              </div>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="email" className="text-gray-300">Email</Label>
                  <Input
                    id="email"
                    type="email"
                    placeholder="you@example.com"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                    className="bg-primary border-gray-600 text-white placeholder:text-gray-500"
                  />
                </div>

                {error && (
                  <div className="p-3 bg-red-900 bg-opacity-30 border border-red-600 rounded-lg">
                    <p className="text-sm text-red-200">{error}</p>
                  </div>
                )}

                <Button
                  type="submit"
                  disabled={loading}
                  className="w-full bg-accent text-primary hover:bg-opacity-90 font-semibold"
                >
                  {loading ? 'Sending…' : 'Send reset link'}
                </Button>

                <p className="text-center text-sm text-gray-400">
                  <Link href="/auth/login" className="text-accent hover:underline inline-flex items-center gap-1">
                    <svg viewBox="0 0 16 16" className="w-3 h-3" fill="currentColor">
                      <path fillRule="evenodd" d="M14 8a.75.75 0 01-.75.75H4.56l3.22 3.22a.75.75 0 11-1.06 1.06l-4.5-4.5a.75.75 0 010-1.06l4.5-4.5a.75.75 0 011.06 1.06L4.56 7.25h8.69A.75.75 0 0114 8z" />
                    </svg>
                    Back to sign in
                  </Link>
                </p>
              </form>
            </CardContent>
          </Card>
        )}
      </div>
    </>
  );
}
