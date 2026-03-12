'use client';

import { useState } from 'react';
import Link from 'next/link';
import { supabase } from '@/lib/supabase/client';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import Logo from '@/components/layout/Logo';

export default function UpdatePasswordPage() {
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [done, setDone] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (password.length < 8) {
      setError('Password must be at least 8 characters');
      return;
    }

    if (password !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    setLoading(true);

    try {
      const { error } = await supabase.auth.updateUser({ password });
      if (error) throw error;
      // Revoke all other sessions — critical if account was compromised
      await supabase.auth.signOut({ scope: 'others' });
      setDone(true);
    } catch (err: any) {
      setError(err.message || 'Failed to update password');
    } finally {
      setLoading(false);
    }
  };

  if (done) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-primary p-4">
        <Card className="w-full max-w-md bg-secondary border-gray-700">
          <CardHeader className="space-y-4">
            <div className="flex justify-center">
              <Logo />
            </div>
            <CardTitle className="text-2xl text-center text-white">Password updated</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4 text-center pb-8">
            <p className="text-gray-300">Your password has been updated successfully.</p>
            <Link
              href="/auth/login"
              className="inline-flex items-center gap-1 text-sm text-accent hover:underline"
            >
              Sign in with your new password
              <svg viewBox="0 0 16 16" className="w-3 h-3" fill="currentColor">
                <path fillRule="evenodd" d="M2 8a.75.75 0 01.75-.75h8.69L8.22 4.03a.75.75 0 011.06-1.06l4.5 4.5a.75.75 0 010 1.06l-4.5 4.5a.75.75 0 01-1.06-1.06l3.22-3.22H2.75A.75.75 0 012 8z" />
              </svg>
            </Link>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-primary p-4">
      <Card className="w-full max-w-md bg-secondary border-gray-700">
        <CardHeader className="space-y-4">
          <div className="flex justify-center">
            <Logo />
          </div>
          <CardTitle className="text-2xl text-center text-white">Set new password</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="password" className="text-gray-300">New password</Label>
              <Input
                id="password"
                type="password"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                className="bg-primary border-gray-600 text-white placeholder:text-gray-500"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="confirmPassword" className="text-gray-300">Confirm new password</Label>
              <Input
                id="confirmPassword"
                type="password"
                placeholder="••••••••"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
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
              {loading ? 'Updating…' : 'Update password'}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
