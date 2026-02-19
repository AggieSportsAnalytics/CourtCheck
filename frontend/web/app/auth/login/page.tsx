'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useAuth } from '@/contexts/AuthContext';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import Image from 'next/image';

export default function LoginPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { signIn } = useAuth();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      await signIn(email, password);
    } catch (err: any) {
      setError(err.message || 'Failed to sign in');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-primary p-4">
      <Card className="w-full max-w-md bg-secondary border-gray-700">
        <CardHeader className="space-y-4">
          <div className="flex flex-col items-center gap-1">
            <div className="flex items-center gap-3">
              <div className="mt-10">
                <Image
                  src="https://raw.githubusercontent.com/AggieSportsAnalytics/CourtCheck/cory/images/courtcheck_ball_logo.png"
                  alt="CourtCheck tennis ball logo"
                  width={48}
                  height={48}
                  className="object-contain"
                  style={{ filter: 'brightness(1.1)' }}
                />
              </div>
              <span className="text-2xl font-bold text-white mt-10">CourtCheck</span>
            </div>
            <p className="text-sm text-gray-400">Tennis Analytics</p>
          </div>
          <CardTitle className="text-2xl text-center text-white">Sign In</CardTitle>
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

            <div className="space-y-2">
              <Label htmlFor="password" className="text-gray-300">Password</Label>
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
              {loading ? 'Signing in...' : 'Sign In'}
            </Button>

            <p className="text-center text-sm text-gray-400">
              Don't have an account?{' '}
              <Link href="/auth/signup" className="text-accent hover:underline">
                Sign up
              </Link>
            </p>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
