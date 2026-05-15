'use client';

import { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import { supabase } from '@/lib/supabase/client';
import type { AuthContextType, AuthUser, SignUpMetadata } from '@/types/auth';

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(true);
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    // Check active session
    supabase.auth.getSession().then(({ data: { session } }) => {
      setUser(session?.user ?? null);
      setLoading(false);
    });

    // Listen for auth changes
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((event, session) => {
      setUser(session?.user ?? null);
      setLoading(false);

      if (
        event === 'SIGNED_IN' &&
        (pathname?.startsWith('/auth') || pathname?.startsWith('/landing'))
      ) {
        // Redirect to the dashboard when the sign-in actually happened on an
        // auth OR landing page. Scoping to those paths avoids bouncing the
        // user when Supabase re-fires SIGNED_IN on tab switch / token refresh
        // while they're mid-task elsewhere in the app.
        router.push('/');
      } else if (event === 'SIGNED_OUT') {
        router.push('/landing');
      }
    });

    return () => subscription.unsubscribe();
  }, [router, pathname]);

  const signUp = async (email: string, password: string, metadata?: SignUpMetadata) => {
    const { data, error } = await supabase.auth.signUp({
      email,
      password,
      options: {
        data: metadata,
      },
    });

    if (error) throw error;

    // Supabase returns a fake user with empty identities when the email is already registered
    // (to prevent email enumeration). Detect this and show a helpful error.
    if (data?.user?.identities?.length === 0) {
      throw new Error('An account with this email already exists. Try signing in instead.');
    }

    // Don't redirect here - let middleware handle it after session is established
  };

  const signIn = async (email: string, password: string) => {
    const { data, error } = await supabase.auth.signInWithPassword({
      email,
      password,
    });

    if (error) throw error;

    // Don't redirect here - let the auth state change trigger the middleware redirect
  };

  const signOut = async () => {
    const { error } = await supabase.auth.signOut();
    if (error) throw error;

    // Redirect is handled by onAuthStateChange listener
  };

  const value = {
    user,
    loading,
    signUp,
    signIn,
    signOut,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
