'use client';

import { ReactNode, useEffect } from 'react';
import { usePathname } from 'next/navigation';
import AppLayout from './AppLayout';

interface ConditionalLayoutProps {
  children: ReactNode;
}

// Auth pages retain their client labels; app route titles come from metadata.
function pageNameFor(pathname: string | null): string | null {
  if (!pathname) return null;
  if (pathname === '/auth/login') return 'Sign in';
  if (pathname === '/auth/signup') return 'Sign up';
  if (pathname === '/auth/forgot-password') return 'Reset password';
  if (pathname === '/auth/update-password') return 'Update password';
  if (pathname.startsWith('/auth')) return 'Account';
  return null;
}

export default function ConditionalLayout({ children }: ConditionalLayoutProps) {
  const pathname = usePathname();

  useEffect(() => {
    if (typeof document === 'undefined' || !pathname?.startsWith('/auth')) return;
    const page = pageNameFor(pathname);
    document.title = page ? `${page} | CourtCheck` : 'CourtCheck';
  }, [pathname]);

  // Don't wrap auth or landing pages with AppLayout
  const isNoLayout = pathname?.startsWith('/auth') || pathname?.startsWith('/landing');

  if (isNoLayout) {
    return <>{children}</>;
  }

  return <AppLayout>{children}</AppLayout>;
}
