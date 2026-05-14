'use client';

import { ReactNode, useEffect } from 'react';
import { usePathname } from 'next/navigation';
import AppLayout from './AppLayout';

interface ConditionalLayoutProps {
  children: ReactNode;
}

// Map a pathname to a short page label. Falls through to null for unknown
// routes which renders the bare brand title.
function pageNameFor(pathname: string | null): string | null {
  if (!pathname) return null;
  if (pathname === '/') return 'Dashboard';
  if (pathname.startsWith('/recordings/') && pathname !== '/recordings') return 'Recording';
  if (pathname === '/recordings') return 'Recordings';
  if (pathname.startsWith('/players/') && pathname !== '/players') return 'Player';
  if (pathname === '/players') return 'Players';
  if (pathname === '/upload') return 'Upload';
  if (pathname === '/profile') return 'Profile';
  if (pathname === '/settings') return 'Settings';
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
    if (typeof document === 'undefined') return;
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
