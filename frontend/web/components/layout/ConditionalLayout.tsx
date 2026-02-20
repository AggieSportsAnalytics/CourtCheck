'use client';

import { ReactNode } from 'react';
import { usePathname } from 'next/navigation';
import AppLayout from './AppLayout';

interface ConditionalLayoutProps {
  children: ReactNode;
}

export default function ConditionalLayout({ children }: ConditionalLayoutProps) {
  const pathname = usePathname();

  // Don't wrap auth or landing pages with AppLayout
  const isNoLayout = pathname?.startsWith('/auth') || pathname?.startsWith('/landing');

  if (isNoLayout) {
    return <>{children}</>;
  }

  return <AppLayout>{children}</AppLayout>;
}
