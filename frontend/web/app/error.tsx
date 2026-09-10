'use client';

import { useEffect } from 'react';
import PageError from '@/components/PageError';

export default function ErrorPage({ error, reset }: { error: Error & { digest?: string }; reset: () => void }) {
  useEffect(() => { console.error('Route render failed', error); }, [error]);
  return <PageError reset={reset} />;
}
