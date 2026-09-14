'use client';

import Link from 'next/link';
import { BrandMark } from '@/components/brand/BrandMark';

export default function PageError({ reset }: { reset: () => void }) {
  return (
    <div className="min-h-[60vh] flex flex-col items-center justify-center gap-5 px-6 text-center">
      <BrandMark size="md" href={null} />
      <h1 className="font-display text-[1.75rem] text-clay">Something went wrong loading this page.</h1>
      <div className="flex flex-wrap justify-center gap-3">
        <button type="button" onClick={reset} className="min-h-11 rounded-full bg-court text-cream px-5 py-3">
          Try again
        </button>
        <Link href="/recordings" className="min-h-11 rounded-full border border-line px-5 py-3 text-court">
          Go to Recordings
        </Link>
      </div>
    </div>
  );
}
