import Link from 'next/link';
import { BrandMark } from '@/components/brand/BrandMark';

export default function NotFound() {
  return (
    <div className="min-h-[60vh] flex flex-col items-center justify-center gap-5 px-6 text-center">
      <BrandMark size="md" href={null} />
      <h1 className="font-display text-[1.75rem]">Page not found.</h1>
      <p className="text-ink-soft">Open your dashboard or Recordings to continue.</p>
      <div className="flex flex-wrap justify-center gap-3">
        <Link href="/" className="min-h-11 rounded-full bg-court text-cream px-5 py-3">Go to Dashboard</Link>
        <Link href="/recordings" className="min-h-11 rounded-full border border-line px-5 py-3 text-court">Go to Recordings</Link>
      </div>
    </div>
  );
}
