import type { ReactNode } from 'react';
import { detailMetadata } from '@/lib/routeMetadata';

export async function generateMetadata({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return detailMetadata('matches', id, 'Recording');
}

export default function Layout({ children }: { children: ReactNode }) {
  return children;
}
