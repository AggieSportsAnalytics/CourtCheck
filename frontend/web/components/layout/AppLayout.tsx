'use client';

import { ReactNode } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import Sidebar from './Sidebar';
import { useAuth } from '@/contexts/AuthContext';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';

interface AppLayoutProps {
  children: ReactNode;
}

export default function AppLayout({ children }: AppLayoutProps) {
  const { user, loading, signOut } = useAuth();
  const router = useRouter();

  if (loading) {
    return (
      <div className="flex h-screen bg-primary text-white items-center justify-center">
        <div className="text-center">
          <div
            className="animate-spin rounded-full h-10 w-10 mx-auto mb-4"
            style={{ border: '2px solid rgba(180,240,0,0.2)', borderTopColor: '#B4F000' }}
          />
          <p className="text-sm" style={{ color: '#5A5A66' }}>Loading…</p>
        </div>
      </div>
    );
  }

  if (!user) {
    router.replace('/landing');
    return null;
  }

  const displayName = user.user_metadata?.name || user.email?.split('@')[0] || 'User';
  const displayEmail = user.email || '';
  const displayImage: string | null = user.user_metadata?.avatar_url || null;
  const initials = displayName
    .split(' ')
    .filter(Boolean)
    .slice(0, 2)
    .map((p: string) => p[0]?.toUpperCase())
    .join('');

  const handleSignOut = async () => {
    try {
      await signOut();
    } catch (error) {
      console.error('Error signing out:', error);
    }
  };

  return (
    <div className="flex h-screen bg-primary text-white overflow-hidden">
      <Sidebar username={displayName} />

      <div className="flex-1 overflow-y-auto">
        {/* Header */}
        <header
          className="px-6 py-3 flex justify-between items-center sticky top-0 z-10"
          style={{
            background: 'rgba(7,7,10,0.85)',
            backdropFilter: 'blur(12px)',
            WebkitBackdropFilter: 'blur(12px)',
            borderBottom: '1px solid rgba(255,255,255,0.06)',
          }}
        >
          <div className="text-sm" style={{ color: '#4A4A55' }}>
            Hey, <span className="text-white font-medium">{displayName}</span>
          </div>

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button className="flex items-center gap-2.5 transition-opacity hover:opacity-80">
                <Avatar className="size-8">
                  <AvatarImage src={displayImage ?? undefined} alt={`${displayName} avatar`} />
                  <AvatarFallback
                    className="text-xs font-semibold"
                    style={{ background: 'rgba(180,240,0,0.15)', color: '#B4F000' }}
                  >
                    {initials || 'U'}
                  </AvatarFallback>
                </Avatar>
                <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" style={{ color: '#4A4A55' }}>
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent
              align="end"
              className="w-52"
              style={{ background: '#111116', border: '1px solid rgba(255,255,255,0.08)' }}
            >
              <DropdownMenuLabel className="text-xs font-semibold" style={{ color: '#5A5A66' }}>
                {displayEmail}
              </DropdownMenuLabel>
              <DropdownMenuSeparator style={{ background: 'rgba(255,255,255,0.07)' }} />
              <DropdownMenuItem
                asChild
                className="cursor-pointer text-sm"
                style={{ color: '#9CA3AF' }}
              >
                <Link href="/profile" className="flex items-center gap-2">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                  </svg>
                  Profile
                </Link>
              </DropdownMenuItem>
              <DropdownMenuItem
                asChild
                className="cursor-pointer text-sm"
                style={{ color: '#9CA3AF' }}
              >
                <Link href="/settings" className="flex items-center gap-2">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                  Settings
                </Link>
              </DropdownMenuItem>
              <DropdownMenuSeparator style={{ background: 'rgba(255,255,255,0.07)' }} />
              <DropdownMenuItem
                onClick={handleSignOut}
                className="cursor-pointer text-sm text-red-400 hover:text-red-300"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                </svg>
                Sign Out
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </header>

        {/* Page content */}
        {children}
      </div>
    </div>
  );
}
