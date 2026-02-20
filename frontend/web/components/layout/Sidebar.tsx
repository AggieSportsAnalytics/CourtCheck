'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useEffect, useState } from 'react';
import Logo from './Logo';

interface SidebarProps {
  username: string;
}

interface NavItem {
  name: string;
  href: string;
  icon: React.ReactNode;
}

interface RecentRecording {
  id: string;
  filename: string;
  status: string;
  createdAt: string;
}

const Sidebar = ({ username }: SidebarProps) => {
  const pathname = usePathname();
  const [recentRecordings, setRecentRecordings] = useState<RecentRecording[]>([]);

  useEffect(() => {
    let cancelled = false;
    fetch('/api/recordings')
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (!cancelled && data?.recordings) {
          setRecentRecordings(
            data.recordings
              .filter((r: RecentRecording) => r.status === 'done')
              .slice(0, 3)
          );
        }
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, []);

  const navItems: NavItem[] = [
    {
      name: 'Dashboard',
      href: '/',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.75} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
        </svg>
      ),
    },
    {
      name: 'Upload Video',
      href: '/upload',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.75} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
        </svg>
      ),
    },
    {
      name: 'Overall Stats',
      href: '/overall-stats',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.75} d="M16 8v8m-4-5v5m-4-2v2m-2 4h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
        </svg>
      ),
    },
    {
      name: 'Recordings',
      href: '/recordings',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.75} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
        </svg>
      ),
    },
  ];

  const isActive = (href: string) => {
    if (href === '/') return pathname === '/';
    return pathname.startsWith(href);
  };

  const initials = username
    .split(' ')
    .filter(Boolean)
    .slice(0, 2)
    .map((p) => p[0]?.toUpperCase())
    .join('');

  return (
    <div className="w-60 bg-secondary h-full flex flex-col shrink-0" style={{ borderRight: '1px solid rgba(255,255,255,0.07)' }}>
      {/* Logo */}
      <div className="p-5 flex items-center gap-3" style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
        <Link href="/landing">
          <Logo />
        </Link>
      </div>

      {/* User */}
      <div className="px-4 py-3" style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
        <div className="flex items-center gap-3">
          <div
            className="w-8 h-8 rounded-full flex items-center justify-center text-primary font-bold text-xs shrink-0"
            style={{ background: '#B4F000' }}
          >
            {initials || 'U'}
          </div>
          <div className="min-w-0">
            <p className="text-sm font-semibold text-white truncate">{username}</p>
            <p className="text-xs" style={{ color: '#5A5A66' }}>Tennis Player</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto p-3">
        <div className="space-y-0.5">
          {navItems.map((item) => (
            <Link
              key={item.name}
              href={item.href}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-150 text-sm ${
                isActive(item.href)
                  ? 'font-semibold'
                  : 'font-medium hover:bg-white/5'
              }`}
              style={
                isActive(item.href)
                  ? { background: 'rgba(180,240,0,0.12)', color: '#B4F000' }
                  : { color: '#6B6B78' }
              }
              aria-label={`Navigate to ${item.name}`}
            >
              {item.icon}
              <span>{item.name}</span>
            </Link>
          ))}
        </div>

        {/* Recent Analyses */}
        <div className="mt-6">
          <div className="flex items-center justify-between px-3 mb-2">
            <h3 className="text-[10px] font-semibold uppercase tracking-widest" style={{ color: '#3A3A44' }}>
              Recent
            </h3>
            {recentRecordings.length > 0 && (
              <Link
                href="/recordings"
                className="text-[10px] transition-colors"
                style={{ color: '#4A4A55' }}
                onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.color = '#B4F000'; }}
                onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.color = '#4A4A55'; }}
              >
                See all
              </Link>
            )}
          </div>

          {recentRecordings.length === 0 ? (
            <div className="px-3 py-2">
              <p className="text-xs" style={{ color: '#3A3A44' }}>
                No analyses yet.{' '}
                <Link href="/upload" className="transition-colors" style={{ color: '#B4F000' }}>
                  Upload a video
                </Link>
              </p>
            </div>
          ) : (
            <div className="space-y-0.5">
              {recentRecordings.map((rec, i) => {
                const shortName =
                  rec.filename.length > 18
                    ? rec.filename.slice(0, 15) + '…'
                    : rec.filename;
                const active = pathname === `/recordings/${rec.id}`;
                return (
                  <Link
                    key={rec.id}
                    href={`/recordings/${rec.id}`}
                    className="flex items-center gap-2.5 px-3 py-2 rounded-lg transition-all duration-150 text-sm"
                    style={
                      active
                        ? { background: 'rgba(180,240,0,0.12)', color: '#B4F000' }
                        : { color: '#6B6B78' }
                    }
                    onMouseEnter={(e) => { if (!active) (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.05)'; if (!active) (e.currentTarget as HTMLElement).style.color = '#FAFAFA'; }}
                    onMouseLeave={(e) => { if (!active) (e.currentTarget as HTMLElement).style.background = 'transparent'; if (!active) (e.currentTarget as HTMLElement).style.color = '#6B6B78'; }}
                    aria-label={`View analysis ${i + 1}`}
                  >
                    <div className="w-1.5 h-1.5 rounded-full shrink-0" style={{ background: active ? '#B4F000' : '#2A2A33' }} />
                    <span className="truncate text-xs">{shortName}</span>
                  </Link>
                );
              })}
            </div>
          )}
        </div>
      </nav>

      {/* Footer */}
      <div className="p-3" style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }}>
        <Link
          href="/upload"
          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg transition-all duration-150 text-sm font-semibold"
          style={{ background: 'rgba(180,240,0,0.1)', color: '#B4F000', border: '1px solid rgba(180,240,0,0.15)' }}
          onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = 'rgba(180,240,0,0.18)'; }}
          onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = 'rgba(180,240,0,0.1)'; }}
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>
          Upload Video
        </Link>
      </div>
    </div>
  );
};

export default Sidebar;
