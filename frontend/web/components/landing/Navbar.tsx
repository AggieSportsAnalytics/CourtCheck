'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { useAuth } from '@/contexts/AuthContext';

const BALL_LOGO = 'https://raw.githubusercontent.com/AggieSportsAnalytics/CourtCheck/cory/images/courtcheck_ball_logo.png';

const NAV_LINKS = [
  { label: 'Features', href: '#features' },
  { label: 'How It Works', href: '#how-it-works' },
  { label: 'Insights', href: '#insights' },
];

export function Navbar() {
  const { user } = useAuth();
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24);
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  return (
    <header
      className="fixed top-0 left-0 right-0 z-50 transition-all duration-300"
      style={{
        background: scrolled ? 'rgba(7,7,10,0.85)' : 'transparent',
        backdropFilter: scrolled ? 'blur(16px)' : 'none',
        borderBottom: scrolled ? '1px solid rgba(255,255,255,0.06)' : '1px solid transparent',
      }}
    >
      <div className="max-w-6xl mx-auto px-5 sm:px-8 h-16 flex items-center justify-between">
        {/* Logo */}
        <Link href="/landing" className="flex items-center gap-2.5">
          <div className="relative w-8 h-8 shrink-0">
            <Image src={BALL_LOGO} alt="CourtCheck" width={32} height={32} className="object-contain" />
          </div>
          <span className="text-[15px] font-semibold text-white tracking-tight">CourtCheck</span>
        </Link>

        {/* Desktop nav */}
        <nav className="hidden md:flex items-center gap-8">
          {NAV_LINKS.map(({ label, href }) => (
            <a
              key={label}
              href={href}
              className="text-sm text-[#8C8C99] hover:text-white transition-colors duration-200"
            >
              {label}
            </a>
          ))}
        </nav>

        {/* CTA */}
        <Link
          href={user ? '/' : '/auth/login'}
          className="text-sm font-semibold px-4 py-2 rounded-lg transition-all duration-200"
          style={{ background: '#B4F000', color: '#07070A' }}
          onMouseEnter={(e) => (e.currentTarget.style.background = '#C7FF00')}
          onMouseLeave={(e) => (e.currentTarget.style.background = '#B4F000')}
        >
          {user ? 'Dashboard' : 'Get Started'}
        </Link>
      </div>
    </header>
  );
}
