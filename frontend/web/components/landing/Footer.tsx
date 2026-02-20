'use client';

import Link from 'next/link';
import Image from 'next/image';

const BALL_LOGO = 'https://raw.githubusercontent.com/AggieSportsAnalytics/CourtCheck/cory/images/courtcheck_ball_logo.png';

export function Footer() {
  return (
    <footer
      className="py-12"
      style={{ borderTop: '1px solid rgba(255,255,255,0.06)', background: '#07070A' }}
    >
      <div className="max-w-6xl mx-auto px-5 sm:px-8">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-8">
          {/* Logo + tagline */}
          <div>
            <Link href="/landing" className="flex items-center gap-2.5 mb-2">
              <Image src={BALL_LOGO} alt="CourtCheck" width={24} height={24} className="object-contain" />
              <span className="text-sm font-semibold text-white">CourtCheck</span>
            </Link>
            <p className="text-xs" style={{ color: '#4A4A55' }}>
              AI-powered tennis analytics for competitive players.
            </p>
          </div>

          {/* Links */}
          <nav className="flex flex-wrap gap-x-8 gap-y-2">
            {[
              { label: 'Features', href: '#features' },
              { label: 'How It Works', href: '#how-it-works' },
              { label: 'Insights', href: '#insights' },
              { label: 'Sign Up', href: '/auth/login' },
            ].map(({ label, href }) => (
              <a
                key={label}
                href={href}
                className="text-xs transition-colors duration-200"
                style={{ color: '#5A5A66' }}
                onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.color = '#FAFAFA'; }}
                onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.color = '#5A5A66'; }}
              >
                {label}
              </a>
            ))}
          </nav>
        </div>

        <div className="mt-10 pt-6 flex items-center justify-between" style={{ borderTop: '1px solid rgba(255,255,255,0.04)' }}>
          <p className="text-xs" style={{ color: '#4A4A55' }}>
            © {new Date().getFullYear()} CourtCheck · Aggie Sports Analytics
          </p>
          <p className="text-xs font-mono" style={{ color: '#2A2A30' }}>
            v2.0 · GPU-accelerated
          </p>
        </div>
      </div>
    </footer>
  );
}
