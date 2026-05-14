'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import { LayoutDashboard, Users, Upload, Film, ChevronLeft, LogOut, UserCircle, Settings } from 'lucide-react'
import { ThemeToggle } from '@/components/brand/ThemeToggle'
import { BrandMark } from '@/components/brand/BrandMark'

type SidebarUser = {
  name: string
  email: string
  initials: string
  imageUrl?: string | null
}

type Props = {
  user: SidebarUser
  onSignOut: () => Promise<void> | void
}

type NavItem = {
  name: string
  href: string
  icon: ReactNode
  match: (pathname: string) => boolean
}

const NAV_ITEMS: NavItem[] = [
  {
    name: 'Dashboard',
    href: '/',
    icon: <LayoutDashboard className="size-[18px]" strokeWidth={1.75} />,
    match: (p) => p === '/',
  },
  {
    name: 'Players',
    href: '/players',
    icon: <Users className="size-[18px]" strokeWidth={1.75} />,
    match: (p) => p.startsWith('/players'),
  },
  {
    name: 'Upload video',
    href: '/upload',
    icon: <Upload className="size-[18px]" strokeWidth={1.75} />,
    match: (p) => p.startsWith('/upload'),
  },
  {
    name: 'Recordings',
    href: '/recordings',
    icon: <Film className="size-[18px]" strokeWidth={1.75} />,
    match: (p) => p.startsWith('/recordings'),
  },
]

export default function Sidebar({ user, onSignOut }: Props) {
  const pathname = usePathname() || '/'
  const [collapsed, setCollapsed] = useState(false)
  const [menuOpen, setMenuOpen] = useState(false)
  const menuRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    try {
      const stored = localStorage.getItem('cc-sidebar-collapsed')
      const start = stored === '1'
      setCollapsed(start)
      document.body.classList.toggle('sidebar-collapsed', start)
    } catch {}

    const mq = window.matchMedia('(max-width: 1100px)')
    function applyViewport() {
      if (mq.matches) {
        document.body.classList.add('sidebar-collapsed')
      } else {
        try {
          const stored = localStorage.getItem('cc-sidebar-collapsed')
          document.body.classList.toggle('sidebar-collapsed', stored === '1')
        } catch {}
      }
    }
    applyViewport()
    mq.addEventListener('change', applyViewport)
    return () => mq.removeEventListener('change', applyViewport)
  }, [])

  useEffect(() => {
    if (!menuOpen) return
    function onClick(e: MouseEvent) {
      if (!menuRef.current?.contains(e.target as Node)) setMenuOpen(false)
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') setMenuOpen(false)
    }
    document.addEventListener('mousedown', onClick)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onClick)
      document.removeEventListener('keydown', onKey)
    }
  }, [menuOpen])

  function toggleCollapse() {
    const mq = window.matchMedia('(max-width: 1100px)')
    if (mq.matches) return
    const next = !collapsed
    setCollapsed(next)
    document.body.classList.toggle('sidebar-collapsed', next)
    try {
      localStorage.setItem('cc-sidebar-collapsed', next ? '1' : '0')
    } catch {}
  }

  return (
    <aside
      aria-label="Primary navigation"
      className="app-sidebar fixed top-0 left-0 z-50 flex flex-col bg-paper border-r border-line-soft box-border"
      style={{
        width: collapsed ? 72 : 200,
        height: '100vh',
        padding: collapsed ? '22px 8px' : '22px 14px',
        transition: 'width 220ms var(--ease-out), padding 220ms var(--ease-out)',
      }}
    >
      {/* Collapse handle on the right edge — clear of the brand-mark, vertically centered. */}
      <button
        type="button"
        onClick={toggleCollapse}
        aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        className="sidebar-collapse absolute z-[51] size-[22px] rounded-full border border-line bg-paper text-ink-mute hover:border-ink hover:text-ink flex items-center justify-center transition-colors cursor-pointer shadow-sm"
        style={{
          top: '50%',
          right: -11,
          transform: 'translateY(-50%)',
        }}
      >
        <ChevronLeft
          className="size-[11px]"
          style={{ transform: collapsed ? 'rotate(180deg)' : 'none', transition: 'transform 220ms var(--ease-out)' }}
        />
      </button>

      {/* Brand */}
      <Link
        href="/"
        className="brand-mark relative flex items-center justify-center"
        style={{
          minHeight: collapsed ? 56 : 76,
          padding: collapsed ? '4px 0 14px' : '8px 4px 16px',
          marginBottom: collapsed ? 8 : 10,
          borderBottom: '1px solid var(--color-line-soft)',
        }}
        aria-label="CourtCheck home"
      >
        {!collapsed ? (
          <BrandMark href={null} heightPx={56} withHoverVideo />
        ) : (
          // Collapsed: show just the checkmark webm (no wordmark room)
          <video
            muted
            playsInline
            preload="metadata"
            className="object-contain dark:invert dark:hue-rotate-180"
            style={{ maxHeight: 48, width: 'auto' }}
            onMouseEnter={(e) => {
              const v = e.currentTarget
              v.currentTime = 0
              v.play().catch(() => {})
            }}
            onMouseLeave={(e) => e.currentTarget.pause()}
          >
            <source src="/CourtCheckCheckmark.webm" type="video/webm" />
          </video>
        )}
      </Link>

      {/* Nav */}
      <nav className="sidebar-nav flex flex-col gap-[2px] flex-1">
        {NAV_ITEMS.map((item) => {
          const active = item.match(pathname)
          return (
            <Link
              key={item.name}
              href={item.href}
              title={item.name}
              aria-current={active ? 'page' : undefined}
              className={`sidebar-link relative flex items-center gap-3 min-h-[44px] rounded-[8px] px-3 text-[0.92rem] transition-colors ${
                active
                  ? 'bg-shade text-ink font-semibold'
                  : 'text-ink-soft font-medium hover:bg-shade hover:text-ink'
              } ${collapsed ? 'justify-center !px-3' : ''}`}
            >
              {active && (
                <span
                  aria-hidden="true"
                  className="absolute top-2 bottom-2 w-[3px] rounded-r-[3px] bg-court"
                  style={{ left: collapsed ? -8 : -14 }}
                />
              )}
              {item.icon}
              {!collapsed && <span className="sidebar-label">{item.name}</span>}
            </Link>
          )
        })}
      </nav>

      {/* Foot row: avatar + theme toggle (collapse handle is on the right edge of the sidebar). */}
      <div
        className={`sidebar-foot flex items-center gap-[10px] ${collapsed ? 'flex-col gap-3 py-[14px_0_4px]' : ''}`}
        style={{
          padding: collapsed ? '14px 0 4px' : '14px 6px 4px',
          borderTop: '1px solid var(--color-line-soft)',
        }}
      >
        <div ref={menuRef} className="relative">
          <button
            type="button"
            onClick={() => setMenuOpen((s) => !s)}
            aria-label="Account menu"
            aria-expanded={menuOpen}
            className="size-9 rounded-full bg-court text-cream font-display text-[0.95rem] font-medium inline-flex items-center justify-center transition-transform hover:-translate-y-[1px] cursor-pointer"
          >
            {user.initials || 'U'}
          </button>
          {menuOpen && (
            <div
              role="menu"
              className="absolute left-0 w-[248px] rounded-[12px] bg-paper border border-line shadow-pop p-[6px] z-[60]"
              style={{ bottom: 'calc(100% + 12px)' }}
            >
              <div className="flex items-center gap-[10px] p-[10px_10px_12px] min-w-0">
                <div className="size-9 rounded-full bg-court text-cream font-display text-[0.95rem] font-medium inline-flex items-center justify-center shrink-0">
                  {user.initials || 'U'}
                </div>
                <div className="flex flex-col min-w-0 gap-[1px]">
                  <div className="font-display font-medium text-[0.96rem] tracking-[-0.012em] text-ink leading-tight truncate">
                    {user.name}
                  </div>
                  <div className="font-mono text-[0.66rem] tracking-[0.04em] text-ink-mute truncate">
                    {user.email}
                  </div>
                </div>
              </div>
              <div className="h-px bg-line-soft my-1" />
              <Link
                href="/profile"
                role="menuitem"
                onClick={() => setMenuOpen(false)}
                className="flex items-center gap-[10px] w-full text-left px-[10px] py-[9px] rounded-[8px] text-[0.9rem] font-medium text-ink hover:bg-shade transition-colors"
              >
                <UserCircle className="size-[15px] text-ink-mute" />
                Profile
              </Link>
              <Link
                href="/settings"
                role="menuitem"
                onClick={() => setMenuOpen(false)}
                className="flex items-center gap-[10px] w-full text-left px-[10px] py-[9px] rounded-[8px] text-[0.9rem] font-medium text-ink hover:bg-shade transition-colors"
              >
                <Settings className="size-[15px] text-ink-mute" />
                Settings
              </Link>
              <div className="h-px bg-line-soft my-1" />
              <button
                type="button"
                role="menuitem"
                onClick={async () => {
                  setMenuOpen(false)
                  await onSignOut()
                }}
                className="flex items-center gap-[10px] w-full text-left px-[10px] py-[9px] rounded-[8px] text-[0.9rem] font-medium text-clay hover:bg-shade transition-colors cursor-pointer"
              >
                <LogOut className="size-[15px]" />
                Sign out
              </button>
            </div>
          )}
        </div>
        <ThemeToggle />
      </div>
    </aside>
  )
}
