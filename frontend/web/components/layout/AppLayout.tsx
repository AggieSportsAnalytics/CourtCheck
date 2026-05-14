'use client'

import { ReactNode, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import Sidebar from './Sidebar'
import BounceLoader from '@/components/upload/BounceLoader'
import { useAuth } from '@/contexts/AuthContext'

export default function AppLayout({ children }: { children: ReactNode }) {
  const { user, loading, signOut } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (!loading && !user) {
      router.replace('/landing')
    }
  }, [loading, user, router])

  useEffect(() => {
    document.body.classList.add('has-sidebar')
    return () => {
      document.body.classList.remove('has-sidebar')
    }
  }, [])

  if (loading) {
    return (
      <div className="fixed inset-0 flex items-center justify-center bg-cream">
        <div className="text-center">
          <BounceLoader size={260} />
          <p className="mt-2 text-sm text-ink-mute">Loading.</p>
        </div>
      </div>
    )
  }

  if (!user) return null

  const displayName = user.user_metadata?.name || user.email?.split('@')[0] || 'User'
  const displayEmail = user.email || ''
  const displayImage = user.user_metadata?.avatar_url || null
  const initials = displayName
    .split(' ')
    .filter(Boolean)
    .slice(0, 2)
    .map((p: string) => p[0]?.toUpperCase())
    .join('')

  return (
    <>
      <Sidebar
        user={{ name: displayName, email: displayEmail, initials, imageUrl: displayImage }}
        onSignOut={signOut}
      />
      <main className="min-h-screen">{children}</main>
    </>
  )
}
