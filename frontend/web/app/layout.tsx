import React from 'react'
import type { Metadata } from 'next'
import { Newsreader, Inter_Tight, JetBrains_Mono } from 'next/font/google'
import { Analytics } from '@vercel/analytics/next'
import { Toaster } from 'sonner'
import { Providers } from '@/components/Providers'
import ConditionalLayout from '@/components/layout/ConditionalLayout'
import './globals.css'

const newsreader = Newsreader({
  subsets: ['latin'],
  variable: '--font-newsreader',
  axes: ['opsz'],
  display: 'swap',
})

const interTight = Inter_Tight({
  subsets: ['latin'],
  variable: '--font-inter-tight',
  display: 'swap',
})

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-jetbrains-mono',
  display: 'swap',
})

export const metadata: Metadata = {
  title: { default: 'CourtCheck', template: '%s | CourtCheck' },
  description:
    'Tennis analytics for college coaches. Upload a match, get every shot, pattern, and percentage worth knowing.',
  icons: {
    icon: [
      { url: '/favicon.ico', sizes: 'any' },
      { url: '/favicon-16.png', type: 'image/png', sizes: '16x16' },
      { url: '/icon-32.png', type: 'image/png', sizes: '32x32' },
      { url: '/icon-192.png', type: 'image/png', sizes: '192x192' },
    ],
    apple: { url: '/apple-icon.png', sizes: '180x180' },
  },
  openGraph: {
    title: 'CourtCheck. See every shot. Know every move.',
    description:
      'Tennis analytics for college coaches. Upload a match, get every shot, pattern, and percentage worth knowing.',
    url: 'https://courtcheck-rho.vercel.app',
    siteName: 'CourtCheck',
    images: [
      {
        url: 'https://courtcheck-rho.vercel.app/CourtCheckLogoLight.png',
        width: 1200,
        height: 630,
      },
    ],
    locale: 'en_US',
    type: 'website',
  },
}

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="en"
      className={`${newsreader.variable} ${interTight.variable} ${jetbrainsMono.variable}`}
      suppressHydrationWarning
    >
      <body className="bg-cream text-ink font-sans antialiased" suppressHydrationWarning>
        <Providers>
          <ConditionalLayout>{children}</ConditionalLayout>
        </Providers>
        <Toaster
          position="bottom-center"
          richColors={false}
          closeButton
          duration={4500}
          toastOptions={{
            classNames: {
              toast: 'cc-toast',
              success: 'cc-toast-success',
              info: 'cc-toast-info',
              warning: 'cc-toast-warn',
              error: 'cc-toast-error',
            },
          }}
        />
        <Analytics />
      </body>
    </html>
  )
}
