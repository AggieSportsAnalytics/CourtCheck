import React from "react"
import type { Metadata } from 'next'
import { Geist, Geist_Mono } from 'next/font/google'
import { Analytics } from '@vercel/analytics/next'
import { Providers } from '@/components/Providers'
import ConditionalLayout from '@/components/layout/ConditionalLayout'
import './globals.css'

const _geist = Geist({ subsets: ["latin"] });
const _geistMono = Geist_Mono({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: 'CourtCheck - Tennis Analytics Dashboard',
  description: 'Tennis analytics and performance tracking dashboard',
  generator: 'v0.app',
  icons: {
    icon: [
      {
        url: '/icon-light-32x32.png',
        media: '(prefers-color-scheme: light)',
      },
      {
        url: '/icon-dark-32x32.png',
        media: '(prefers-color-scheme: dark)',
      },
],
    apple: '/apple-icon.png',
  },
  openGraph: {
    title: 'CourtCheck - Tennis Analytics Platform',
    description: 'AI-powered tennis analytics platform for match analysis, performance metrics, and scouting insights.',
    url: 'https://courtcheck-rho.vercel.app',
    siteName: 'CourtCheck',
    images: [
      {
        url: 'https://courtcheck-rho.vercel.app/courtcheck_logo.png',
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
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body className="font-sans antialiased bg-primary text-white" suppressHydrationWarning>
        <Providers>
          <ConditionalLayout>{children}</ConditionalLayout>
        </Providers>
        <Analytics />
      </body>
    </html>
  )
}
