import { type NextRequest } from 'next/server'
import { updateSession } from '@/lib/supabase/proxy'

export async function proxy(request: NextRequest) {
  return await updateSession(request)
}

export const config = {
  matcher: [
    // Skip Next internals, common asset extensions, and the static landing HTML
    // (auth middleware was 307-redirecting /Bounce_Animated.webm etc. to /landing).
    '/((?!_next/static|_next/image|favicon.ico|landing\\.html|.*\\.(?:svg|png|jpg|jpeg|gif|webp|webm|mp4|mp3|wav|ogg|ico|woff2?|ttf|css|js|map|html)$).*)',
  ],
}