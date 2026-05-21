import { createServerClient } from '@supabase/ssr'
import { NextResponse, type NextRequest } from 'next/server'

export async function updateSession(request: NextRequest) {
  let response = NextResponse.next({
    request,
  })

  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return request.cookies.getAll()
        },
        setAll(cookiesToSet) {
          cookiesToSet.forEach(({ name, value }) => request.cookies.set(name, value))
          response = NextResponse.next({
            request,
          })
          cookiesToSet.forEach(({ name, value, options }) =>
            response.cookies.set(name, value, options)
          )
        },
      },
    }
  )

  // getSession() for middleware: fast cookie decode, used only for redirect logic (not data access)
  const { data: { session } } = await supabase.auth.getSession()
  const isAuthenticated = !!session

  const path = request.nextUrl.pathname
  const isAuthRoute = path.startsWith('/auth')
  const isLandingRoute = path.startsWith('/landing')
  const isOnboardingRoute = path === '/onboarding' || path.startsWith('/api/onboarding')
  const isProtectedRoute = !isAuthRoute && !isLandingRoute

  // If not authenticated and trying to access protected route, redirect to landing
  if (!isAuthenticated && isProtectedRoute) {
    return NextResponse.redirect(new URL('/landing', request.url))
  }

  // If authenticated and trying to access auth pages, redirect to dashboard
  if (isAuthenticated && isAuthRoute) {
    return NextResponse.redirect(new URL('/', request.url))
  }

  // If authenticated but not yet onboarded, force /onboarding before any
  // app surface loads. /onboarding itself and /api/onboarding are always allowed.
  if (isAuthenticated && !isOnboardingRoute && !isAuthRoute && !isLandingRoute) {
    const onboarded = (session?.user?.user_metadata as { onboarded?: boolean } | undefined)?.onboarded === true
    if (!onboarded) {
      return NextResponse.redirect(new URL('/onboarding', request.url))
    }
  }

  return response
}
