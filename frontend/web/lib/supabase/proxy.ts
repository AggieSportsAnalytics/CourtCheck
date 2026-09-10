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
  const isApiRoute = path.startsWith('/api/')
  const isAuthRoute = path.startsWith('/auth')
  const isLandingRoute = path.startsWith('/landing')
  const isOnboardingRoute = path === '/onboarding' || path.startsWith('/api/onboarding')
  const isProtectedRoute = !isAuthRoute && !isLandingRoute

  const onboarded = session?.user?.user_metadata?.onboarded === true
  const redirectWithCookies = (path: string) => {
    const redirect = NextResponse.redirect(new URL(path, request.url))
    response.cookies.getAll().forEach(c => redirect.cookies.set(c))
    return redirect
  }

  if (!isAuthenticated && isApiRoute) {
    const unauthorized = NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
    response.cookies.getAll().forEach(c => unauthorized.cookies.set(c))
    return unauthorized
  }

  if (isAuthenticated && onboarded && path === '/onboarding') {
    return redirectWithCookies('/')
  }

  // If not authenticated and trying to access protected route, redirect to landing
  if (!isAuthenticated && isProtectedRoute) {
    return redirectWithCookies('/landing')
  }

  // If authenticated and trying to access auth pages, redirect to dashboard
  if (isAuthenticated && isAuthRoute && path !== '/auth/update-password') {
    return redirectWithCookies('/')
  }

  // If authenticated but not yet onboarded, force /onboarding before any
  // app surface loads. /onboarding itself and /api/onboarding are always allowed.
  if (isAuthenticated && !isApiRoute && !isOnboardingRoute && !isAuthRoute && !isLandingRoute) {
    if (!onboarded) {
      return redirectWithCookies('/onboarding')
    }
  }

  return response
}
