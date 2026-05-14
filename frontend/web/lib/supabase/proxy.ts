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

  const isAuthRoute = request.nextUrl.pathname.startsWith('/auth')
  const isLandingRoute = request.nextUrl.pathname.startsWith('/landing')
  const isProtectedRoute = !isAuthRoute && !isLandingRoute

  // If not authenticated and trying to access protected route, redirect to landing
  if (!isAuthenticated && isProtectedRoute) {
    return NextResponse.redirect(new URL('/landing', request.url))
  }

  // If authenticated and trying to access auth pages, redirect to dashboard
  if (isAuthenticated && isAuthRoute) {
    return NextResponse.redirect(new URL('/', request.url))
  }

  return response
}
