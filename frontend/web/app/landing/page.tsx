import { redirect } from 'next/navigation'
import { cookies } from 'next/headers'
import { createServerClient } from '@supabase/ssr'

// The landing page is the locked brand-drop mock served verbatim from
// public/landing.html (1844 lines of design-fidelity HTML/CSS/JS). The
// React shell would re-derive what's already pixel-locked there.
//
// Guard first: an authenticated user must never be shown the marketing
// landing. AppLayout (unauth gate) and the SIGNED_OUT handler both route
// through /landing, so without this a logged-in user who hits this route
// (load race, stale link, back button) gets dumped on marketing with no
// way back. Logged in -> dashboard. Logged out -> static landing.
export default async function LandingPage() {
  const cookieStore = await cookies()
  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return cookieStore.getAll()
        },
        // Read-only in a Server Component render — session is refreshed by
        // the client SDK / auth routes, not here.
        setAll() {},
      },
    }
  )

  const {
    data: { user },
  } = await supabase.auth.getUser()

  if (user) {
    redirect('/')
  }

  redirect('/landing.html')
}
