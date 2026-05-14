import { redirect } from 'next/navigation'

// The landing page is the locked brand-drop mock served verbatim from
// public/landing.html (1844 lines of design-fidelity HTML/CSS/JS). The
// React shell would re-derive what's already pixel-locked there.
export default function LandingPage() {
  redirect('/landing.html')
}
