import { NextRequest, NextResponse } from 'next/server';
import { createServerClient } from '@supabase/ssr';
import { cookies } from 'next/headers';
import { checkRateLimit, clientIp, rateLimitResponse } from '@/lib/ratelimit';

// `googleusercontent.com` (suffix) covers the rotating lh3/lh4/lh5 subdomains
// Google serves OAuth profile photos from — required for Google sign-in avatars.
const ALLOWED_HOSTS = [
  'ucdavisaggies.com',
  'images.sidearmdev.com',
  'dxbhsrqyrr690.cloudfront.net',
  'googleusercontent.com',
];
const MAX_REDIRECTS = 3;
// Hard cap on response size — prevents an attacker from chaining the allowlist
// to a multi-GB asset and burning Vercel egress. 8 MiB is plenty for player photos.
const MAX_RESPONSE_BYTES = 8 * 1024 * 1024;

function isAllowed(u: URL): boolean {
  return (
    u.protocol === 'https:' &&
    ALLOWED_HOSTS.some((h) => u.hostname === h || u.hostname.endsWith(`.${h}`))
  );
}

export async function GET(req: NextRequest) {
  // Require auth — proxy-image is only used in-app for player photos. Closing
  // this prevents the endpoint from being used as a free image proxy / amplifier.
  const cookieStore = await cookies();
  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() { return cookieStore.getAll(); },
        setAll(cookiesToSet) {
          cookiesToSet.forEach(({ name, value, options }) =>
            cookieStore.set(name, value, options)
          );
        },
      },
    }
  );
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  // Rate-limit per user: 120/hr (~2/min). Player list renders 20-30 cards
  // so a normal page view bursts ~30 images; this leaves room for re-renders
  // without throttling a legit user.
  const rl = await checkRateLimit({
    userId: user.id,
    ip: clientIp(req),
    bucket: 'proxy-image',
    limit: 120,
    windowSec: 3600,
  });
  if (!rl.ok) return rateLimitResponse(rl.retryAfterSec);

  const url = req.nextUrl.searchParams.get('url');
  if (!url) {
    return NextResponse.json({ error: 'Missing url param' }, { status: 400 });
  }

  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    return NextResponse.json({ error: 'Invalid url' }, { status: 400 });
  }

  if (parsed.protocol !== 'https:') {
    return NextResponse.json({ error: 'Only https is allowed' }, { status: 403 });
  }

  if (!ALLOWED_HOSTS.some((h) => parsed.hostname === h || parsed.hostname.endsWith(`.${h}`))) {
    return NextResponse.json({ error: 'Host not allowed' }, { status: 403 });
  }

  try {
    // redirect: 'manual' so an allowlisted host can't 302 us to an internal
    // address (e.g. http://169.254.169.254/latest/meta-data on Vercel/AWS).
    // ucdavisaggies.com legitimately 302s to its allowlisted CDN, so we follow
    // redirects but re-validate every hop against the same https+allowlist gate.
    let currentUrl = url;
    let upstream = await fetch(currentUrl, {
      headers: { 'User-Agent': 'Mozilla/5.0' },
      redirect: 'manual',
    });

    for (let hop = 0; upstream.status >= 300 && upstream.status < 400; hop++) {
      if (hop >= MAX_REDIRECTS) {
        return NextResponse.json({ error: 'Too many redirects' }, { status: 502 });
      }
      const location = upstream.headers.get('location');
      if (!location) {
        return NextResponse.json({ error: 'Redirect without location' }, { status: 502 });
      }
      let next: URL;
      try {
        next = new URL(location, currentUrl);
      } catch {
        return NextResponse.json({ error: 'Invalid redirect location' }, { status: 502 });
      }
      if (!isAllowed(next)) {
        return NextResponse.json({ error: 'Redirect target not allowed' }, { status: 403 });
      }
      currentUrl = next.toString();
      upstream = await fetch(currentUrl, {
        headers: { 'User-Agent': 'Mozilla/5.0' },
        redirect: 'manual',
      });
    }

    if (!upstream.ok) {
      return NextResponse.json({ error: 'Upstream error' }, { status: 502 });
    }

    // Refuse anything not image/* — including a missing Content-Type header.
    // Treating "no header" as a default image type would let an upstream that
    // strips its content-type bypass the gate.
    const contentType = upstream.headers.get('content-type') ?? '';
    if (!contentType.startsWith('image/')) {
      return NextResponse.json({ error: 'Upstream not an image' }, { status: 502 });
    }

    // Reject oversized responses via Content-Length first; if missing, cap via stream.
    const declared = Number(upstream.headers.get('content-length') ?? '0');
    if (declared > MAX_RESPONSE_BYTES) {
      return NextResponse.json({ error: 'Upstream too large' }, { status: 502 });
    }
    const buffer = await upstream.arrayBuffer();
    if (buffer.byteLength > MAX_RESPONSE_BYTES) {
      return NextResponse.json({ error: 'Upstream too large' }, { status: 502 });
    }

    return new NextResponse(buffer, {
      status: 200,
      headers: {
        'Content-Type': contentType,
        'Cache-Control': 'public, max-age=86400',
      },
    });
  } catch (e) {
    console.error('[proxy-image]', e);
    return NextResponse.json({ error: 'Fetch failed' }, { status: 502 });
  }
}
