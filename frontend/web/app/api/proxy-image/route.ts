import { NextRequest, NextResponse } from 'next/server';

const ALLOWED_HOSTS = ['ucdavisaggies.com', 'images.sidearmdev.com', 'dxbhsrqyrr690.cloudfront.net'];

export async function GET(req: NextRequest) {
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
    // The allowlisted CDNs serve images directly without redirecting, so 3xx
    // is treated as upstream failure.
    const upstream = await fetch(url, {
      headers: { 'User-Agent': 'Mozilla/5.0' },
      redirect: 'manual',
    });

    if (upstream.status >= 300 && upstream.status < 400) {
      return NextResponse.json({ error: 'Upstream redirect refused' }, { status: 502 });
    }

    if (!upstream.ok) {
      return NextResponse.json({ error: 'Upstream error' }, { status: 502 });
    }

    const contentType = upstream.headers.get('content-type') ?? 'image/webp';
    const buffer = await upstream.arrayBuffer();

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
