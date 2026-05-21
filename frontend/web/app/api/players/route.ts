import { NextResponse } from 'next/server';
import { supabaseAdmin } from '@/lib/supabase/server';
import { createServerClient } from '@supabase/ssr';
import { cookies } from 'next/headers';
import { checkRateLimit, rateLimitResponse, clientIp } from '@/lib/ratelimit';
import { isAdmin } from '@/lib/admin';

function isHttpsUrl(s: unknown): s is string {
  if (typeof s !== 'string' || s.length === 0) return false;
  try {
    return new URL(s).protocol === 'https:';
  } catch {
    return false;
  }
}

async function getAuthenticatedUser() {
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
  return user ?? null;
}

export async function GET() {
  try {
    const user = await getAuthenticatedUser();
    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    // Pre-migration safety: try with handedness, retry without if the column
    // hasn't been added yet so existing environments stay responsive.
    let { data, error } = await supabaseAdmin
      .from('players')
      .select('id, name, position, year, photo_url, handedness, created_at')
      .order('name', { ascending: true });
    if (error && typeof error.message === 'string' && error.message.includes('does not exist')) {
      const retry = await supabaseAdmin
        .from('players')
        .select('id, name, position, year, photo_url, created_at')
        .order('name', { ascending: true });
      data = retry.data as typeof data;
      error = retry.error;
    }

    if (error) {
      console.error('Players fetch error', error);
      return NextResponse.json({ error: 'Failed to fetch players' }, { status: 500 });
    }

    return NextResponse.json({ players: data ?? [] });
  } catch (e) {
    console.error(e);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}

export async function POST(req: Request) {
  try {
    const user = await getAuthenticatedUser();
    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    // Players are the shared team roster; only admins (env-configured) mutate.
    if (!isAdmin(user.email)) {
      return NextResponse.json({ error: 'Forbidden' }, { status: 403 });
    }

    const rl = await checkRateLimit({
      userId: user.id,
      ip: clientIp(req),
      bucket: 'players-post',
      limit: 10,
      windowSec: 3600,
    });
    if (!rl.ok) return rateLimitResponse(rl.retryAfterSec);

    const body = await req.json();
    const name = (body.name as string | undefined)?.trim();
    if (!name || name.length > 100) {
      return NextResponse.json({ error: 'name required (max 100 chars)' }, { status: 400 });
    }
    const handedness =
      body.handedness === 'left' ? 'left' : body.handedness === 'right' ? 'right' : null;

    // photo_url must be a real https URL or null — blocks javascript:, data:,
    // file:, and internal-network URIs from being stored and later rendered.
    let photoUrl: string | null = null;
    if (body.photo_url !== undefined && body.photo_url !== null) {
      if (!isHttpsUrl(body.photo_url)) {
        return NextResponse.json({ error: 'photo_url must be https' }, { status: 400 });
      }
      photoUrl = body.photo_url;
    }

    const insertRow: Record<string, unknown> = {
      name,
      position: typeof body.position === 'string' ? body.position.slice(0, 50) : null,
      year: typeof body.year === 'string' ? body.year.slice(0, 20) : null,
      photo_url: photoUrl,
    };
    if (handedness) insertRow.handedness = handedness;

    const { data, error } = await supabaseAdmin
      .from('players')
      .insert([insertRow])
      .select()
      .single();

    if (error) {
      console.error('Player insert error', error);
      return NextResponse.json({ error: 'Failed to create player' }, { status: 500 });
    }

    return NextResponse.json({ player: data }, { status: 201 });
  } catch (e) {
    console.error(e);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}
