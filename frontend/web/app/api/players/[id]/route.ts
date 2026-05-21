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
        getAll() {
          return cookieStore.getAll();
        },
        setAll(cookiesToSet) {
          cookiesToSet.forEach(({ name, value, options }) =>
            cookieStore.set(name, value, options),
          );
        },
      },
    },
  );
  const {
    data: { user },
  } = await supabase.auth.getUser();
  return user ?? null;
}

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  try {
    const user = await getAuthenticatedUser();
    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const { id } = await params;

    // Pre-migration safety: drop handedness from the select on column-missing.
    let { data, error } = await supabaseAdmin
      .from('players')
      .select('id, name, position, year, photo_url, handedness, created_at')
      .eq('id', id)
      .single();
    if (error && typeof error.message === 'string' && error.message.includes('does not exist')) {
      const retry = await supabaseAdmin
        .from('players')
        .select('id, name, position, year, photo_url, created_at')
        .eq('id', id)
        .single();
      data = retry.data as typeof data;
      error = retry.error;
    }

    if (error || !data) {
      return NextResponse.json({ error: 'Player not found' }, { status: 404 });
    }

    return NextResponse.json({ player: data });
  } catch (e) {
    console.error(e);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}

export async function PATCH(
  req: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  try {
    const user = await getAuthenticatedUser();
    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    if (!isAdmin(user.email)) {
      return NextResponse.json({ error: 'Forbidden' }, { status: 403 });
    }

    const rl = await checkRateLimit({
      userId: user.id,
      ip: clientIp(req),
      bucket: 'players-patch',
      limit: 30,
      windowSec: 3600,
    });
    if (!rl.ok) return rateLimitResponse(rl.retryAfterSec);

    const { id } = await params;
    const body = await req.json();

    const updates: Record<string, unknown> = {};
    if (typeof body.name === 'string') {
      const name = body.name.trim();
      if (!name) {
        return NextResponse.json({ error: 'name cannot be empty' }, { status: 400 });
      }
      if (name.length > 100) {
        return NextResponse.json({ error: 'name too long (max 100)' }, { status: 400 });
      }
      updates.name = name;
    }
    if (body.position !== undefined) {
      updates.position = typeof body.position === 'string' ? body.position.slice(0, 50) : null;
    }
    if (body.year !== undefined) {
      updates.year = typeof body.year === 'string' ? body.year.slice(0, 20) : null;
    }
    if (body.photo_url !== undefined) {
      if (body.photo_url === null || body.photo_url === '') {
        updates.photo_url = null;
      } else if (!isHttpsUrl(body.photo_url)) {
        return NextResponse.json({ error: 'photo_url must be https' }, { status: 400 });
      } else {
        updates.photo_url = body.photo_url;
      }
    }
    if (body.handedness !== undefined) {
      if (body.handedness !== 'left' && body.handedness !== 'right') {
        return NextResponse.json(
          { error: "handedness must be 'left' or 'right'" },
          { status: 400 },
        );
      }
      updates.handedness = body.handedness;
    }

    if (Object.keys(updates).length === 0) {
      return NextResponse.json({ error: 'Nothing to update' }, { status: 400 });
    }

    const { error } = await supabaseAdmin
      .from('players')
      .update(updates)
      .eq('id', id);

    if (error) {
      // Likely the handedness column hasn't been migrated yet — surface a
      // helpful message rather than a generic 500 so the operator can run
      // the migration without grep-spelunking.
      if (typeof error.message === 'string' && error.message.includes('does not exist')) {
        return NextResponse.json(
          { error: 'handedness column missing — apply 20260513_add_player_handedness.sql' },
          { status: 500 },
        );
      }
      console.error('Player update error', error);
      return NextResponse.json({ error: 'Failed to update player' }, { status: 500 });
    }

    return NextResponse.json(updates);
  } catch (e) {
    console.error(e);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}
