import { NextResponse } from 'next/server';
import { supabaseAdmin } from '@/lib/supabase/server';
import { createServerClient } from '@supabase/ssr';
import { cookies } from 'next/headers';
import { checkRateLimit, rateLimitResponse, clientIp } from '@/lib/ratelimit';

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

// Fetch a player row by id, returning the user_id so callers can check ownership.
async function fetchPlayerForOwnershipCheck(id: string) {
  return supabaseAdmin
    .from('players')
    .select('id, user_id')
    .eq('id', id)
    .single();
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

    const includeTemplates = !user.user_metadata?.onboarding_template;
    const ownerFilter = includeTemplates ? `user_id.is.null,user_id.eq.${user.id}` : `user_id.eq.${user.id}`;
    let { data, error } = await supabaseAdmin
      .from('players')
      .select('id, name, position, year, photo_url, handedness, user_id, created_at')
      .eq('id', id)
      .or(ownerFilter)
      .single();
    if (error && typeof error.message === 'string' && error.message.includes('does not exist')) {
      const retry = await supabaseAdmin
        .from('players')
        .select('id, name, position, year, photo_url, user_id, created_at')
        .eq('id', id)
        .or(ownerFilter)
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

    const rl = await checkRateLimit({
      userId: user.id,
      ip: clientIp(req),
      bucket: 'players-patch',
      limit: 30,
      windowSec: 3600,
    });
    if (!rl.ok) return rateLimitResponse(rl.retryAfterSec);

    const { id } = await params;

    // Ownership check: only the row's owner can edit. Template rows (user_id null)
    // are read-only for everyone. Returns 404 (not 403) for non-owned rows
    // to avoid leaking existence of other users' players.
    const { data: ownerRow, error: ownerErr } = await fetchPlayerForOwnershipCheck(id);
    if (ownerErr || !ownerRow) {
      return NextResponse.json({ error: 'Player not found' }, { status: 404 });
    }
    if (ownerRow.user_id !== user.id) {
      return NextResponse.json({ error: 'Player not found' }, { status: 404 });
    }

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

    // Defense in depth: re-assert ownership in the UPDATE itself so a race
    // between the fetch above and the write can't strip a row from another
    // user (the .eq('user_id', ...) means a mismatched row updates zero rows).
    const { error } = await supabaseAdmin
      .from('players')
      .update(updates)
      .eq('id', id)
      .eq('user_id', user.id);

    if (error) {
      if (typeof error.message === 'string' && error.message.includes('does not exist')) {
        console.error('Handedness column missing. Apply 20260513_add_player_handedness.sql', error);
        return NextResponse.json(
          { error: 'Could not save handedness. Try again in a moment.' },
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

export async function DELETE(
  req: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  try {
    const user = await getAuthenticatedUser();
    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    // Cascading delete destroys recordings + their storage objects; cap per-user.
    const rl = await checkRateLimit({
      userId: user.id,
      ip: clientIp(req),
      bucket: 'players-delete',
      limit: 10,
      windowSec: 3600,
    });
    if (!rl.ok) return rateLimitResponse(rl.retryAfterSec);

    const { id } = await params;

    // Ownership check: only the row's owner can delete. Template rows (user_id null)
    // are not deletable. 404 (not 403) for non-owned rows to avoid leaking
    // existence of other users' players.
    const { data: ownerRow, error: ownerErr } = await fetchPlayerForOwnershipCheck(id);
    if (ownerErr || !ownerRow) {
      return NextResponse.json({ error: 'Player not found' }, { status: 404 });
    }
    if (ownerRow.user_id !== user.id) {
      return NextResponse.json({ error: 'Player not found' }, { status: 404 });
    }

    // Cascade: the caller chose to delete this player's recordings too. The
    // matches.player_id FK has no ON DELETE rule, so the matches rows must go
    // before the player row or the final delete fails with a FK violation.
    // Fetch storage paths first so we can clean up the buckets afterward.
    const { data: matches, error: matchErr } = await supabaseAdmin
      .from('matches')
      .select(
        'id, input_path, results_path, bounce_heatmap_path, player_heatmap_path, player_shot_map_path',
      )
      .eq('player_id', id)
      .eq('user_id', user.id);

    if (matchErr) {
      console.error('Player delete: match lookup error', matchErr);
      return NextResponse.json({ error: 'Failed to delete player' }, { status: 500 });
    }

    if (matches && matches.length > 0) {
      const { error: matchDeleteErr } = await supabaseAdmin
        .from('matches')
        .delete()
        .eq('player_id', id)
        .eq('user_id', user.id);

      if (matchDeleteErr) {
        console.error('Player delete: match delete error', matchDeleteErr);
        return NextResponse.json({ error: 'Failed to delete recordings' }, { status: 500 });
      }

      // Rows are gone — clean up storage. Non-blocking: log but don't fail the
      // request, since the DB (source of truth) is already consistent.
      const rawPaths = matches
        .map((m) => m.input_path)
        .filter(Boolean) as string[];
      if (rawPaths.length > 0) {
        const { error: rawError } = await supabaseAdmin.storage
          .from('raw-videos')
          .remove(rawPaths);
        if (rawError) console.error('Storage cleanup error (raw-videos):', rawError);
      }

      const resultsPaths = matches
        .flatMap((m) => [
          m.results_path,
          m.bounce_heatmap_path,
          m.player_heatmap_path,
          m.player_shot_map_path,
        ])
        .filter(Boolean) as string[];
      if (resultsPaths.length > 0) {
        const { error: resultsError } = await supabaseAdmin.storage
          .from('results')
          .remove(resultsPaths);
        if (resultsError) console.error('Storage cleanup error (results):', resultsError);
      }
    }

    // Re-assert ownership in the delete itself (defense in depth against a race
    // between the ownership fetch above and this write).
    const { error: delErr } = await supabaseAdmin
      .from('players')
      .delete()
      .eq('id', id)
      .eq('user_id', user.id);

    if (delErr) {
      console.error('Player delete error', delErr);
      return NextResponse.json({ error: 'Failed to delete player' }, { status: 500 });
    }

    return new NextResponse(null, { status: 204 });
  } catch (e) {
    console.error(e);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}
