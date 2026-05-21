// Onboarding endpoint.
//
// GET  /api/onboarding         → returns the available templates (preview)
// POST /api/onboarding         → body { template: 'uc-davis' | null }
//                                if 'uc-davis': clones the template players into
//                                the caller's roster.
//                                Either way: sets user_metadata.onboarded = true.

import { NextResponse } from 'next/server';
import { supabaseAdmin } from '@/lib/supabase/server';
import { createServerClient } from '@supabase/ssr';
import { cookies } from 'next/headers';
import { checkRateLimit, rateLimitResponse, clientIp } from '@/lib/ratelimit';

const TEMPLATE_KEYS = ['uc-davis'] as const;
type TemplateKey = (typeof TEMPLATE_KEYS)[number];

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
  return { user: user ?? null, supabase };
}

// Templates live as players with user_id IS NULL.
async function fetchTemplatePlayers() {
  return supabaseAdmin
    .from('players')
    .select('name, position, year, photo_url, handedness')
    .is('user_id', null)
    .order('name');
}

export async function GET() {
  const { user } = await getAuthenticatedUser();
  if (!user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const { data, error } = await fetchTemplatePlayers();
  if (error) {
    console.error('[onboarding] fetch templates', error);
    return NextResponse.json({ error: 'Failed to load templates' }, { status: 500 });
  }

  return NextResponse.json({
    templates: [
      {
        key: 'uc-davis',
        name: 'UC Davis Tennis',
        description: 'Start with the UC Davis women’s tennis roster',
        players: data ?? [],
      },
    ],
  });
}

export async function POST(req: Request) {
  const { user, supabase } = await getAuthenticatedUser();
  if (!user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  // Rate limit: a user only legitimately calls this once or twice.
  // Cap aggressively to prevent abuse if a bug ever loops it.
  const rl = await checkRateLimit({
    userId: user.id,
    ip: clientIp(req),
    bucket: 'onboarding',
    limit: 5,
    windowSec: 3600,
  });
  if (!rl.ok) return rateLimitResponse(rl.retryAfterSec);

  let body: { template?: string | null } = {};
  try {
    body = await req.json();
  } catch {
    // Empty body = treat as 'skip'
  }

  const tplInput = body.template ?? null;
  const template = TEMPLATE_KEYS.includes(tplInput as TemplateKey)
    ? (tplInput as TemplateKey)
    : null;
  if (tplInput !== null && template === null) {
    return NextResponse.json({ error: 'Unknown template key' }, { status: 400 });
  }

  let clonedCount = 0;
  if (template === 'uc-davis') {
    const { data: templatePlayers, error: fetchErr } = await fetchTemplatePlayers();
    if (fetchErr) {
      console.error('[onboarding] fetch templates for clone', fetchErr);
      return NextResponse.json({ error: 'Failed to read template' }, { status: 500 });
    }
    const rows = (templatePlayers ?? []).map((p) => ({
      name: p.name,
      position: p.position,
      year: p.year,
      photo_url: p.photo_url,
      handedness: p.handedness,
      user_id: user.id,
    }));
    if (rows.length > 0) {
      const { error: insertErr } = await supabaseAdmin.from('players').insert(rows);
      if (insertErr) {
        console.error('[onboarding] clone insert', insertErr);
        return NextResponse.json({ error: 'Failed to clone roster' }, { status: 500 });
      }
      clonedCount = rows.length;
    }
  }

  // Mark the user as onboarded. updateUser writes to user_metadata which is
  // exposed via supabase.auth.getUser().user_metadata on next call.
  const { error: updateErr } = await supabase.auth.updateUser({
    data: { onboarded: true, onboarding_template: template ?? 'empty' },
  });
  if (updateErr) {
    console.error('[onboarding] updateUser', updateErr);
    return NextResponse.json({ error: 'Failed to save onboarding state' }, { status: 500 });
  }

  return NextResponse.json({ ok: true, template, cloned: clonedCount });
}
