// app/api/trigger-process/route.ts
//
// Two entry paths:
//   1. First-time processing — called by useVideoUpload after the signed-upload
//      PUT lands. Body MUST include file_key + match_id (legacy contract).
//   2. Reprocess — called from the recording detail page Reprocess button. Body
//      may omit file_key; the server derives it from match.input_path and
//      verifies the raw video still exists in storage before kicking off.
//      Useful after a pipeline algorithm change (e.g. bounce SoT refactor) when
//      a coach wants to rerun without re-uploading the source.

import { createServerClient } from '@supabase/ssr';
import { cookies } from 'next/headers';
import { supabaseAdmin } from '@/lib/supabase/server';

export async function POST(req: Request) {
  try {
    // Check authentication
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
              cookieStore.set(name, value, options)
            );
          },
        },
      }
    );

    const { data: { user } } = await supabase.auth.getUser();

    if (!user) {
      return Response.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const body = await req.json().catch(() => ({}));
    const { match_id } = body;
    let { file_key } = body;
    if (!match_id) {
      return Response.json({ error: 'Missing match_id' }, { status: 400 });
    }

    // Fetch ownership + input_path. Server derives file_key when missing.
    const { data: match, error: matchError } = await supabaseAdmin
      .from('matches')
      .select('user_id, input_path, status')
      .eq('id', match_id)
      .single();

    if (matchError || !match || match.user_id !== user.id) {
      return Response.json({ error: 'Forbidden' }, { status: 403 });
    }

    if (!file_key) {
      if (!match.input_path) {
        return Response.json(
          { error: 'Raw video no longer in storage — re-upload to reprocess.' },
          { status: 409 },
        );
      }
      file_key = match.input_path as string;
    }

    if (match.status === 'processing') {
      return Response.json(
        { error: 'Already processing — wait for the current run to finish.' },
        { status: 409 },
      );
    }

    // Verify the raw video still exists in storage. After a successful first
    // run we auto-delete raws (DELETE_RAW_AFTER_PROCESS), so a reprocess on a
    // long-finished match will hit this — give the user a clear error rather
    // than a silent Modal failure 5 min later.
    if (match.status === 'done' || match.status === 'failed') {
      const lastSlash = (file_key as string).lastIndexOf('/');
      const dir = lastSlash > 0 ? (file_key as string).slice(0, lastSlash) : '';
      const name = lastSlash >= 0 ? (file_key as string).slice(lastSlash + 1) : (file_key as string);
      const { data: listing, error: listErr } = await supabaseAdmin.storage
        .from('raw-videos')
        .list(dir, { search: name, limit: 1 });
      const exists = !listErr && Array.isArray(listing) && listing.some((f) => f.name === name);
      if (!exists) {
        return Response.json(
          { error: 'Raw video no longer in storage — re-upload to reprocess.' },
          { status: 409 },
        );
      }
    }

    // Rate limit: max 20 processing jobs per user per hour (counts ALL match
    // rows the user created in the window — covers both fresh uploads and
    // reprocess triggers since both bump load on Modal).
    const oneHourAgo = new Date(Date.now() - 60 * 60 * 1000).toISOString();
    const { count } = await supabaseAdmin
      .from('matches')
      .select('id', { count: 'exact', head: true })
      .eq('user_id', user.id)
      .gte('created_at', oneHourAgo);

    if ((count ?? 0) >= 20) {
      return Response.json({ error: 'Too many requests' }, { status: 429 });
    }

    // Pre-write a processing-state heartbeat BEFORE the Modal webhook fires.
    // Without this the row stays at its previous status until Modal's first
    // breadcrumb lands 15-90s later. For reprocess this also clears stale
    // error/progress fields from the previous run so the UI doesn't show
    // a confusing mix of old and new state.
    try {
      await supabaseAdmin
        .from('matches')
        .update({
          status: 'processing',
          progress: 0.005,
          processing_stage: 'Queueing compute',
          error: null,
        })
        .eq('id', match_id);
    } catch (e) {
      console.warn('trigger-process pre-write failed, retrying without stage', e);
      await supabaseAdmin
        .from('matches')
        .update({ status: 'processing', progress: 0.005, error: null })
        .eq('id', match_id);
    }

    // Fire Modal webhook but DO NOT await the response. Modal's
    // @modal.fastapi_endpoint binds the HTTP connection to the function
    // execution — process_video blocks for the entire pipeline (~5 min).
    // Awaiting it stalls /api/trigger-process from returning to the client,
    // which means pollStatus on the upload page never starts until the
    // pipeline is done. That's the "STARTING for the whole upload" bug.
    //
    // The pre-write above (progress=0.005, stage='Queueing compute') is
    // already in the DB, so polling has something to read the moment we
    // return.
    fetch(process.env.MODAL_FUNCTION_URL!, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.MODAL_WEBHOOK_SECRET}`,
      },
      body: JSON.stringify({ file_key, match_id }),
    })
      .then(async (res) => {
        if (!res.ok) {
          const errorText = await res.text().catch(() => '');
          console.error('Modal returned non-ok', res.status, errorText.slice(0, 200));
          await supabaseAdmin
            .from('matches')
            .update({ status: 'failed', error: 'Failed to start compute' })
            .eq('id', match_id);
        }
      })
      .catch((err) => {
        console.error('Modal fetch threw', err);
      });

    return Response.json({ status: 'ok' });
  } catch (e) {
    console.error(e);
    return Response.json({ error: 'Internal server error' }, { status: 500 });
  }
}
