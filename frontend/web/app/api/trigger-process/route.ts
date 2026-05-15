// app/api/trigger-process/route.ts
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

    const body = await req.json();
    const { file_key, match_id } = body;
    if (!file_key || !match_id) {
      return Response.json(
        { error: "Missing file_key or match_id" },
        { status: 400 }
      );
    }

    // Verify the match belongs to the authenticated user
    const { data: match, error: matchError } = await supabaseAdmin
      .from("matches")
      .select("user_id")
      .eq("id", match_id)
      .single();

    if (matchError || !match || match.user_id !== user.id) {
      return Response.json({ error: 'Forbidden' }, { status: 403 });
    }

    // Rate limit: max 10 processing jobs per user per hour
    const oneHourAgo = new Date(Date.now() - 60 * 60 * 1000).toISOString();
    const { count } = await supabaseAdmin
      .from("matches")
      .select("id", { count: "exact", head: true })
      .eq("user_id", user.id)
      .gte("created_at", oneHourAgo);

    if ((count ?? 0) >= 20) {
      return Response.json({ error: "Too many requests" }, { status: 429 });
    }

    // Pre-write a processing-state heartbeat BEFORE the Modal webhook fires.
    // Without this the row stays status='pending', progress=0, stage=null for
    // the whole 15-90s Modal cold-start + download + import window, and the
    // UI shows "Calibrating the court · STARTING" the entire time because it
    // falls back to a percent-derived stage label.
    try {
      await supabaseAdmin
        .from("matches")
        .update({
          status: "processing",
          progress: 0.005,
          processing_stage: "Queueing compute",
        })
        .eq("id", match_id);
    } catch (e) {
      // Defensive — if processing_stage column hasn't been migrated yet, retry
      // without it so the trigger still works on partial deploys.
      console.warn("trigger-process pre-write failed, retrying without stage", e);
      await supabaseAdmin
        .from("matches")
        .update({ status: "processing", progress: 0.005 })
        .eq("id", match_id);
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
    // return. Modal's _stage breadcrumbs (0.008 'Downloading recording',
    // 0.012 'Loading models', ...) update the row as the pipeline progresses.
    //
    // If Modal returns non-ok asynchronously, we revert the row to 'failed'
    // in the .then() callback. On Vercel this runs in the function's
    // post-response window; if the function is killed before it resolves,
    // the row stays in 'processing' and a separate cleanup job will
    // eventually mark it failed (TODO).
    fetch(process.env.MODAL_FUNCTION_URL!, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${process.env.MODAL_WEBHOOK_SECRET}`,
      },
      body: JSON.stringify({ file_key, match_id }),
    })
      .then(async (res) => {
        if (!res.ok) {
          const errorText = await res.text().catch(() => "");
          console.error("Modal returned non-ok", res.status, errorText.slice(0, 200));
          await supabaseAdmin
            .from("matches")
            .update({ status: "failed", error: "Failed to start compute" })
            .eq("id", match_id);
        }
      })
      .catch((err) => {
        console.error("Modal fetch threw", err);
      });

    return Response.json({ status: "ok" });
  } catch (e) {
    console.error(e);
    return Response.json({ error: "Internal server error" }, { status: 500 });
  }
}