// app/api/status/route.ts
import { NextResponse } from "next/server";
import { supabaseAdmin } from "@/lib/supabase/server";
import { createServerClient } from '@supabase/ssr';
import { cookies } from 'next/headers';

// Polling endpoint — never cache the response. Without this Next.js's route
// cache + browser cache pin /api/status to its first value, so progress stays
// at the initial 0 even after Modal writes new values to Supabase.
export const dynamic = 'force-dynamic';
export const revalidate = 0;

export async function GET(req: Request) {
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
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const { searchParams } = new URL(req.url);
    const match_id = searchParams.get("match_id");

    if (!match_id) {
      return NextResponse.json({ error: "missing match_id" }, { status: 400 });
    }

    // Filter by user_id to ensure users can only access their own matches
    // Pre-migration safety: drop optional columns from the select and retry if
    // they aren't on the schema yet, so the upload flow stays responsive while
    // a migration is being applied.
    let { data, error } = await supabaseAdmin
      .from("matches")
      .select("status, results_path, progress, processing_stage, error, user_id")
      .eq("id", match_id)
      .eq("user_id", user.id)
      .single();
    if (error && typeof error.message === "string" && error.message.includes("does not exist")) {
      const retry = await supabaseAdmin
        .from("matches")
        .select("status, results_path, progress, error, user_id")
        .eq("id", match_id)
        .eq("user_id", user.id)
        .single();
      data = retry.data;
      error = retry.error;
    }

    if (error || !data) {
      return NextResponse.json({ error: "data from Supabase not found" }, { status: 404 });
    }

    let videoUrl = null;

    if (data.status === "done" && data.results_path) {
      const { data: signed, error: signErr } =
        await supabaseAdmin.storage
          .from("results")
          .createSignedUrl(data.results_path, 3600);

      if (signErr) {
        console.error("Signed URL error", signErr);
      } else {
        videoUrl = signed.signedUrl;
      }
    }

    return NextResponse.json({
      status: data.status,
      progress: data.progress || 0,
      stage: data.processing_stage || null,
      error: data.error || null,
      videoUrl,
    });
  } catch (e) {
    console.error(e);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}