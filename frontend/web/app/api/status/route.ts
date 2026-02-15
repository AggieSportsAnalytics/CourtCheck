// app/api/status/route.ts
import { NextResponse } from "next/server";
import { supabaseAdmin } from "@/lib/supabase/server";
import { createServerClient } from '@supabase/ssr';
import { cookies } from 'next/headers';

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

    const { data: authData } = await supabase.auth.getClaims();

    if (!authData?.claims) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const { searchParams } = new URL(req.url);
    const match_id = searchParams.get("match_id");

    if (!match_id) {
      return NextResponse.json({ error: "missing match_id" }, { status: 400 });
    }

    // Filter by user_id to ensure users can only access their own matches
    const { data, error } = await supabaseAdmin
      .from("matches")
      .select("status, results_path, progress, error, user_id")
      .eq("id", match_id)
      .eq("user_id", authData.claims.sub)
      .single();

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
      error: data.error || null,
      videoUrl,
    });
  } catch (e) {
    console.error(e);
    return NextResponse.json({ error: (e as Error).message }, { status: 500 });
  }
}