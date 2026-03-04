import { NextResponse } from "next/server";
import { supabaseAdmin } from "@/lib/supabase/server";
import { createServerClient } from '@supabase/ssr';
import { cookies } from 'next/headers';

export async function GET() {
  try {
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

    const { data, error } = await supabaseAdmin
      .from("matches")
      .select("id, status, progress, error, results_path, input_path, created_at, fps, num_frames, bounce_count, shot_count, rally_count, forehand_count, backhand_count, serve_count, in_bounds_bounces, out_bounds_bounces, bounce_heatmap_path, player_heatmap_path")
      .eq("user_id", authData.claims.sub)
      .order("created_at", { ascending: false });

    if (error) {
      console.error("Recordings fetch error", error);
      return NextResponse.json({ error: "Failed to fetch recordings" }, { status: 500 });
    }

    // Generate signed URLs for completed matches
    const recordings = await Promise.all(
      (data || []).map(async (match) => {
        let videoUrl = null;

        if (match.status === "done" && match.results_path) {
          const { data: signed } = await supabaseAdmin.storage
            .from("results")
            .createSignedUrl(match.results_path, 3600);

          videoUrl = signed?.signedUrl ?? null;
        }

        return {
          id: match.id,
          status: match.status,
          progress: match.progress || 0,
          error: match.error || null,
          videoUrl,
          createdAt: match.created_at,
          filename: match.input_path?.split("/").pop() || "Unknown",
          fps: match.fps,
          numFrames: match.num_frames,
          bounceCount:      match.bounce_count      ?? null,
          shotCount:        match.shot_count        ?? null,
          rallyCount:       match.rally_count       ?? null,
          forehandCount:    match.forehand_count    ?? null,
          backhandCount:    match.backhand_count    ?? null,
          serveCount:       match.serve_count       ?? null,
          inBoundsBounces:  match.in_bounds_bounces  ?? null,
          outBoundsBounces: match.out_bounds_bounces ?? null,
          hasBounceHeatmap: !!match.bounce_heatmap_path,
          hasPlayerHeatmap: !!match.player_heatmap_path,
        };
      })
    );

    return NextResponse.json({ recordings });
  } catch (e) {
    console.error(e);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
