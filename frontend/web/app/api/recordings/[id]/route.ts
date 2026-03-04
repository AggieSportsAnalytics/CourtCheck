import { NextResponse } from "next/server";
import { supabaseAdmin } from "@/lib/supabase/server";
import { createServerClient } from '@supabase/ssr';
import { cookies } from 'next/headers';

export async function GET(
  req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
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

    const { id } = await params;

    const { data, error } = await supabaseAdmin
      .from("matches")
      .select("id, status, progress, error, results_path, input_path, created_at, fps, num_frames, bounce_heatmap_path, player_heatmap_path, bounce_count, shot_count, rally_count, forehand_count, backhand_count, serve_count, in_bounds_bounces, out_bounds_bounces, scouting_report")
      .eq("id", id)
      .eq("user_id", authData.claims.sub)
      .single();

    if (error || !data) {
      return NextResponse.json({ error: "Recording not found" }, { status: 404 });
    }

    let videoUrl = null;
    if (data.status === "done" && data.results_path) {
      const { data: signed } = await supabaseAdmin.storage
        .from("results")
        .createSignedUrl(data.results_path, 3600);
      videoUrl = signed?.signedUrl ?? null;
    }

    let bounceHeatmapUrl = null;
    let playerHeatmapUrl = null;
    if (data.status === "done") {
      if (data.bounce_heatmap_path) {
        const { data: signed } = await supabaseAdmin.storage
          .from("results")
          .createSignedUrl(data.bounce_heatmap_path, 3600);
        bounceHeatmapUrl = signed?.signedUrl ?? null;
      }
      if (data.player_heatmap_path) {
        const { data: signed } = await supabaseAdmin.storage
          .from("results")
          .createSignedUrl(data.player_heatmap_path, 3600);
        playerHeatmapUrl = signed?.signedUrl ?? null;
      }
    }

    return NextResponse.json({
      recording: {
        id: data.id,
        status: data.status,
        progress: data.progress || 0,
        error: data.error || null,
        videoUrl,
        bounceHeatmapUrl,
        playerHeatmapUrl,
        createdAt: data.created_at,
        filename: data.input_path?.split("/").pop() || "Unknown",
        fps: data.fps,
        numFrames: data.num_frames,
        bounceCount: data.bounce_count ?? null,
        shotCount: data.shot_count ?? null,
        rallyCount: data.rally_count ?? null,
        forehandCount: data.forehand_count ?? null,
        backhandCount: data.backhand_count ?? null,
        serveCount: data.serve_count ?? null,
        inBoundsBounces: data.in_bounds_bounces ?? null,
        outBoundsBounces: data.out_bounds_bounces ?? null,
        scoutingReport: data.scouting_report ?? null,
      },
    });
  } catch (e) {
    console.error(e);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
