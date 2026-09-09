import { NextResponse } from "next/server";
import { supabaseAdmin } from "@/lib/supabase/server";
import { createServerClient } from '@supabase/ssr';
import { cookies } from 'next/headers';
import { checkRateLimit, rateLimitResponse, clientIp } from '@/lib/ratelimit';

// Polled by the recording detail page every 5s while a recording is processing.
// Force-dynamic so Next.js doesn't pin progress/stage/status to a cached value.
export const dynamic = 'force-dynamic';
export const revalidate = 0;

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

    const { data: { user } } = await supabase.auth.getUser();

    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const { id } = await params;

    const FULL_COLS = "id, name, status, progress, processing_stage, error, results_path, input_path, created_at, fps, num_frames, bounce_heatmap_path, player_heatmap_path, player_shot_map_path, bounce_count, shot_count, rally_count, forehand_count, backhand_count, serve_count, in_bounds_bounces, out_bounds_bounces, scouting_report, player_id, favorited, keypoints, notes, shots, coverage_grid, position_summary, net_approach_summary, error_summary, rallies, rally_summary";
    // Pre-migration safety: if a column the API references hasn't been added
    // to `matches` yet, Postgres returns "column matches.<name> does not
    // exist". Retry up to N times, each time stripping ONLY the column named
    // in the error so previously-migrated optional columns keep flowing.
    // Earlier versions stripped a hard-coded OPTIONAL_COLS list on the first
    // miss, which silently nulled out summaries the UI depended on (errors,
    // net approach) the moment a newer column landed mid-deploy.
    const OPTIONAL_COLS = new Set([
      "processing_stage",
      "favorited",
      "shots",
      "coverage_grid",
      "position_summary",
      "net_approach_summary",
      "error_summary",
      "rallies",
      "rally_summary",
    ]);
    let { data, error } = await supabaseAdmin
      .from("matches")
      .select(FULL_COLS)
      .eq("id", id)
      .eq("user_id", user.id)
      .single();
    // If a column is missing, retry — but strip ONLY the column named in the
    // error so previously-migrated optionals (error_summary, etc.) keep
    // flowing. Earlier versions stripped a hard-coded OPTIONAL_COLS list on
    // the first miss, which silently nulled out summaries the UI depended on
    // the moment a newer column landed mid-deploy.
    let activeCols = FULL_COLS.split(", ");
    let attempt = 0;
    while (error && typeof error.message === "string" && error.message.includes("does not exist") && attempt < OPTIONAL_COLS.size) {
      const msg = error.message;
      const match =
        msg.match(/column [^.\s]+\.(\w+)/i) ||
        msg.match(/column ['"`]?(\w+)['"`]?/i) ||
        msg.match(/'([a-z_][\w]*)' column/i);
      const missing = match?.[1];
      if (!missing || !OPTIONAL_COLS.has(missing) || !activeCols.includes(missing)) break;
      activeCols = activeCols.filter((c) => c !== missing);
      const retry = await supabaseAdmin
        .from("matches")
        .select(activeCols.join(", "))
        .eq("id", id)
        .eq("user_id", user.id)
        .single();
      // Dynamic select() string can't be inferred by postgrest-js. Runtime
      // shape is a strict subset of FULL_COLS, so downstream `data.x ?? null`
      // accessors stay safe.
      data = retry.data as typeof data;
      error = retry.error;
      attempt += 1;
    }

    if (error || !data) {
      return NextResponse.json({ error: "Recording not found" }, { status: 404 });
    }

    // Pull the near player's handedness so the UI can surface it as a badge.
    // Defaults to 'right' (no badge shown frontside) if the player isn't bound
    // or the handedness column hasn't been migrated yet.
    let playerHandedness: 'right' | 'left' | null = null;
    if (data.player_id) {
      const handednessRes = await supabaseAdmin
        .from("players")
        .select("handedness")
        .eq("id", data.player_id)
        .single();
      const h = (handednessRes.data as { handedness?: string } | null)?.handedness;
      if (h === 'left' || h === 'right') {
        playerHandedness = h;
      }
    }

    let videoUrl = null;
    let bounceHeatmapUrl = null;
    let playerHeatmapUrl = null;
    let playerShotMapUrl = null;
    if (data.status === "done") {
      const [videoSigned, bounceSigned, playerSigned, shotMapSigned] = await Promise.all([
        data.results_path
          ? supabaseAdmin.storage.from("results").createSignedUrl(data.results_path, 3600)
          : Promise.resolve({ data: null }),
        data.bounce_heatmap_path
          ? supabaseAdmin.storage.from("results").createSignedUrl(data.bounce_heatmap_path, 3600)
          : Promise.resolve({ data: null }),
        data.player_heatmap_path
          ? supabaseAdmin.storage.from("results").createSignedUrl(data.player_heatmap_path, 3600)
          : Promise.resolve({ data: null }),
        data.player_shot_map_path
          ? supabaseAdmin.storage.from("results").createSignedUrl(data.player_shot_map_path, 3600)
          : Promise.resolve({ data: null }),
      ]);
      videoUrl = videoSigned.data?.signedUrl ?? null;
      bounceHeatmapUrl = bounceSigned.data?.signedUrl ?? null;
      playerHeatmapUrl = playerSigned.data?.signedUrl ?? null;
      playerShotMapUrl = shotMapSigned.data?.signedUrl ?? null;
    }

    return NextResponse.json({
      recording: {
        id: data.id,
        status: data.status,
        progress: data.progress || 0,
        stage: data.processing_stage || null,
        error: data.error || null,
        videoUrl,
        bounceHeatmapUrl,
        playerHeatmapUrl,
        playerShotMapUrl,
        createdAt: data.created_at,
        name: data.name || data.input_path?.split("/").pop() || "Unknown",
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
        favorited: data.favorited ?? false,
        playerId: data.player_id ?? null,
        playerHandedness,
        keypoints: data.keypoints ?? [],
        notes: data.notes ?? [],
        shots: Array.isArray(data.shots) ? data.shots : [],
        coverageGrid: Array.isArray(data.coverage_grid) ? data.coverage_grid : [],
        positionSummary: data.position_summary && typeof data.position_summary === "object" ? data.position_summary : null,
        netApproachSummary: data.net_approach_summary && typeof data.net_approach_summary === "object" ? data.net_approach_summary : null,
        errorSummary: data.error_summary && typeof data.error_summary === "object" ? data.error_summary : null,
        rallies: Array.isArray(data.rallies) ? data.rallies : [],
        rallySummary: data.rally_summary && typeof data.rally_summary === "object" ? data.rally_summary : null,
      },
    });
  } catch (e) {
    console.error(e);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}

export async function PATCH(
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

    const { data: { user } } = await supabase.auth.getUser();

    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const rl = await checkRateLimit({
      userId: user.id,
      ip: clientIp(req),
      bucket: 'recordings-patch',
      limit: 30,
      windowSec: 3600,
    });
    if (!rl.ok) return rateLimitResponse(rl.retryAfterSec);

    const { id } = await params;
    const body = await req.json();

    const updates: Record<string, unknown> = {};

    if (typeof body.name === 'string') {
      let name = body.name.trim().replace(/[\x00-\x1F\x7F]/g, '');
      if (!name) return NextResponse.json({ error: 'Name cannot be empty' }, { status: 400 });
      if (name.length > 100) return NextResponse.json({ error: 'Name too long (max 100 characters)' }, { status: 400 });
      updates.name = name;
    }

    if (typeof body.favorited === 'boolean') {
      updates.favorited = body.favorited;
    }

    if (Array.isArray(body.notes)) {
      const validNotes = body.notes.every(
        (n: unknown) =>
          typeof n === 'object' && n !== null &&
          typeof (n as { timestamp_sec: number }).timestamp_sec === 'number' &&
          typeof (n as { text: string }).text === 'string'
      );
      if (!validNotes) return NextResponse.json({ error: 'Invalid notes format' }, { status: 400 });
      updates.notes = body.notes.map((n: { timestamp_sec: number; text: string }) => ({
        timestamp_sec: n.timestamp_sec,
        text: n.text.slice(0, 1000),
      }));
    }

    if (Array.isArray(body.keypoints)) {
      // Validate each keypoint: { type, timestamp_sec }
      const valid = body.keypoints.every(
        (k: unknown) =>
          typeof k === 'object' && k !== null &&
          ['set_start', 'side_switch', 'cut'].includes((k as { type: string }).type) &&
          typeof (k as { timestamp_sec: number }).timestamp_sec === 'number'
      );
      if (!valid) return NextResponse.json({ error: 'Invalid keypoints format' }, { status: 400 });
      updates.keypoints = body.keypoints;
    }

    if (Object.keys(updates).length === 0) {
      return NextResponse.json({ error: 'Nothing to update' }, { status: 400 });
    }

    const { error } = await supabaseAdmin
      .from("matches")
      .update(updates)
      .eq("id", id)
      .eq("user_id", user.id);

    if (error) {
      console.error("PATCH error", error);
      return NextResponse.json({ error: "Failed to update" }, { status: 500 });
    }

    return NextResponse.json(updates);
  } catch (e) {
    console.error(e);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}

export async function DELETE(
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

    const { data: { user } } = await supabase.auth.getUser();
    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    // Deletes destroy data + storage objects; cap per-user to slow abuse.
    const rl = await checkRateLimit({
      userId: user.id,
      ip: clientIp(req),
      bucket: 'recordings-delete',
      limit: 10,
      windowSec: 3600,
    });
    if (!rl.ok) return rateLimitResponse(rl.retryAfterSec);

    const { id } = await params;

    // Fetch row first to get storage paths and verify ownership
    const { data, error: fetchError } = await supabaseAdmin
      .from("matches")
      .select("id, input_path, results_path, bounce_heatmap_path, player_heatmap_path, player_shot_map_path")
      .eq("id", id)
      .eq("user_id", user.id)
      .single();

    if (fetchError || !data) {
      return NextResponse.json({ error: "Recording not found" }, { status: 404 });
    }

    // Delete DB row first — if this fails, storage files remain intact (recoverable)
    const { error: deleteError } = await supabaseAdmin
      .from("matches")
      .delete()
      .eq("id", id)
      .eq("user_id", user.id);

    if (deleteError) {
      console.error("Delete error", deleteError);
      return NextResponse.json({ error: "Failed to delete" }, { status: 500 });
    }

    // DB row is gone — clean up storage files (non-blocking: log errors but don't return 500)
    if (data.input_path) {
      const { error: rawError } = await supabaseAdmin.storage
        .from("raw-videos")
        .remove([data.input_path]);
      if (rawError) {
        console.error("Storage cleanup error (raw-videos):", rawError);
      }
    }

    const resultsPaths = [
      data.results_path,
      data.bounce_heatmap_path,
      data.player_heatmap_path,
      data.player_shot_map_path,
    ].filter(Boolean) as string[];

    if (resultsPaths.length > 0) {
      const { error: resultsError } = await supabaseAdmin.storage
        .from("results")
        .remove(resultsPaths);
      if (resultsError) {
        console.error("Storage cleanup error (results):", resultsError);
      }
    }

    return new NextResponse(null, { status: 204 });
  } catch (e) {
    console.error(e);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
