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

    const { data: { user } } = await supabase.auth.getUser();

    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const BASE_COLS = "id, name, status, progress, error, results_path, input_path, created_at, fps, num_frames, bounce_count, shot_count, rally_count, forehand_count, backhand_count, serve_count, in_bounds_bounces, out_bounds_bounces, bounce_heatmap_path, player_heatmap_path, player_id";
    const listQuery = (cols: string) =>
      supabaseAdmin
        .from("matches")
        .select(cols)
        .eq("user_id", user.id)
        .order("created_at", { ascending: false })
        .limit(100);

    let { data, error } = await listQuery(`${BASE_COLS}, favorited`);
    // Pre-migration safety: if the `favorited` column hasn't been added yet,
    // retry without it so the list still loads (favorited defaults to false).
    if (error && typeof error.message === "string" && error.message.includes("favorited")) {
      ({ data, error } = await listQuery(BASE_COLS));
    }

    if (error) {
      console.error("Recordings fetch error", error);
      return NextResponse.json({ error: "Failed to fetch recordings" }, { status: 500 });
    }

    // Dynamic select() string can't be inferred by postgrest-js, so it widens
    // to GenericStringError. Runtime shape is BASE_COLS (+ optional favorited).
    type MatchRow = {
      id: string;
      name: string | null;
      status: string;
      progress: number | null;
      error: string | null;
      results_path: string | null;
      input_path: string | null;
      created_at: string;
      fps: number | null;
      num_frames: number | null;
      bounce_count: number | null;
      shot_count: number | null;
      rally_count: number | null;
      forehand_count: number | null;
      backhand_count: number | null;
      serve_count: number | null;
      in_bounds_bounces: number | null;
      out_bounds_bounces: number | null;
      bounce_heatmap_path: string | null;
      player_heatmap_path: string | null;
      player_id: string | null;
      favorited?: boolean | null;
    };
    const rows = (data ?? []) as unknown as MatchRow[];

    // Pull player names for any matches with a bound player_id. One query
    // instead of N+1 — collect unique IDs, fetch once, build a lookup map.
    // The recordings list filters/displays by REAL player name (not parsed
    // off the recording title), so the page needs both fields.
    const playerIds = Array.from(
      new Set(
        rows
          .map((m) => m.player_id)
          .filter((x): x is string => typeof x === 'string' && x.length > 0),
      ),
    );
    const playerMap: Record<string, string> = {};
    if (playerIds.length > 0) {
      const { data: playerRows, error: playerErr } = await supabaseAdmin
        .from("players")
        .select("id, name")
        .in("id", playerIds);
      if (playerErr) {
        // Non-fatal — fall back to no player names; UI will show "—".
        console.warn("recordings: player join failed", playerErr);
      } else {
        for (const p of (playerRows || []) as { id: string; name: string }[]) {
          playerMap[p.id] = p.name;
        }
      }
    }

    // Generate signed URLs for completed matches
    const recordings = await Promise.all(
      rows.map(async (match) => {
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
          name: match.name || match.input_path?.split("/").pop() || "Unknown",
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
          player_id: match.player_id ?? null,
          playerName: match.player_id ? (playerMap[match.player_id] ?? null) : null,
          favorited: match.favorited ?? false,
        };
      })
    );

    return NextResponse.json({ recordings });
  } catch (e) {
    console.error(e);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
