import { NextResponse } from 'next/server';
import { supabaseAdmin } from '@/lib/supabase/server';
import { createServerClient } from '@supabase/ssr';
import { cookies } from 'next/headers';

function safeDurationSeconds(fps: number | null, numFrames: number | null) {
  if (!fps || !numFrames || fps <= 0 || numFrames <= 0) return null;
  return numFrames / fps;
}

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
            cookiesToSet.forEach(({ name, value, options }) => cookieStore.set(name, value, options));
          },
        },
      }
    );

    const { data: authData } = await supabase.auth.getClaims();
    if (!authData?.claims) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const { data, error } = await supabaseAdmin
      .from('matches')
      .select('id, status, progress, error, created_at, fps, num_frames, bounce_heatmap_path, player_heatmap_path, bounce_count, shot_count, rally_count, forehand_count, backhand_count, serve_count, in_bounds_bounces, out_bounds_bounces')
      .eq('user_id', authData.claims.sub)
      .order('created_at', { ascending: false })
      .limit(50);

    if (error) {
      console.error('Dashboard summary fetch error', error);
      return NextResponse.json({ error: error.message }, { status: 500 });
    }

    const matches = data ?? [];
    const doneMatches = matches.filter((m) => m.status === 'done');

    const totals = {
      total: matches.length,
      done: doneMatches.length,
      processing: matches.filter((m) => m.status === 'processing').length,
      failed: matches.filter((m) => m.status === 'failed').length,
    };

    // Aggregated tennis stats across ALL done matches
    const tennisStats = doneMatches.reduce(
      (acc, m) => ({
        totalBounces:    acc.totalBounces    + (m.bounce_count     ?? 0),
        totalShots:      acc.totalShots      + (m.shot_count       ?? 0),
        totalRallies:    acc.totalRallies    + (m.rally_count      ?? 0),
        totalForehands:  acc.totalForehands  + (m.forehand_count   ?? 0),
        totalBackhands:  acc.totalBackhands  + (m.backhand_count   ?? 0),
        totalServes:     acc.totalServes     + (m.serve_count      ?? 0),
        totalInBounds:   acc.totalInBounds   + (m.in_bounds_bounces  ?? 0),
        totalOutBounds:  acc.totalOutBounds  + (m.out_bounds_bounces ?? 0),
      }),
      {
        totalBounces: 0, totalShots: 0, totalRallies: 0,
        totalForehands: 0, totalBackhands: 0, totalServes: 0,
        totalInBounds: 0, totalOutBounds: 0,
      }
    );

    // Whether any match has tennis stats (for empty-state detection)
    const hasTennisStats = doneMatches.some(
      (m) => m.bounce_count !== null || m.shot_count !== null
    );

    const doneDurations = doneMatches
      .map((m) => safeDurationSeconds(m.fps, m.num_frames))
      .filter((d): d is number => d !== null);

    const totalGameplaySeconds = doneDurations.reduce((a, b) => a + b, 0);
    const avgDurationSeconds =
      doneDurations.length > 0 ? totalGameplaySeconds / doneDurations.length : 0;

    const withHeatmapsCount = doneMatches.filter(
      (m) => !!m.bounce_heatmap_path || !!m.player_heatmap_path
    ).length;

    // Per-session data for charts (last 8 done matches)
    const games = doneMatches.slice(0, 8).map((m) => ({
      id: m.id,
      createdAt: m.created_at,
      fps: m.fps,
      numFrames: m.num_frames,
      durationSeconds: safeDurationSeconds(m.fps, m.num_frames),
      hasBallHeatmap: !!m.bounce_heatmap_path,
      hasPlayerHeatmap: !!m.player_heatmap_path,
      bounceCount:    m.bounce_count    ?? null,
      shotCount:      m.shot_count      ?? null,
      rallyCount:     m.rally_count     ?? null,
      forehandCount:  m.forehand_count  ?? null,
      backhandCount:  m.backhand_count  ?? null,
      serveCount:     m.serve_count     ?? null,
      inBounces:      m.in_bounds_bounces  ?? null,
      outBounces:     m.out_bounds_bounces ?? null,
    }));

    return NextResponse.json({
      totals,
      tennisStats,
      hasTennisStats,
      totalGameplaySeconds,
      withHeatmapsCount,
      avgDurationSeconds,
      games,
    });
  } catch (e) {
    console.error(e);
    return NextResponse.json({ error: (e as Error).message }, { status: 500 });
  }
}
