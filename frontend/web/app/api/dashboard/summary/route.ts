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
      .select('id, status, progress, error, created_at, fps, num_frames, bounce_heatmap_path, player_heatmap_path')
      .eq('user_id', authData.claims.sub)
      .order('created_at', { ascending: false })
      .limit(50);

    if (error) {
      console.error('Dashboard summary fetch error', error);
      return NextResponse.json({ error: error.message }, { status: 500 });
    }

    const matches = data ?? [];

    const totals = {
      total: matches.length,
      done: matches.filter((m) => m.status === 'done').length,
      processing: matches.filter((m) => m.status === 'processing').length,
      failed: matches.filter((m) => m.status === 'failed').length,
    };

    // "This month" = last 30 days (for dashboard display)
    const now = Date.now();
    const cutoffMs = now - 30 * 24 * 60 * 60 * 1000;
    const recentDone = matches.filter((m) => {
      const t = new Date(m.created_at).getTime();
      return t >= cutoffMs && m.status === 'done';
    });

    const totalGameplaySeconds = recentDone.reduce((acc, m) => {
      const d = safeDurationSeconds(m.fps, m.num_frames);
      return acc + (d ?? 0);
    }, 0);

    const games = matches
      .filter((m) => m.status === 'done')
      .slice(0, 3)
      .map((m) => ({
        id: m.id,
        createdAt: m.created_at,
        fps: m.fps,
        numFrames: m.num_frames,
        durationSeconds: safeDurationSeconds(m.fps, m.num_frames),
        hasBallHeatmap: !!m.bounce_heatmap_path,
        hasPlayerHeatmap: !!m.player_heatmap_path,
      }));

    return NextResponse.json({
      totals,
      totalGameplaySeconds,
      games,
    });
  } catch (e) {
    console.error(e);
    return NextResponse.json({ error: (e as Error).message }, { status: 500 });
  }
}

