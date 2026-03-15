import { NextResponse } from 'next/server';
import { supabaseAdmin } from '@/lib/supabase/server';
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
      .select('id, name, input_path, created_at, status, bounce_heatmap_path, player_heatmap_path, bounce_count, shot_count, rally_count')
      .eq('user_id', authData.claims.sub)
      .eq('status', 'done')
      .order('created_at', { ascending: false })
      .limit(1);

    if (error) {
      console.error('Latest heatmaps fetch error', error);
      return NextResponse.json({ error: error.message }, { status: 500 });
    }

    const latest = data?.[0];
    if (!latest) {
      return NextResponse.json({ heatmaps: null }, { status: 404 });
    }

    let bounceHeatmapUrl: string | null = null;
    let playerHeatmapUrl: string | null = null;
    let playerShotMapUrl: string | null = null;

    if (latest.bounce_heatmap_path) {
      const { data: signed } = await supabaseAdmin.storage
        .from('results')
        .createSignedUrl(latest.bounce_heatmap_path, 3600);
      bounceHeatmapUrl = signed?.signedUrl ?? null;
    }

    if (latest.player_heatmap_path) {
      const { data: signed } = await supabaseAdmin.storage
        .from('results')
        .createSignedUrl(latest.player_heatmap_path, 3600);
      playerHeatmapUrl = signed?.signedUrl ?? null;
    }

    // player_shot_map_path not yet in schema — skip gracefully

    return NextResponse.json({
      heatmaps: {
        matchId: latest.id,
        matchName: latest.name || latest.input_path?.split('/').pop() || 'Untitled',
        createdAt: latest.created_at,
        bounceHeatmapUrl,
        playerHeatmapUrl,
        playerShotMapUrl,
        bounceCount: latest.bounce_count,
        shotCount: latest.shot_count,
        rallyCount: latest.rally_count,
      },
    });
  } catch (e) {
    console.error(e);
    return NextResponse.json({ error: (e as Error).message }, { status: 500 });
  }
}

