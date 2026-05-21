// app/api/create-upload/route.ts
import { NextResponse } from "next/server";
import { supabaseAdmin } from "@/lib/supabase/server";
import { createServerClient } from '@supabase/ssr';
import { cookies } from 'next/headers';
import { v4 as uuidv4 } from "uuid";
import { checkRateLimit, rateLimitResponse, clientIp } from '@/lib/ratelimit';

const ALLOWED_EXTS = ['mp4', 'mov', 'avi'];

function sanitizeFilename(raw: string): string | null {
  const ext = raw.split('.').pop()?.toLowerCase() || '';
  if (!ALLOWED_EXTS.includes(ext)) return null;
  const base = raw.slice(0, raw.lastIndexOf('.'));
  const safeBase = base.replace(/[^a-zA-Z0-9\s\-_]/g, '').trim().slice(0, 100) || 'upload';
  return `${safeBase}.${ext}`;
}

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
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    // 10 upload-URL creations / hour. Real users upload 1-3/day; this leaves
    // room for retries while blocking automated abuse.
    const rl = await checkRateLimit({
      userId: user.id,
      ip: clientIp(req),
      bucket: 'create-upload',
      limit: 10,
      windowSec: 3600,
    });
    if (!rl.ok) return rateLimitResponse(rl.retryAfterSec);

    const body = await req.json();
    const playerId: string | null = typeof body.player_id === 'string' ? body.player_id : null;
    const safeFilename = sanitizeFilename((body.filename as string) || '');
    if (!safeFilename) {
      return NextResponse.json({ error: 'Invalid file type. Allowed: mp4, mov, avi' }, { status: 400 });
    }

    // Optional user-supplied metadata. Trim + bound; ignore empties.
    const rawName = typeof body.name === 'string' ? body.name.trim().replace(/[\x00-\x1F\x7F]/g, '') : '';
    const customName = rawName ? rawName.slice(0, 100) : null;
    const rawMatchDate = typeof body.matchDate === 'string' ? body.matchDate.trim() : '';
    // HTML5 date input emits YYYY-MM-DD. Pass through only if it matches.
    const matchDate = /^\d{4}-\d{2}-\d{2}$/.test(rawMatchDate) ? rawMatchDate : null;

    const match_id = uuidv4();
    const file_key = `${match_id}/${safeFilename}`;

    // create row in matches table (status: pending) with user_id
    const insertRow: Record<string, unknown> = {
      id: match_id,
      status: "pending",
      input_path: file_key,
      user_id: user.id,
      created_at: new Date().toISOString(),
      progress: 0,
      error: null,
      results_path: null,
      fps: null,
      num_frames: null,
      player_id: playerId,
    };
    if (customName) insertRow.name = customName;
    await supabaseAdmin.from("matches").insert([insertRow]);

    // match_date column may not exist on the matches table. Attempt a follow-up
    // update so a missing column doesn't break the insert.
    if (matchDate) {
      const { error: matchDateError } = await supabaseAdmin
        .from("matches")
        .update({ match_date: matchDate })
        .eq("id", match_id);
      if (matchDateError) {
        console.warn("match_date update skipped:", matchDateError.message);
      }
    }

    // generate signed PUT URL for direct upload (1 hour)
    const { data, error } = await supabaseAdmin.storage
      .from("raw-videos")
      .createSignedUploadUrl(file_key, { upsert: false });

    if (error) {
      console.error("Supabase signed url error", error);
      return NextResponse.json({ error: "Upload failed" }, { status: 500 });
    }

    return NextResponse.json({
      upload_url: data.signedUrl,
      token: data.token,
      file_key,
      match_id
    });
  } catch (e) {
    console.error(e);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
