// app/api/create-upload/route.ts
import { NextResponse } from "next/server";
import { supabaseAdmin } from "@/lib/supabase/server";
import { createServerClient } from '@supabase/ssr';
import { cookies } from 'next/headers';
import { v4 as uuidv4 } from "uuid";

export async function POST(req: Request) {
  console.log("ENV CHECK", {
    hasUrl: !!process.env.NEXT_PUBLIC_SUPABASE_URL,
    hasServiceKey: !!process.env.SUPABASE_SERVICE_ROLE_KEY,
  });

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

    const body = await req.json();
    const filename = body.filename || `upload-${Date.now()}.mp4`;
    const match_id = uuidv4();
    const file_key = `${match_id}/${filename}`;

    // create row in matches table (status: pending) with user_id
    await supabaseAdmin.from("matches").insert([{
      id: match_id,
      status: "pending",
      input_path: file_key,
      user_id: authData.claims.sub,
      created_at: new Date().toISOString(),
      progress: 0,
      error: null,
      results_path: null,
      fps: null,
      num_frames: null
    }]);

    // generate signed PUT URL for direct upload (1 hour)
    const { data, error } = await supabaseAdmin.storage
      .from("raw-videos")
      .createSignedUploadUrl(file_key, { upsert: false });

    if (error) {
      console.error("Supabase signed url error", error);
      return NextResponse.json({ error: error.message }, { status: 500 });
    }

    return NextResponse.json({
      upload_url: data.signedUrl,
      token: data.token,
      file_key,
      match_id
    });
  } catch (e) {
    console.error(e);
    return NextResponse.json({ error: (e as Error).message }, { status: 500 });
  }
}
