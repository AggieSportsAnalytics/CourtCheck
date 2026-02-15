// app/api/trigger-process/route.ts
import { createServerClient } from '@supabase/ssr';
import { cookies } from 'next/headers';
import { supabaseAdmin } from '@/lib/supabase/server';

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

    const { data: authData } = await supabase.auth.getClaims();

    if (!authData?.claims) {
      return Response.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const body = await req.json();
    const { file_key, match_id } = body;
    console.log("Trigger payload:", body);
    if (!file_key || !match_id) {
      return Response.json(
        { error: "Missing file_key or match_id" },
        { status: 400 }
      );
    }

    // Verify the match belongs to the authenticated user
    const { data: match, error: matchError } = await supabaseAdmin
      .from("matches")
      .select("user_id")
      .eq("id", match_id)
      .single();

    if (matchError || !match || match.user_id !== authData.claims.sub) {
      return Response.json({ error: 'Forbidden' }, { status: 403 });
    }

    // Call Modal function to process video
    const res = await fetch(
      process.env.MODAL_FUNCTION_URL!, 
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          file_key,
          match_id
        }),
      }
    );

    if (!res.ok) {
      const errorText = await res.text();
      console.error("Modal function error", errorText);
      return Response.json({ error: "processing failed" }, { status: 500 });
    }
    
    return Response.json({ status: "ok" });
  } catch (e) {
    console.error(e);
    return Response.json({ error: (e as Error).message }, { status: 500 });
  }
}