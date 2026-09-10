import type { Metadata } from 'next';
import { cookies } from 'next/headers';
import { createServerClient } from '@supabase/ssr';
import { supabaseAdmin } from '@/lib/supabase/server';

/** Detail titles follow the same ownership rules as their API routes. */
export async function detailMetadata(table: 'matches' | 'players', id: string, fallback: string): Promise<Metadata> {
  let name = fallback;
  try {
    const cookieStore = await cookies();
    const supabase = createServerClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
      { cookies: {
        getAll: () => cookieStore.getAll(),
        // Server layouts cannot write cookies; the auth proxy refreshes them.
        setAll: () => {},
      } },
    );
    const { data: { user }, error: authError } = await supabase.auth.getUser();
    if (authError) console.error('Title authentication failed', authError);
    if (user) {
      const columns = table === 'matches' ? 'name, input_path' : 'name';
      let query = supabaseAdmin.from(table).select(columns).eq('id', id);
      query = table === 'players' && !user.user_metadata?.onboarding_template
        ? query.or(`user_id.is.null,user_id.eq.${user.id}`)
        : query.eq('user_id', user.id);
      const { data, error } = await query.maybeSingle();
      if (error) console.error('Detail title lookup failed', error);
      const row = data as { name?: string | null; input_path?: string | null } | null;
      if (!error && row) name = row.name || row.input_path?.split('/').pop() || fallback;
    }
  } catch (error) {
    console.error('Detail title failed', error);
  }
  return { title: { absolute: `${name} · CourtCheck` } };
}
