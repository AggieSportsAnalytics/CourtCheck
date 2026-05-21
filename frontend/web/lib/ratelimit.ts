// Per-user (or per-IP) sliding-window rate limiter backed by Supabase.
// Pattern: count events in the window, allow if under limit, then record this event.
// Fails open on infra error so a degraded DB doesn't take down legitimate traffic.

import { supabaseAdmin } from './supabase/server';

export type RateLimitResult = { ok: true } | { ok: false; retryAfterSec: number };

export interface RateLimitOpts {
  bucket: string;        // logical identifier, e.g. 'create-upload', 'proxy-image'
  limit: number;         // max events allowed in the window
  windowSec: number;     // window size in seconds
  userId?: string | null;
  ip?: string | null;
}

export async function checkRateLimit(opts: RateLimitOpts): Promise<RateLimitResult> {
  const { bucket, limit, windowSec, userId, ip } = opts;
  if (!userId && !ip) {
    // No identifier means we can't count this caller — refuse rather than
    // let the request through unrate-limited. Misconfigured caller is a bug,
    // not a reason to skip enforcement.
    console.warn('[ratelimit] no userId or ip provided', { bucket });
    return { ok: false, retryAfterSec: windowSec };
  }

  const since = new Date(Date.now() - windowSec * 1000).toISOString();

  let query = supabaseAdmin
    .from('rate_limit_events')
    .select('id', { count: 'exact', head: true })
    .eq('bucket', bucket)
    .gte('at', since);

  if (userId) query = query.eq('user_id', userId);
  else if (ip) query = query.eq('ip', ip);

  const { count, error } = await query;

  if (error) {
    console.error('[ratelimit] count error', { bucket, error: error.message });
    return { ok: true }; // fail open
  }

  if ((count ?? 0) >= limit) {
    return { ok: false, retryAfterSec: windowSec };
  }

  const { error: insertErr } = await supabaseAdmin
    .from('rate_limit_events')
    .insert({ user_id: userId ?? null, ip: ip ?? null, bucket });

  if (insertErr) {
    console.error('[ratelimit] insert error', { bucket, error: insertErr.message });
    // Still allow — we already counted under limit.
  }

  return { ok: true };
}

export function clientIp(req: Request): string | null {
  // Vercel: x-forwarded-for is set by their edge; pick the first hop.
  const xff = req.headers.get('x-forwarded-for');
  if (xff) return xff.split(',')[0].trim();
  const real = req.headers.get('x-real-ip');
  if (real) return real.trim();
  return null;
}

export function rateLimitResponse(retryAfterSec: number) {
  return new Response(
    JSON.stringify({ error: 'Too many requests' }),
    {
      status: 429,
      headers: {
        'Content-Type': 'application/json',
        'Retry-After': String(retryAfterSec),
      },
    }
  );
}
