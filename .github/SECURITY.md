# Security Policy

## Reporting a Vulnerability

Email **brile761@gmail.com** with the subject `[CourtCheck Security]`. Do not open a public GitHub issue for security reports.

Include:
- What you found
- Steps to reproduce
- Impact assessment
- Suggested fix (optional)

You'll get a reply within 72 hours.

## Supported Versions

Only `main` is supported. Pin a tag at your own risk.

## Secret Rotation Policy

Rotate the following on any of these triggers: maintainer leaves the project, secret appears in logs or a public commit, or every 6 months on a calendar.

| Secret | Where stored | Rotation steps |
|---|---|---|
| Supabase `SERVICE_ROLE_KEY` | Vercel env, Modal env | Supabase dashboard → Settings → API → reset service role key → update Vercel + Modal |
| Supabase `ANON_KEY` | Public (client) + Vercel env | Settings → API → rotate anon key → redeploy Vercel |
| `MODAL_WEBHOOK_SECRET` | Vercel env, Modal env | Generate new value (`openssl rand -hex 32`) → set on Vercel + Modal → redeploy both |
| `OPENAI_API_KEY` | Modal env | OpenAI dashboard → revoke + create → update Modal |
| Google OAuth Client Secret | Supabase Auth Providers | Google Cloud Console → Credentials → reset → paste into Supabase |

## Player Ownership Model

Per-user dashboards. The model:

| `players.user_id` | Visibility | Mutability |
|---|---|---|
| `NULL` | Read-only for every authed user (UC Davis demo data) | Service-role only (DB dashboard) |
| `<uuid>` | Only the owning user (caller = user_id) | Only the owner |

Enforced two ways:
- **RLS policies** on `public.players` — `players_read_demo_or_own`, `players_insert_own`, `players_update_own`, `players_delete_own`. Service-role bypasses (admin API routes still work) but the policies cover any direct anon/authenticated REST call.
- **API ownership checks** in `app/api/players/[id]/route.ts` — fetch + match `user_id` before PATCH/DELETE; the UPDATE itself also `.eq('user_id', user.id)` for race-safety.

POST always pins `user_id = caller`, so a client can't claim a row as someone else's. GET filters to `(user_id IS NULL OR user_id = caller)`.

## Automated Scanning

CI runs on every PR and push to `main`:
- **CodeQL** — security-and-quality queries for JS/TS + Python
- **npm audit** — fails on `high`+ severity
- **pip-audit** — fails on any known CVE
- **gitleaks** — secret scan on full history

Dependabot runs weekly, groups minor + patch into single PRs per ecosystem.

## Supabase Dashboard Toggles

Some hardening can't live in code. Apply these in the Supabase dashboard for the CourtCheck project:

1. **Leaked-password protection (HIBP)** — *requires Supabase Pro plan.* Auth → Providers → Email → "Prevent use of leaked passwords". On Free tier, our signup form enforces minimum 10-char passwords as the next-best floor; revisit when we upgrade. ([docs](https://supabase.com/docs/guides/auth/password-security#password-strength-and-leaked-password-protection))
2. **Google OAuth — enabled.** Auth → Providers → Google. Authorized redirect URI in Google Cloud: `https://qfqcadgzvflsowzmmfmx.supabase.co/auth/v1/callback`.
3. **Enforce MFA on the AggieSportsAnalytics GitHub org owner account** — Settings → Password and authentication → 2FA → authenticator app. One-time; protects the project from a phished maintainer password.

## Storage Bucket Visibility + Caps

| Bucket | Public | Size cap | Mime types | Why |
|---|---|---|---|---|
| `raw-videos` | **private** | 2 GiB | video/mp4, mov, avi, webm | Coach-uploaded match footage. Signed URLs only. |
| `results` | **private** | 500 MiB | any | Processed outputs + heatmaps. Signed URLs only. |
| `swing-clips` | private | 100 MiB | any | Per-annotator training clips. |
| `assets` | public | unlimited | any | Brand assets served via plain `<img src>`. |

Any new bucket holding user-uploaded content defaults to private with a size cap. Reach for `getPublicUrl` only for static brand assets.

## Rate Limits (per authenticated user, rolling 1-hour window)

Backed by `public.rate_limit_events` table + `lib/ratelimit.ts`. Fails open on infra error to avoid taking down legit traffic. 429 with `Retry-After` header on breach.

| Endpoint | Limit | Reason |
|---|---|---|
| `POST /api/create-upload` | 10 | One upload per ~6 min; covers retries |
| `GET /api/proxy-image` | 120 | A page renders ~20 player cards; leaves room for re-renders |
| `PATCH /api/recordings/:id` | 30 | Renames are uncommon |
| `DELETE /api/recordings/:id` | 10 | Destructive — capped tight |
| `POST /api/players` | 10 | Roster additions are infrequent |
| `PATCH /api/players/:id` | 30 | Profile edits |
| `POST /api/trigger-process` | 20 | Processing jobs are expensive (Modal A10G) |

Prune old rows via `SELECT public.rate_limit_prune();` (or wire pg_cron).

## Audit Log

`public.audit_log` captures all INSERT/UPDATE/DELETE on `matches` and `players` via a `SECURITY DEFINER` trigger. Stores `actor` (auth.uid), full before/after JSONB. Service-role-only (RLS enabled, no policies).

Query examples:
```sql
-- Recent deletes on matches
SELECT at, actor, row_id, old_row->>'name'
FROM public.audit_log
WHERE op = 'DELETE' AND table_name = 'matches'
ORDER BY at DESC LIMIT 50;

-- Activity by user
SELECT op, table_name, COUNT(*)
FROM public.audit_log
WHERE actor = '<uuid>' AND at > now() - interval '7 days'
GROUP BY op, table_name;
```

## Supabase Dashboard Toggles

1. **Leaked-password protection (HIBP)** — *requires Supabase Pro plan.* Auth → Providers → Email → "Prevent use of leaked passwords". On Free tier, our signup form enforces minimum 10-char passwords as the next-best floor; revisit when we upgrade.
2. **Min password length** — Auth → Providers → Email → set min length to 10 to match the client-side rule. ([docs](https://supabase.com/docs/guides/auth/password-security))
3. **Google OAuth — enabled.** Auth → Providers → Google. Authorized redirect URI in Google Cloud: `https://qfqcadgzvflsowzmmfmx.supabase.co/auth/v1/callback`.
4. **Rate limit auth endpoints** — Auth → Rate Limits → set 'Sign-ups per IP' and 'Sign-ins per IP' to reasonable floors (e.g., 30/hour). Catches credential stuffing.
5. **Enforce MFA on the AggieSportsAnalytics GitHub org owner account** — Settings → Password and authentication → 2FA → authenticator app.
