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

## Automated Scanning

CI runs on every PR and push to `main`:
- **CodeQL** — security-and-quality queries for JS/TS + Python
- **npm audit** — fails on `high`+ severity
- **pip-audit** — fails on any known CVE
- **gitleaks** — secret scan on full history

Dependabot runs weekly, groups minor + patch into single PRs per ecosystem.

## Supabase Dashboard Toggles

Some hardening can't live in code. Apply these in the Supabase dashboard for the CourtCheck project:

1. **Enable leaked-password protection** — Auth → Policies → toggle on "Check passwords against HaveIBeenPwned". Blocks signups using known-breached passwords. ([docs](https://supabase.com/docs/guides/auth/password-security#password-strength-and-leaked-password-protection))
2. **Enable Google OAuth** — Auth → Providers → Google → enable. Paste the OAuth Client ID + Secret from Google Cloud Console (Credentials → OAuth 2.0 Client IDs → CourtCheck). Authorized redirect URI in Google Cloud must match: `https://qfqcadgzvflsowzmmfmx.supabase.co/auth/v1/callback`.
3. **Enforce MFA on the org owner account** — Settings → Account → enable TOTP. One-time; protects the project from a phished maintainer password.

## Storage Bucket Visibility

| Bucket | Public | Why |
|---|---|---|
| `raw-videos` | **private** | Coach-uploaded match footage. Access via signed URLs only. |
| `results` | **private** | Processed match outputs + heatmaps. Signed URLs only. |
| `swing-clips` | private | Per-annotator training clips. |
| `assets` | public | Brand assets (logos, marketing) served via plain `<img src>`. |

Any new bucket holding user-uploaded content defaults to private. Reach for `getPublicUrl` only for static brand assets.
