// Admin gate for write operations on shared records (e.g. players, which are
// the team roster — any authenticated user can read, but only listed admins
// can mutate).
//
// Source: ADMIN_EMAILS env var, comma-separated. Default empty (deny-all),
// so a misconfigured deploy fails closed.
//
// Set in Vercel: ADMIN_EMAILS=brile761@gmail.com,coach@school.edu

const ADMIN_EMAILS = new Set(
  (process.env.ADMIN_EMAILS ?? '')
    .split(',')
    .map((s) => s.trim().toLowerCase())
    .filter(Boolean)
);

export function isAdmin(email: string | null | undefined): boolean {
  if (!email) return false;
  return ADMIN_EMAILS.has(email.toLowerCase());
}
