import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/**
 * Build the canonical `/api/proxy-image` URL for a player headshot.
 *
 * Every player card MUST go through this so the dashboard and the players list
 * request the *same* URL for a given player — one browser cache entry, one hit
 * against the proxy's per-user rate limit. Previously the two cards built
 * different URLs (raw vs. `width=300`), doubling proxy traffic and splitting the
 * cache, which made headshots intermittently fall back to initials.
 *
 * Normalizes any `width=NN` query param to 300 (good enough for a 56px avatar
 * and the largest size we display) so the output is stable regardless of the
 * width the caller's source URL happened to carry.
 *
 * @returns the proxy URL, or null when there's no photo to load.
 */
export function playerPhotoProxyUrl(photoUrl: string | null | undefined): string | null {
  if (!photoUrl) return null
  const normalized = photoUrl.replace(/([?&]width=)\d+/, '$1300')
  return `/api/proxy-image?url=${encodeURIComponent(normalized)}`
}
