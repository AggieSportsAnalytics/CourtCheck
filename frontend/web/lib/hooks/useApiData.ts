'use client';

// Shared SWR hooks for the read-heavy authed pages (dashboard, players list,
// player detail, recordings list). These all hit the same handful of GET
// endpoints, so keying them through SWR gives stale-while-revalidate caching:
// a revisit renders cached data instantly and revalidates in the background,
// and concurrent requests for the same key dedup into one.
//
// Demo mode (?demo=1) short-circuits the network: the hook serves the
// fabricated roster/recordings via `fallbackData` and passes a null key so SWR
// never fetches.

import useSWR, { type SWRConfiguration, type SWRResponse } from 'swr';
import {
  isDemoMode,
  DEMO_PLAYERS,
  DEMO_RECORDINGS,
  DEMO_SUMMARY,
} from '@/lib/demo/demoData';

export const fetcher = async (url: string) => {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Request failed (${res.status})`);
  return res.json();
};

function demoOn(): boolean {
  return isDemoMode(typeof window !== 'undefined' ? window.location.search : null);
}

// Defaults: revalidate on focus/reconnect (SWR defaults) so the UI self-heals
// without a manual refresh; keepPreviousData so navigating between pages shows
// the last-known data instead of flashing a skeleton while revalidating.
const BASE_CONFIG: SWRConfiguration = {
  revalidateOnFocus: true,
  revalidateOnReconnect: true,
  keepPreviousData: true,
  dedupingInterval: 2000,
};

// ---- Response row shapes (broad supersets; pages cast to their local types) ----

export interface PlayerRow {
  id: string;
  name: string;
  position: string | null;
  year: string | null;
  photo_url: string | null;
  created_at: string;
  handedness?: string | null;
}
export interface PlayersData {
  players: PlayerRow[];
}

export interface RecordingRow {
  id: string;
  status: string;
  progress?: number;
  error?: string | null;
  createdAt: string;
  name: string;
  filename: string;
  fps?: number | null;
  numFrames?: number | null;
  bounceCount?: number | null;
  shotCount?: number | null;
  rallyCount?: number | null;
  forehandCount?: number | null;
  backhandCount?: number | null;
  serveCount?: number | null;
  inBoundsBounces?: number | null;
  outBoundsBounces?: number | null;
  player_id?: string | null;
  playerName?: string | null;
}
export interface RecordingsData {
  recordings: RecordingRow[];
}

/** Players roster. Cached + deduped across the dashboard, list, and detail pages. */
export function usePlayersData(): SWRResponse<PlayersData> {
  const demo = demoOn();
  return useSWR<PlayersData>(demo ? null : '/api/players', fetcher, {
    ...BASE_CONFIG,
    // Demo roster matches the API shape closely enough for the card views.
    fallbackData: demo
      ? ({ players: DEMO_PLAYERS } as unknown as PlayersData)
      : undefined,
  });
}

/**
 * Recordings list. While any recording is still `processing`, polls every 5s so
 * a `processing → done` transition shows live (mirrors the detail page's
 * /api/status polling); once everything is settled, polling stops so we're not
 * hammering the DB on an idle list.
 */
export function useRecordingsData(): SWRResponse<RecordingsData> {
  const demo = demoOn();
  return useSWR<RecordingsData>(demo ? null : '/api/recordings', fetcher, {
    ...BASE_CONFIG,
    refreshInterval: (latest) =>
      latest?.recordings?.some((r) => r.status === 'processing') ? 5000 : 0,
    fallbackData: demo
      ? ({ recordings: DEMO_RECORDINGS } as unknown as RecordingsData)
      : undefined,
  });
}

// Loosely typed: the dashboard owns the full summary shape and casts on consume.
export function useDashboardSummary(): SWRResponse<unknown> {
  const demo = demoOn();
  return useSWR<unknown>(demo ? null : '/api/dashboard/summary', fetcher, {
    ...BASE_CONFIG,
    fallbackData: demo ? (DEMO_SUMMARY as unknown) : undefined,
  });
}
