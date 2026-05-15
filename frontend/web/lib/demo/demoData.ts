/**
 * Demo fixtures — used ONLY when `?demo=1` is in the URL (or
 * NEXT_PUBLIC_DEMO_MODE=1). Makes the dashboard + recordings list look like a
 * coach has been using CourtCheck for a season, for screen-recording the
 * pitch. No Supabase writes, no cloned videos: match-detail is still demoed
 * from a real processed recording. Toggle off by dropping the query param.
 */

export const DEMO_LS_KEY = 'cc-demo';

/** True when demo mode is active: persistent toggle (localStorage), env, or
 *  a `?demo=1` URL param. The toggle is the primary switch; the param still
 *  works and the <DemoToggle> syncs it into localStorage so it persists
 *  across navigation. */
export function isDemoMode(search?: URLSearchParams | string | null): boolean {
  if (process.env.NEXT_PUBLIC_DEMO_MODE === '1') return true;
  if (typeof window !== 'undefined') {
    try {
      if (window.localStorage.getItem(DEMO_LS_KEY) === '1') return true;
    } catch {
      /* localStorage blocked — fall through */
    }
  }
  if (search) {
    const params =
      typeof search === 'string' ? new URLSearchParams(search) : search;
    const v = params.get('demo');
    if (v === '1' || v === 'true') return true;
  }
  return false;
}

export function getDemoFlag(): boolean {
  if (typeof window === 'undefined') return false;
  try {
    return window.localStorage.getItem(DEMO_LS_KEY) === '1';
  } catch {
    return false;
  }
}

export function setDemoFlag(on: boolean): void {
  if (typeof window === 'undefined') return;
  try {
    if (on) window.localStorage.setItem(DEMO_LS_KEY, '1');
    else window.localStorage.removeItem(DEMO_LS_KEY);
  } catch {
    /* localStorage blocked — no-op */
  }
}

function daysAgo(n: number): string {
  const d = new Date();
  d.setDate(d.getDate() - n);
  d.setHours(15, 12, 0, 0);
  return d.toISOString();
}

export type DemoPlayer = {
  id: string;
  name: string;
  position: string | null;
  year: string | null;
  photo_url: string | null;
  created_at: string;
};

export type DemoRecording = {
  id: string;
  status: 'done';
  createdAt: string;
  name: string;
  filename: string;
  forehandCount: number;
  backhandCount: number;
  serveCount: number;
  inBoundsBounces: number;
  outBoundsBounces: number;
  shotCount: number;
  bounceCount: number;
  rallyCount: number;
  fps: number;
  numFrames: number;
  player_id: string | null;
};

// UC Davis women's tennis is the design partner. Plausible roster (not real
// individuals) so the screen recording reads as a real, populated program.
export const DEMO_PLAYERS: DemoPlayer[] = [
  { id: 'demo-p1', name: 'Maya Lin', position: 'Singles 1', year: 'Senior', photo_url: null, created_at: daysAgo(120) },
  { id: 'demo-p2', name: 'Jordan Rivera', position: 'Singles 2', year: 'Junior', photo_url: null, created_at: daysAgo(118) },
  { id: 'demo-p3', name: 'Aspen Park', position: 'Singles 3', year: 'Sophomore', photo_url: null, created_at: daysAgo(110) },
  { id: 'demo-p4', name: 'Daniela Cruz', position: 'Singles 4', year: 'Senior', photo_url: null, created_at: daysAgo(108) },
  { id: 'demo-p5', name: 'Sophie Whitman', position: 'Singles 5', year: 'Freshman', photo_url: null, created_at: daysAgo(96) },
  { id: 'demo-p6', name: 'Renata Oyelaran', position: 'Singles 6', year: 'Sophomore', photo_url: null, created_at: daysAgo(92) },
  { id: 'demo-p7', name: 'Priya Sharma', position: 'Doubles A', year: 'Junior', photo_url: null, created_at: daysAgo(86) },
  { id: 'demo-p8', name: 'Camille Boucher', position: 'Doubles B', year: 'Senior', photo_url: null, created_at: daysAgo(80) },
  { id: 'demo-p9', name: 'Hana Watanabe', position: 'Reserve', year: 'Freshman', photo_url: null, created_at: daysAgo(64) },
];

const OPPONENTS = ['Stanford', 'Cal', 'USC', 'UCLA', 'Pepperdine', 'Saint Mary’s', 'Fresno State', 'Pacific'];

const REC_FPS = 30;
// 52 min per recording. 18 recordings -> 18 * 52 = 936 min = 15.6 hours.
const REC_SECONDS = 52 * 60;
const REC_FRAMES = REC_SECONDS * REC_FPS;

function rec(
  i: number,
  playerIdx: number,
  daysBack: number,
  fh: number,
  bh: number,
  sv: number,
  inB: number,
  outB: number,
): DemoRecording {
  const p = DEMO_PLAYERS[playerIdx];
  const opp = OPPONENTS[i % OPPONENTS.length];
  const last = p.name.split(' ')[1] ?? p.name;
  const shots = fh + bh + sv;
  return {
    id: `demo-rec-${i}`,
    status: 'done',
    createdAt: daysAgo(daysBack),
    name: `${last} vs ${opp} · Ct ${1 + (i % 4)}`,
    filename: `ucd_${last.toLowerCase()}_${opp.toLowerCase().replace(/\W/g, '')}.mp4`,
    forehandCount: fh,
    backhandCount: bh,
    serveCount: sv,
    inBoundsBounces: inB,
    outBoundsBounces: outB,
    shotCount: shots,
    bounceCount: inB + outB,
    rallyCount: Math.round(shots / 4.3),
    fps: REC_FPS,
    numFrames: REC_FRAMES,
    player_id: p.id,
  };
}

// 18 recordings (2 per player across the 9-player roster), spread across the
// last ~7 weeks. 18 * 52 min = 15.6 hours recorded.
export const DEMO_RECORDINGS: DemoRecording[] = [
  rec(1, 0, 2, 38, 21, 9, 54, 14),
  rec(2, 1, 4, 31, 27, 7, 49, 16),
  rec(3, 2, 5, 44, 18, 11, 61, 12),
  rec(4, 3, 7, 35, 24, 8, 52, 15),
  rec(5, 4, 9, 29, 30, 6, 47, 18),
  rec(6, 5, 11, 41, 16, 12, 58, 11),
  rec(7, 6, 13, 33, 25, 9, 50, 17),
  rec(8, 7, 15, 26, 22, 5, 41, 13),
  rec(9, 8, 17, 30, 19, 7, 44, 14),
  rec(10, 0, 20, 42, 19, 11, 60, 11),
  rec(11, 1, 23, 36, 28, 8, 55, 15),
  rec(12, 2, 26, 47, 20, 13, 65, 10),
  rec(13, 3, 29, 31, 26, 7, 48, 17),
  rec(14, 4, 32, 28, 31, 6, 46, 19),
  rec(15, 5, 35, 43, 17, 12, 60, 12),
  rec(16, 6, 38, 34, 24, 9, 53, 15),
  rec(17, 7, 42, 25, 21, 5, 40, 13),
  rec(18, 8, 46, 32, 18, 8, 45, 14),
];

const totalSeconds = DEMO_RECORDINGS.reduce((s, r) => s + r.numFrames / r.fps, 0);

export type DemoSummary = {
  totals: { total: number; done: number; processing: number; failed: number };
  tennisStats: {
    totalBounces: number;
    totalShots: number;
    totalRallies: number;
    totalForehands: number;
    totalBackhands: number;
    totalServes: number;
    totalInBounds: number;
    totalOutBounds: number;
  };
  hasTennisStats: boolean;
  totalGameplaySeconds: number;
  withHeatmapsCount: number;
  avgDurationSeconds: number;
  games: Array<{
    id: string;
    createdAt: string;
    durationSeconds: number | null;
    shotCount: number | null;
    forehandCount: number | null;
    backhandCount: number | null;
    serveCount: number | null;
  }>;
};

const agg = DEMO_RECORDINGS.reduce(
  (a, r) => ({
    fh: a.fh + r.forehandCount,
    bh: a.bh + r.backhandCount,
    sv: a.sv + r.serveCount,
    inB: a.inB + r.inBoundsBounces,
    outB: a.outB + r.outBoundsBounces,
    shots: a.shots + r.shotCount,
    bounces: a.bounces + r.bounceCount,
    rallies: a.rallies + r.rallyCount,
  }),
  { fh: 0, bh: 0, sv: 0, inB: 0, outB: 0, shots: 0, bounces: 0, rallies: 0 },
);

export const DEMO_SUMMARY: DemoSummary = {
  totals: { total: DEMO_RECORDINGS.length, done: DEMO_RECORDINGS.length, processing: 0, failed: 0 },
  tennisStats: {
    totalBounces: agg.bounces,
    totalShots: agg.shots,
    totalRallies: agg.rallies,
    totalForehands: agg.fh,
    totalBackhands: agg.bh,
    totalServes: agg.sv,
    totalInBounds: agg.inB,
    totalOutBounds: agg.outB,
  },
  hasTennisStats: true,
  totalGameplaySeconds: totalSeconds,
  withHeatmapsCount: DEMO_RECORDINGS.length,
  avgDurationSeconds: Math.round(totalSeconds / DEMO_RECORDINGS.length),
  games: DEMO_RECORDINGS.map((r) => ({
    id: r.id,
    createdAt: r.createdAt,
    durationSeconds: REC_SECONDS,
    shotCount: r.shotCount,
    forehandCount: r.forehandCount,
    backhandCount: r.backhandCount,
    serveCount: r.serveCount,
  })),
};
