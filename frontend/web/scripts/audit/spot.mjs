import { chromium } from 'playwright';
import { mkdir } from 'node:fs/promises';

const BASE = process.env.CC_AUDIT_BASE ?? 'http://localhost:3000';
const STORAGE = 'scripts/audit/storageState.json';
const OUT = 'scripts/audit/screenshots';

const RECORDING_ID = process.env.CC_AUDIT_RECORDING_ID ?? null;
const PLAYER_ID = process.env.CC_AUDIT_PLAYER_ID ?? null;

const ROUTES = {
  'landing':                 { path: '/landing',                 auth: false },
  'auth-login':              { path: '/auth/login',              auth: false },
  'auth-signup':             { path: '/auth/signup',             auth: false },
  'auth-forgot-password':    { path: '/auth/forgot-password',    auth: false },
  'auth-reset-password':     { path: '/auth/reset-password',     auth: false },
  'auth-update-password':    { path: '/auth/update-password',    auth: false },
  'auth-confirm':            { path: '/auth/confirm',            auth: false },
  'dashboard':               { path: '/',                        auth: true  },
  'players-list':            { path: '/players',                 auth: true  },
  'recordings':              { path: '/recordings',              auth: true  },
  'upload':                  { path: '/upload',                  auth: true  },
  'match-stats':             { path: '/match-stats',             auth: true  },
  'overall-stats':           { path: '/overall-stats',           auth: true  },
  'opponents':               { path: '/opponents',               auth: true  },
  'profile':                 { path: '/profile',                 auth: true  },
  'settings':                { path: '/settings',                auth: true  },
  'onboarding':              { path: '/onboarding',              auth: true  },
  'player-detail':           { path: PLAYER_ID    ? `/players/${PLAYER_ID}`       : null, auth: true },
  'recording-detail':        { path: RECORDING_ID ? `/recordings/${RECORDING_ID}` : null, auth: true },
};

const [, , slug, widthStr] = process.argv;
const width = widthStr ? Number(widthStr) : 1024;

if (!slug || !ROUTES[slug]) {
  console.error('Usage: node scripts/audit/spot.mjs <page-slug> <width>');
  console.error('Slugs:', Object.keys(ROUTES).join(', '));
  process.exit(1);
}

const route = ROUTES[slug];
if (!route.path) {
  console.error(`Skipped: ${slug} requires PLAYER_ID or RECORDING_ID env var.`);
  process.exit(1);
}

await mkdir(OUT, { recursive: true });

const browser = await chromium.launch();
const ctx = await browser.newContext(
  route.auth ? { storageState: STORAGE } : {}
);
await ctx.addInitScript(() => {
  try {
    sessionStorage.setItem('ccDashSplashSeen', '1');
    sessionStorage.setItem('ccLandingSplashSeen', '1');
    if (location && location.pathname !== '/landing' && location.pathname !== '/auth') {
      localStorage.setItem('cc-demo', '1');
    }
  } catch {}
});
const page = await ctx.newPage();
await page.setViewportSize({ width, height: 900 });
try {
  await page.goto(`${BASE}${route.path}`, { waitUntil: 'networkidle', timeout: 30000 });
} catch {}
await page.waitForTimeout(700);
const file = `${OUT}/${slug}-${width}.png`;
await page.screenshot({ path: file, fullPage: true });
console.log(`shot ${file}`);
await browser.close();
