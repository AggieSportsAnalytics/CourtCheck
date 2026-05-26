import { chromium } from 'playwright';
import { mkdir } from 'node:fs/promises';

const BASE = process.env.CC_AUDIT_BASE ?? 'http://localhost:3000';
const STORAGE = 'scripts/audit/storageState.json';
const OUT = 'scripts/audit/screenshots';

const RECORDING_ID = process.env.CC_AUDIT_RECORDING_ID ?? null;
const PLAYER_ID = process.env.CC_AUDIT_PLAYER_ID ?? null;

const VIEWPORTS = [1440, 1280, 1024, 900, 768, 600];

const PUBLIC_PAGES = [
  { slug: 'landing',                 path: '/landing' },
  { slug: 'auth-login',              path: '/auth/login' },
  { slug: 'auth-signup',             path: '/auth/signup' },
  { slug: 'auth-forgot-password',    path: '/auth/forgot-password' },
  { slug: 'auth-reset-password',     path: '/auth/reset-password' },
  { slug: 'auth-update-password',    path: '/auth/update-password' },
  { slug: 'auth-confirm',            path: '/auth/confirm' },
];

const APP_PAGES = [
  { slug: 'dashboard',     path: '/' },
  { slug: 'players-list',  path: '/players' },
  { slug: 'recordings',    path: '/recordings' },
  { slug: 'upload',        path: '/upload' },
  { slug: 'match-stats',   path: '/match-stats' },
  { slug: 'overall-stats', path: '/overall-stats' },
  { slug: 'opponents',     path: '/opponents' },
  { slug: 'profile',       path: '/profile' },
  { slug: 'settings',      path: '/settings' },
  { slug: 'onboarding',    path: '/onboarding' },
];

if (PLAYER_ID) APP_PAGES.push({ slug: 'player-detail', path: `/players/${PLAYER_ID}` });
if (RECORDING_ID) APP_PAGES.push({ slug: 'recording-detail', path: `/recordings/${RECORDING_ID}` });

await mkdir(OUT, { recursive: true });

async function shoot(page, p, w) {
  await page.setViewportSize({ width: w, height: 900 });
  try {
    await page.goto(`${BASE}${p.path}`, { waitUntil: 'networkidle', timeout: 30000 });
  } catch {
    // Continue even if networkidle times out — some pages stream forever.
  }
  await page.waitForTimeout(700);
  const file = `${OUT}/${p.slug}-${w}.png`;
  await page.screenshot({ path: file, fullPage: true });
  console.log(`shot ${file}`);
}

// Public pages — fresh anonymous context per page so cookies don't leak.
{
  const browser = await chromium.launch();
  for (const p of PUBLIC_PAGES) {
    const ctx = await browser.newContext();
    const page = await ctx.newPage();
    for (const w of VIEWPORTS) {
      await shoot(page, p, w);
    }
    await ctx.close();
  }
  await browser.close();
}

// App pages — reuse authenticated storage state. Demo flag seeded via
// localStorage so populated layouts get stressed.
{
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ storageState: STORAGE });
  await ctx.addInitScript(() => {
    try {
      localStorage.setItem('cc-demo', '1');
      sessionStorage.setItem('ccDashSplashSeen', '1');
      sessionStorage.setItem('ccLandingSplashSeen', '1');
    } catch {}
  });
  const page = await ctx.newPage();
  for (const p of APP_PAGES) {
    for (const w of VIEWPORTS) {
      await shoot(page, p, w);
    }
  }
  await ctx.close();
  await browser.close();
}

console.log('audit complete');
