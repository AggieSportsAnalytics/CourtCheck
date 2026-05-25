import { chromium } from 'playwright';

const BASE = process.env.CC_AUDIT_BASE ?? 'http://localhost:3000';
const STORAGE = 'scripts/audit/storageState.json';
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
  { slug: 'overall-stats', path: '/overall-stats' },
  { slug: 'opponents',     path: '/opponents' },
  { slug: 'profile',       path: '/profile' },
  { slug: 'settings',      path: '/settings' },
  { slug: 'onboarding',    path: '/onboarding' },
];

if (PLAYER_ID) APP_PAGES.push({ slug: 'player-detail', path: `/players/${PLAYER_ID}` });
if (RECORDING_ID) APP_PAGES.push({ slug: 'recording-detail', path: `/recordings/${RECORDING_ID}` });

async function scanOverflow(page, slug, w) {
  const offenders = await page.evaluate((vw) => {
    const out = [];
    for (const el of document.querySelectorAll('*')) {
      const rect = el.getBoundingClientRect();
      if (rect.right > vw + 0.5 && rect.width > 0 && rect.width < vw + 200) {
        out.push({
          tag: el.tagName.toLowerCase(),
          klass: (el.className?.toString?.() || '').slice(0, 60),
          right: rect.right,
          overflow: rect.right - vw,
          text: (el.textContent?.trim() || '').slice(0, 50),
        });
      }
    }
    // Outermost offenders only (skip children whose parent already overflows worse).
    return out.filter((cur, idx) => !out.some((o, oi) => oi !== idx && o.right >= cur.right - 1 && o.overflow > cur.overflow))
      .slice(0, 5);
  }, w);
  if (offenders.length === 0) return;
  console.log(`OVERFLOW: ${slug} @ ${w}px`);
  for (const o of offenders) {
    console.log(`  <${o.tag}.${o.klass}> ${o.overflow.toFixed(0)}px past viewport. text="${o.text.replace(/\s+/g, ' ')}"`);
  }
}

const browser = await chromium.launch();

// Public pages
{
  for (const p of PUBLIC_PAGES) {
    const ctx = await browser.newContext();
    const page = await ctx.newPage();
    for (const w of VIEWPORTS) {
      await page.setViewportSize({ width: w, height: 900 });
      try {
        await page.goto(`${BASE}${p.path}`, { waitUntil: 'networkidle', timeout: 30000 });
      } catch {}
      await page.waitForTimeout(400);
      await scanOverflow(page, p.slug, w);
    }
    await ctx.close();
  }
}

// App pages
{
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
      await page.setViewportSize({ width: w, height: 900 });
      try {
        await page.goto(`${BASE}${p.path}`, { waitUntil: 'networkidle', timeout: 30000 });
      } catch {}
      await page.waitForTimeout(400);
      await scanOverflow(page, p.slug, w);
    }
  }
  await ctx.close();
}

console.log('done');
await browser.close();
