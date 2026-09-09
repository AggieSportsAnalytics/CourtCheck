// Post-deploy check against a live host: every app page at desktop and phone width,
// horizontal-overflow measurement, console errors, and a screenshot per page.
//
//   CC_AUDIT_BASE=https://courtcheck-rho.vercel.app node scripts/audit/verify-prod.mjs
//
// Needs a fresh scripts/audit/storageState.json (node scripts/audit/sign-in.mjs).
// Writes the rotated session back at the end so repeated runs do not log you out.
import { chromium } from 'playwright';
import { mkdir } from 'node:fs/promises';

const BASE = process.env.CC_AUDIT_BASE ?? 'http://localhost:3000';
const STORAGE = 'scripts/audit/storageState.json';
const OUT = 'scripts/audit/screenshots/prod';
const RECORDING_ID = process.env.CC_AUDIT_RECORDING_ID ?? null;
const PAGES = [
  ['dashboard', '/'],
  ['recordings', '/recordings'],
  ['players', '/players'],
  ['upload', '/upload'],
  ['settings', '/settings'],
  ['profile', '/profile'],
];
if (RECORDING_ID) PAGES.push(['recording', `/recordings/${RECORDING_ID}`]);
const WIDTHS = [1440, 390];

await mkdir(OUT, { recursive: true });
const browser = await chromium.launch();
const ctx = await browser.newContext({ storageState: STORAGE });
await ctx.addInitScript(() => {
  try {
    sessionStorage.setItem('ccDashSplashSeen', '1');
  } catch {}
});

let failures = 0;
for (const [slug, path] of PAGES) {
  for (const w of WIDTHS) {
    const page = await ctx.newPage();
    await page.setViewportSize({ width: w, height: w < 500 ? 844 : 900 });
    const errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    try {
      await page.goto(`${BASE}${path}`, { waitUntil: 'domcontentloaded', timeout: 60000 });
    } catch {}
    await page.waitForTimeout(3000);
    const landed = new URL(page.url()).pathname;
    if (/\/(landing|auth)/.test(landed)) {
      console.error(`FAIL ${slug} ${w}: session expired, landed on ${landed}. Re-run scripts/audit/sign-in.mjs.`);
      failures += 1;
      await page.close();
      continue;
    }
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    );
    const file = `${OUT}/${slug}-${w}.png`;
    await page.screenshot({ path: file, fullPage: true });
    const bad = overflow > 0 || errors.length > 0;
    if (bad) failures += 1;
    console.log(`${bad ? 'WARN' : 'ok  '} ${slug} ${w}: overflow=${overflow}px errors=${errors.length} -> ${file}`);
    await page.close();
  }
}

await ctx.storageState({ path: STORAGE });
await browser.close();
console.log(failures ? `${failures} page(s) need attention` : 'all pages clean');
process.exit(failures ? 1 : 0);
