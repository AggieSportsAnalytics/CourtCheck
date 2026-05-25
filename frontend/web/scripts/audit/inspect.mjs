import { chromium } from 'playwright';

const BASE = process.env.CC_AUDIT_BASE ?? 'http://localhost:3000';
const STORAGE = 'scripts/audit/storageState.json';

const browser = await chromium.launch();
const ctx = await browser.newContext({ storageState: STORAGE });
await ctx.addInitScript(() => {
  try { localStorage.setItem('cc-demo', '1'); } catch {}
});
const page = await ctx.newPage();

for (const w of [1440, 1280, 1024, 900, 768, 600]) {
  await page.setViewportSize({ width: w, height: 900 });
  await page.goto(`${BASE}/`, { waitUntil: 'networkidle' });
  await page.waitForTimeout(800);
  const debug = await page.evaluate(() => {
    const body = document.body;
    const aside = document.querySelector('.app-sidebar');
    const h1 = document.querySelector('h1');
    const cs = (el) => el ? getComputedStyle(el) : null;
    return {
      bodyClasses: body.className,
      bodyPaddingLeft: cs(body).paddingLeft,
      bodyWidth: body.clientWidth,
      sidebarRect: aside?.getBoundingClientRect(),
      sidebarComputedWidth: cs(aside)?.width,
      h1Rect: h1?.getBoundingClientRect(),
      h1Text: h1?.textContent,
      htmlScrollWidth: document.documentElement.scrollWidth,
      viewportWidth: window.innerWidth,
    };
  });
  console.log(`--- ${w}px ---`);
  console.log(JSON.stringify(debug, null, 2));
}

await browser.close();
