import { chromium } from 'playwright';

const BASE = process.env.CC_AUDIT_BASE ?? 'http://localhost:3000';
const STORAGE = 'scripts/audit/storageState.json';

const TARGETS = [
  { slug: 'landing', path: '/landing', auth: false },
  { slug: 'recording-detail', path: '/recordings/84624cd3-8a35-43ef-bc17-6f87604c044b', auth: true },
  { slug: 'dashboard', path: '/', auth: true },
];

const browser = await chromium.launch();
for (const t of TARGETS) {
  const ctx = await browser.newContext(t.auth ? { storageState: STORAGE } : {});
  await ctx.addInitScript(() => {
    try {
      sessionStorage.setItem('ccDashSplashSeen', '1');
      sessionStorage.setItem('ccLandingSplashSeen', '1');
      localStorage.setItem('cc-demo', '1');
    } catch {}
  });
  const page = await ctx.newPage();
  for (const w of [1440, 900, 600]) {
    await page.setViewportSize({ width: w, height: 900 });
    await page.goto(`${BASE}${t.path}`, { waitUntil: 'networkidle' });
    await page.waitForTimeout(700);
    const info = await page.evaluate(() => {
      const main = document.querySelector('main') ?? document.body;
      const mainBox = main.getBoundingClientRect();
      // Find fixed/sticky elements that might clip at right
      const fixedEls = Array.from(document.querySelectorAll('*')).filter((el) => {
        const cs = getComputedStyle(el);
        return (cs.position === 'fixed' || cs.position === 'sticky') && el.offsetParent !== null;
      }).slice(0, 10).map((el) => ({
        tag: el.tagName.toLowerCase(),
        klass: el.className?.toString()?.slice(0, 60),
        rect: el.getBoundingClientRect(),
        position: getComputedStyle(el).position,
      }));
      return {
        viewport: window.innerWidth,
        mainWidth: mainBox.width,
        mainLeft: mainBox.left,
        mainRight: mainBox.right,
        scrollWidth: document.documentElement.scrollWidth,
        fixedEls,
      };
    });
    console.log(`--- ${t.slug} @ ${w}px ---`);
    console.log(JSON.stringify(info, null, 2));
  }
  await ctx.close();
}
await browser.close();
