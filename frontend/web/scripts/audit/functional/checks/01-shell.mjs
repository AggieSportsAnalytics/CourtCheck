import { APP_PAGES, BASE, FIXTURES, check, assert, expect, note, probe, visit, navigateClick, getData, runModule, ROWS } from '../lib.mjs';

export const area = 'nav';
// Drift: the desktop label is on <aside>, not its nav; mobile is the only labelled nav.
const desktop = 'aside[aria-label="Primary navigation"]';
const mobile = 'nav.app-bottom-nav[aria-label="Primary navigation"]';
const drifts = [{ id: 'DRIFT-NAV', evidence: 'components/layout/Sidebar.tsx: desktop aside[aria-label="Primary navigation"] wraps nav.sidebar-nav; mobile nav.app-bottom-nav is shown below 768px (app/globals.css). DemoToggle.tsx uses right:14, aria-label="Toggle demo mode"; demoData.ts stores explicit off as cc-demo="0".', expected: 'Plan describes two labelled nav elements, mobile at 820px, bottom-left demo switch and a removed localStorage flag.' }];

export const checks = [
  check('NAV-01', 'Primary navigation fails to navigate or mark the current page.', 'All four primary links navigate and carry a visible active state.', async (page, r) => {
    await visit(page, '/', r);
    for (const path of ['/', '/players', '/upload', '/recordings']) {
      await probe(page, r, path, async () => {
        const link = page.locator(`${desktop} nav a[href="${path}"]`);
        await navigateClick(page, link, path);
        const current = await link.getAttribute('aria-current');
        const classes = await link.getAttribute('class');
        note(r, { path, current, classes });
        assert(current === 'page' || /\bactive\b|bg-shade/.test(classes), 'No active marker');
      });
    }
  }),
  check('NAV-02', 'The account menu cannot close or navigate to account pages.', 'Profile/Settings are visible in the menu; Escape and outside click close it; both links work.', async (page, r) => {
    await visit(page, '/', r);
    const account = page.getByLabel('Account menu', { exact: true });
    await account.click();
    for (const path of ['/profile', '/settings']) await expect(page.getByRole('menu').locator(`a[href="${path}"]`)).toBeVisible();
    await page.keyboard.press('Escape'); await expect(account).toHaveAttribute('aria-expanded', 'false');
    await account.click(); await page.locator('main h1').click();
    await expect(account).toHaveAttribute('aria-expanded', 'false');
    for (const path of ['/profile', '/settings']) {
      await account.click();
      await navigateClick(page, page.getByRole('menu').locator(`a[href="${path}"]`), path);
    }
  }),
  check('NAV-03', 'Navigation layout or tap targets fail at a coach viewport.', 'Mobile tabs at 390/820, desktop rail at 1024/1440; all navigation targets at least 44px high.', async (page, r) => {
    for (const width of [390, 820, 1024, 1440]) {
      await probe(page, r, String(width), async () => {
        await page.setViewportSize({ width, height: 900 }); await visit(page, '/', r);
        const mobileVisible = await page.locator(mobile).isVisible();
        const desktopVisible = await page.locator(desktop).isVisible();
        const active = mobileVisible ? mobile : `${desktop} nav`;
        const heights = await page.locator(`${active} a`).evaluateAll((els) => els.map((el) => el.getBoundingClientRect().height));
        note(r, { width, mobileVisible, desktopVisible, heights });
        assert(heights.length >= 4 && heights.every((height) => height >= 44), 'Navigation tap target smaller than 44px');
        // Product breakpoint is 768px (md): iPad portrait gets the desktop rail with 44px targets, which is acceptable.
        assert(mobileVisible === (width < 768) && desktopVisible === (width >= 768), 'Navigation differs from the 768px breakpoint');
      });
    }
    await page.setViewportSize({ width: 1440, height: 900 });
  }),
  check('NAV-04', 'An internal application link is broken or redirects away from its destination.', 'Each discovered internal href returns 200 in the authenticated context.', async (page, r) => {
    const seen = new Set();
    for (const path of APP_PAGES) {
      await visit(page, path, r);
      const links = await page.locator('a[href^="/"]').evaluateAll((els) => els.map((el) => el.getAttribute('href')).filter((href) => !href.startsWith('//')));
      assert(links.length > 0, `${path}: no internal links to crawl`);
      for (const href of links) {
        if (seen.has(href)) continue;
        seen.add(href);
        await probe(page, r, href, async () => {
          const res = await page.request.get(new URL(href, BASE).href);
          const landed = new URL(res.url()).pathname;
          if (/^\/(landing|auth)/.test(landed)) {
            const { SessionDied } = await import('../lib.mjs');
            throw new SessionDied(`${href} landed on ${landed}`);
          }
          note(r, { href, status: res.status(), landed });
          assert(res.status() === 200, `${href}: ${res.status()}`);
        });
      }
    }
  }),
  check('NAV-05', 'Browser history restores a blank or incorrect recording page.', 'Back restores the populated list and forward returns to the same recording.', async (page, r) => {
    await visit(page, '/', r);
    await navigateClick(page, page.locator(`${desktop} a[href="/recordings"]`), '/recordings');
    await expect(page.locator(ROWS).first()).toBeVisible();
    const count = await page.locator(ROWS).count();
    const fixture = await getData(page, `/api/recordings/${FIXTURES.recordingAssigned}`, 'recording');
    const { recordingRow } = await import('../lib.mjs');
    const row = recordingRow(page, fixture);
    await row.scrollIntoViewIfNeeded();
    const scroll = await page.evaluate(() => window.scrollY);
    await navigateClick(page, row, `/recordings/${fixture.id}`);
    await page.goBack(); await expect(page.locator(ROWS)).toHaveCount(count);
    note(r, { previousScroll: scroll, restoredScroll: await page.evaluate(() => window.scrollY) });
    await page.goForward(); await expect(page).toHaveURL((url) => url.pathname === `/recordings/${fixture.id}`);
    await expect(page.locator('main h1')).toBeVisible();
  }),
  check('NAV-06', 'A coach can enable fabricated demo data through a URL parameter.', 'Pilot decision: demo access is intentional and turning it off restores live data.', async (page, r) => {
    await visit(page, '/', r);
    const live = await getData(page, '/api/recordings', 'recordings');
    const originalFlag = await page.evaluate(() => localStorage.getItem('cc-demo'));
    const metrics = page.locator('[aria-label="Team metrics"]');
    const before = await metrics.innerText();
    try {
      await probe(page, r, 'demo removed', async () => {
        // Demo mode was removed on the fix branch: ?demo=1 must change nothing and no switch may exist.
        await visit(page, '/?demo=1', r);
        await expect(page.getByRole('switch', { name: 'Toggle demo mode' })).toHaveCount(0);
        const withDemoParam = await metrics.innerText();
        assert(withDemoParam === before, `Team metrics changed with ?demo=1: ${withDemoParam}`);
        note(r, { liveRecordings: live.length, liveMetrics: before, withDemoParam });
        const flag = await page.evaluate(() => localStorage.getItem('cc-demo'));
        assert(flag === null, `Demo flag written: ${flag}`);
      });
    } finally {
      await page.evaluate((flag) => { if (flag === null) localStorage.removeItem('cc-demo'); else localStorage.setItem('cc-demo', flag); }, originalFlag);
      await visit(page, '/', r);
    }
    assert(false, 'P1 pilot decision required: ?demo=1 exposes fabricated recording statistics to authenticated coaches.');
  }),
  check('NAV-07', 'An application page takes more than three seconds to load.', 'All six desktop pages settle within 3000ms.', async (page, r) => {
    for (const path of APP_PAGES) await probe(page, r, path, async () => {
      const data = await visit(page, path, r);
      assert(data.loadMs <= 3000, `${path}: ${data.loadMs}ms (${data.settle})`);
    });
  }, 'P2'),
];
export default async function (args) { await runModule(args, checks, area, { drifts }); }
