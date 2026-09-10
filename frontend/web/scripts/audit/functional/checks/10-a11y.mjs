import { REVIEW_PAGES, check, assert, expect, note, probe, visit, runModule } from '../lib.mjs';

export const area = 'a11y';
async function routes(page, r, fn) {
  for (const path of REVIEW_PAGES) await probe(page, r, path, async () => { await visit(page, path, r); await fn(path); });
}
async function focusAppearance(page) {
  return page.evaluate(() => {
    const el = document.activeElement;
    if (!el || el === document.body || el === document.documentElement) return { valid: false, reason: 'No document control focused' };
    const sample = (node) => {
      const css = getComputedStyle(node);
      return { outlineStyle: css.outlineStyle, outlineWidth: parseFloat(css.outlineWidth), outlineColor: css.outlineColor, shadow: css.boxShadow };
    };
    const nodes = [el, el.parentElement, el.parentElement?.parentElement].filter(Boolean);
    const focused = nodes.map(sample);
    el.blur();
    const blurred = nodes.map(sample);
    el.focus();
    const ring = focused.some((css, i) => (css.outlineStyle !== 'none' && css.outlineWidth > 0 && css.outlineColor !== 'rgba(0, 0, 0, 0)' && css.outlineColor !== 'transparent') || (css.shadow !== 'none' && css.shadow !== blurred[i].shadow));
    const rect = el.getBoundingClientRect();
    return { valid: ring && rect.width > 0 && rect.height > 0 && rect.bottom > 0 && rect.top < innerHeight, element: el.outerHTML.slice(0, 350), focused, blurred };
  });
}

export const checks = [
  check('A11Y-01', 'Pages have missing/duplicate main headings or skipped heading levels.', 'Exactly one h1 and no downward jump greater than one level on all six pages.', async (page, r) => {
    await routes(page, r, async (path) => {
      await expect(page.locator('h1')).toHaveCount(1);
      const headings = await page.locator('h1,h2,h3,h4,h5,h6').evaluateAll((els) => els.filter((el) => el.checkVisibility()).map((el) => ({ level: Number(el.tagName[1]), text: el.textContent.trim() })));
      const skips = headings.filter((heading, i) => i > 0 && heading.level > headings[i - 1].level + 1);
      note(r, { path, headings, skips });
      assert(headings[0]?.level === 1 && skips.length === 0, `Heading sequence skips levels: ${JSON.stringify(headings)}`);
    });
  }, 'P2'),
  check('A11Y-02', 'Images or interactive controls lack accessible names.', 'Every image has alt; visible buttons/links have a nonempty computed accessible name.', async (page, r) => {
    await routes(page, r, async (path) => {
      const missingAlt = await page.locator('img:not([alt])').evaluateAll((els) => els.map((el) => el.outerHTML.slice(0, 250)));
      const controls = page.locator('button:visible,a:visible');
      const unnamed = [];
      for (let i = 0; i < await controls.count(); i++) {
        const control = controls.nth(i);
        try { await expect(control).toHaveAccessibleName(/\S/, { timeout: 100 }); }
        catch (e) { unnamed.push({ element: await control.evaluate((el) => el.outerHTML.slice(0, 250)), error: e.message }); }
      }
      note(r, { path, checkedControls: await controls.count(), missingAlt, unnamed });
      assert(!missingAlt.length && !unnamed.length, `${missingAlt.length} images missing alt; ${unnamed.length} unnamed controls`);
    });
  }, 'P2'),
  check('A11Y-03', 'Keyboard focus is not visibly indicated.', 'First 25 tab stops on each page have a visible outline or changed box shadow.', async (page, r) => {
    await routes(page, r, async (path) => {
      const stops = [];
      for (let i = 0; i < 25; i++) {
        await page.keyboard.press('Tab');
        const appearance = await focusAppearance(page);
        stops.push({ tab: i + 1, ...appearance });
      }
      const bad = stops.filter((stop) => !stop.valid);
      note(r, { path, checkedStops: stops.length, bad });
      assert(stops.length === 25 && bad.length === 0, `Invisible focus on ${bad.length} of 25 stops`);
    });
  }, 'P2'),
  check('A11Y-04', 'Pages lack document language or unique titles.', 'Nonempty html lang and a distinct nonempty title for every reviewed route.', async (page, r) => {
    const seen = new Map();
    await routes(page, r, async (path) => {
      const lang = await page.locator('html').getAttribute('lang');
      const title = (await page.title()).trim();
      note(r, { path, lang, title });
      assert(lang?.trim() && title, 'Missing lang/title');
      const previous = seen.get(title);
      seen.set(title, path);
      assert(!previous, `Title "${title}" reused on ${previous} and ${path}`);
    });
  }, 'P2'),
];
export default async function (args) { await runModule(args, checks, area); }
