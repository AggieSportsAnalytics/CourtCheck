import { readFile } from 'node:fs/promises';
import { check, assert, expect, note, visit, getData, runModule } from '../lib.mjs';

export const area = 'onboarding';
export const checks = [
  check('ONB-01', 'Onboarding templates are unavailable or missing UC Davis.', 'Authenticated GET 200 returns templates including uc-davis; no POST.', async (page, r) => {
    await visit(page, '/', r);
    const templates = await getData(page, '/api/onboarding', 'templates');
    note(r, { keys: templates.map((template) => template.key) });
    assert(Array.isArray(templates) && templates.some((template) => template.key === 'uc-davis'), 'uc-davis template missing');
  }),
  check('ONB-02', 'An onboarded coach can duplicate a roster without a warning.', 'Already-onboarded users redirect or receive an explicit duplication warning before template selection.', async (page, r) => {
    const data = await visit(page, '/onboarding', r);
    note(r, { landed: data.landed, text: await page.locator('main').innerText() });
    if (data.landed !== '/onboarding') {
      assert(data.landed === '/', `Unexpected onboarded redirect ${data.landed}`);
      await expect(page.getByLabel('Team metrics')).toBeVisible();
      return;
    }
    const picker = page.getByRole('button', { name: 'Use UC Davis', exact: true });
    await expect(picker).toBeVisible();
    const source = await readFile('app/onboarding/page.tsx', 'utf8');
    const directClone = /onClick=\{\(\) => pick\('uc-davis'\)\}/.test(source) && /method: 'POST'/.test(source);
    const text = await page.locator('main').innerText();
    const warns = /duplicat|already.*roster|replace.*roster|existing players|add.*another.*roster/i.test(text);
    note(r, { directClone, warns, pickerEnabled: await picker.isEnabled(), source: 'app/onboarding/page.tsx pick() POST /api/onboarding' });
    assert(!directClone || warns || !await picker.isEnabled(), 'Onboarded user can click Use UC Davis to POST/cloned roster immediately, with no duplication warning. No template clicked.');
  }),
];
export default async function (args) { await runModule(args, checks, area); }
