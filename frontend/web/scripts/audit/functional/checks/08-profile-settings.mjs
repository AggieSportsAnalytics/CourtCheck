import { readFile } from 'node:fs/promises';
import { BASE, check, assert, expect, note, visit, getData, viewportChecks, runModule } from '../lib.mjs';

export const area = 'settings';
export async function saveDisplayName(page, value) {
  await page.getByPlaceholder('Your name', { exact: true }).fill(value);
  await page.getByRole('button', { name: 'Save', exact: true }).click();
  await expect(page.getByText('Display name updated.', { exact: true })).toBeVisible({ timeout: 15000 });
  await page.reload({ waitUntil: 'domcontentloaded' });
  await expect(page.getByPlaceholder('Your name', { exact: true })).toHaveValue(value, { timeout: 15000 });
}

export const checks = [
  { ...check('PRF-01', 'Profile career statistics disagree with the dashboard or quick actions are broken.', 'Four API-backed numeric career stats and 200 upload/recordings/settings links.', async (page, r) => {
    await visit(page, '/profile', r);
    const summary = await getData(page, '/api/dashboard/summary');
    const expected = { 'Recordings analyzed': summary.totals.done, 'Total shots': summary.tennisStats.totalShots, 'Total bounces': summary.tennisStats.totalBounces, 'Total rallies': summary.tennisStats.totalRallies };
    const stats = page.getByLabel('Career stats');
    await expect(stats.locator('.cc-stat-tile')).toHaveCount(4);
    for (const [label, value] of Object.entries(expected)) {
      assert(Number.isFinite(value), `API ${label} is not numeric`);
      const tile = stats.locator('.cc-stat-tile').filter({ hasText: label });
      await expect(tile.locator(':scope > div').last()).toHaveText(value.toLocaleString('en-US'));
    }
    note(r, { stats: expected });
    for (const path of ['/upload', '/recordings', '/settings']) {
      const link = page.getByLabel('Quick actions').locator(`a[href="${path}"]`);
      await expect(link).toBeVisible();
      const res = await page.request.get(`${BASE}${path}`);
      assert(res.status() === 200 && new URL(res.url()).pathname === path, `${path}: ${res.status()} ${res.url()}`);
    }
  }), area: 'profile' },
  check('SET-01', 'Display-name edits fail to persist or restore.', 'Prefilled name; save QA suffix with success, reload persistence, then restore and verify.', async (page, r, shared) => {
    await visit(page, '/settings', r);
    const input = page.getByPlaceholder('Your name', { exact: true });
    await expect(input).not.toHaveValue('');
    const original = await input.inputValue();
    shared.displayNameOriginal ??= original;
    note(r, { originalName: original });
    try { await saveDisplayName(page, `${original} QA`); await saveDisplayName(page, original); }
    finally {
      await visit(page, '/settings', r);
      if (await input.inputValue() !== original) await saveDisplayName(page, original);
    }
  }),
  check('SET-02', 'Sign out is absent, disabled, unfocusable or unwired.', 'Enabled focusable button; source wiring to confirmation and signOut verified without clicking.', async (page, r) => {
    await visit(page, '/settings', r);
    const button = page.getByRole('button', { name: 'Sign out', exact: true });
    await expect(button).toBeVisible(); await expect(button).toBeEnabled();
    assert(await button.evaluate((el) => el.tagName === 'BUTTON' && el.tabIndex >= 0), 'Sign out is not a focusable button');
    await button.focus(); await expect(button).toBeFocused();
    const source = await readFile('app/settings/page.tsx', 'utf8');
    assert(/onClick=\{\(\) => setConfirmSignOut\(true\)\}/.test(source) && /onClick=\{\(\) => signOut\(\)\}/.test(source), 'Source signOut/confirmation wiring missing');
    note(r, 'Enabled BUTTON focused; app/settings/page.tsx wires confirmation then signOut(). Neither button clicked.');
  }),
  check('SET-03', 'Profile or settings overflows or logs errors at a supported width.', 'Both account pages have zero overflow/errors at all four widths.', (page, r) => viewportChecks(page, r, ['/profile', '/settings'])),
];
export default async function (args) { await runModule(args, checks, area); }
