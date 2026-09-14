import { FIXTURES, BOGUS, check, assert, expect, note, visit, getData, navigateClick, actionResponse, remember, restore, viewportChecks, runModule } from '../lib.mjs';

export const area = 'players';
const drifts = [{ id: 'DRIFT-PLY', evidence: 'app/players/[id]/page.tsx RecentTrendBars renders CSS div bars within links, not svg rect. Handedness radios are named Right-handed and Left-handed. app/players/page.tsx populated CTA reads Upload; Upload a recording appears in empty states.', expected: 'Plan names SVG trend bars, Left/Right and Upload a recording on the populated list.' }];
const cards = (page) => page.locator('main a[href^="/players/"]');
async function players(page, r) { await visit(page, '/players', r); await expect(page.getByPlaceholder('Search by name')).toBeVisible(); }

export const checks = [
  check('PLY-01', 'Player search or sorting gives incorrect results.', 'Nine cards, Kaia gives one, zzzz shows empty message, sorting matches API aggregates.', async (page, r) => {
    await players(page, r);
    const roster = await getData(page, '/api/players', 'players');
    const recordings = await getData(page, '/api/recordings', 'recordings');
    await expect(cards(page)).toHaveCount(9); assert(roster.length === 9, `API roster count ${roster.length}`);
    const search = page.getByPlaceholder('Search by name');
    await search.fill('Kaia'); await expect(cards(page)).toHaveCount(1); await expect(cards(page).first()).toContainText('Kaia');
    await search.fill('zzzz'); await expect(cards(page)).toHaveCount(0); await expect(page.getByText('No players match those filters.')).toBeVisible();
    await search.fill(''); await expect(cards(page)).toHaveCount(9);
    const before = await cards(page).evaluateAll((els) => els.map((el) => el.getAttribute('href')));
    await page.getByLabel('Sort by').selectOption('recordings');
    const sorted = await cards(page).evaluateAll((els) => els.map((el) => el.getAttribute('href').split('/').at(-1)));
    const counts = sorted.map((id) => recordings.filter((rec) => rec.player_id === id).length);
    note(r, { before, sorted, counts });
    assert(counts.every((n, i) => i === 0 || counts[i - 1] >= n), 'Most recorded ordering is wrong');
    assert(JSON.stringify(before) !== JSON.stringify(sorted.map((id) => `/players/${id}`)), 'Sort did not change order');
    await page.getByLabel('Sort by').selectOption('name');
    const names = (await cards(page).evaluateAll((els) => els.map((el) => el.getAttribute('href')))).map((href) => roster.find((p) => href === `/players/${p.id}`).name);
    assert(JSON.stringify(names) === JSON.stringify([...names].sort((a, b) => a.localeCompare(b))), 'Name A–Z order is wrong');
  }),
  check('PLY-02', 'Player-card or upload navigation is broken.', 'Player card opens its profile and populated Upload CTA opens /upload.', async (page, r) => {
    await players(page, r);
    await navigateClick(page, cards(page).filter({ hasText: 'Kaia' }), `/players/${FIXTURES.playerLeft}`);
    await players(page, r); await navigateClick(page, page.locator('main a[href="/upload"]').first(), '/upload');
  }),
  check('PLY-03', 'Kaia’s profile has incorrect handedness, history or trend.', 'Kaia Wolfe, Left selected, API-backed history and rendered trend bars or an honest empty state.', async (page, r) => {
    await visit(page, `/players/${FIXTURES.playerLeft}`, r);
    await expect(page.locator('main h1')).toHaveText('Kaia Wolfe');
    await expect(page.getByRole('radiogroup', { name: 'Player handedness' }).getByRole('radio', { name: 'Left-handed' })).toHaveAttribute('aria-checked', 'true');
    const recs = (await getData(page, '/api/recordings', 'recordings')).filter((x) => x.player_id === FIXTURES.playerLeft);
    await expect(page.locator('a.cc-match-row')).toHaveCount(recs.length);
    if (recs.length) {
      const trend = page.locator('section').filter({ has: page.getByRole('heading', { name: 'Recent accuracy', exact: true }) });
      const bars = trend.locator('a[href^="/recordings/"] div[style*="height:"]');
      await expect(bars).toHaveCount(Math.min(5, recs.length));
      const heights = await bars.evaluateAll((els) => els.map((el) => el.getBoundingClientRect().height));
      assert(heights.every((h) => h > 0), `Trend bars have zero height: ${heights}`);
      note(r, { recordingCount: recs.length, barHeights: heights });
    } else {
      await expect(page.getByText('No recordings yet.', { exact: true })).toBeVisible();
      note(r, 'No assigned recordings; explicit empty recording history shown.');
    }
  }),
  check('PLY-04', 'Handedness changes do not persist or restore.', 'Aileena switches Left then Right via PATCH 200 and ends right-handed.', async (page, r, shared) => {
    await visit(page, `/players/${FIXTURES.playerRight}`, r);
    const path = `/api/players/${FIXTURES.playerRight}`;
    const original = await remember(shared, page, path, 'player', ['handedness']);
    assert(original.handedness === 'right', `Fixture drift: Aileena originally ${original.handedness}; refusing to assume right.`);
    try {
      for (const [name, value] of [['Left-handed', 'left'], ['Right-handed', 'right']]) {
        const control = page.getByRole('radiogroup', { name: 'Player handedness' }).getByRole('radio', { name, exact: true });
        const res = await actionResponse(page, path, 'PATCH', () => control.click());
        assert(res.status() === 200 && res.request().postDataJSON().handedness === value, `PATCH ${value} returned ${res.status()}`);
        await expect(control).toHaveAttribute('aria-checked', 'true');
        assert((await getData(page, path, 'player')).handedness === value, `GET did not persist ${value}`);
      }
    } finally { await restore(page, path, 'player', original); }
  }),
  check('PLY-05', 'An unknown player has no friendly recovery state.', 'Player not found, Back to roster and zero console/page errors.', async (page, r) => {
    const data = await visit(page, `/players/${BOGUS}`, r);
    await expect(page.getByText('Player not found.', { exact: true })).toBeVisible();
    await expect(page.locator('main a[href="/players"]').filter({ hasText: 'Back to roster' })).toBeVisible();
    note(r, { text: await page.locator('main').innerText(), errors: data.errors });
    assert(data.errors.length === 0, data.errors.join('; '));
  }),
  check('PLY-06', 'Player list or profile overflows or logs errors.', 'Both list and Kaia profile pass all four viewports.', (page, r) => viewportChecks(page, r, ['/players', `/players/${FIXTURES.playerLeft}`])),
];
export default async function (args) { await runModule(args, checks, area, { drifts }); }
