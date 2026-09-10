import { FIXTURES, ROWS, check, assert, expect, note, probe, visit, navigateClick, getData, recordingRow, remember, restore, actionResponse, cleanTitle, viewportChecks, runModule } from '../lib.mjs';

export const area = 'recordings';
const path = `/api/recordings/${FIXTURES.recordingAssigned}`;
// Drift: static span headers, div rows/checkbox wrappers, no done status or player text.
const drifts = [{ id: 'DRIFT-REC', evidence: 'app/recordings/page.tsx: rows are div.cc-match-row[role="button"]; first child div toggles selection. Date/Title are static spans. Titles are cleanTitle(name); done rows omit status and every row omits player/Unassigned.', expected: 'Plan describes sortable headers and title/date/status/player columns.' }];
async function list(page, r) { await visit(page, '/recordings', r); await expect(page.getByPlaceholder('Search by title or filename')).toBeVisible(); }
async function choosePlayer(page, name) {
  await page.getByRole('button', { name: /^Player(?: ·|$)/ }).click();
  await page.getByRole('button', { name, exact: true }).click();
}

export const checks = [
  check('REC-01', 'Recording rows omit required metadata or disagree with the API.', '13 rows, each with title/date/status and player or Unassigned.', async (page, r) => {
    await list(page, r);
    const recordings = await getData(page, '/api/recordings', 'recordings');
    await expect(page.locator(ROWS)).toHaveCount(recordings.length);
    note(r, { apiCount: recordings.length, uiCount: await page.locator(ROWS).count() });
    assert(recordings.length === 13, `Fixture drift: expected 13, observed ${recordings.length}`);
    for (const recording of recordings) await probe(page, r, recording.id, async () => {
      const row = recordingRow(page, recording);
      await expect(row).toHaveCount(1);
      const text = await row.innerText();
      note(r, { id: recording.id, text });
      assert(text.includes(cleanTitle(recording.name)), 'Missing title');
      assert(text.toLowerCase().includes(new Date(recording.createdAt).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }).toLowerCase()), 'Missing date');
      assert(/done|processed|complete|processing|failed|pending/i.test(text), 'Missing status');
      assert(text.includes(recording.playerName ?? 'Unassigned'), 'Missing player/Unassigned');
    });
  }),
  check('REC-02', 'Recording search returns wrong counts or a blank empty state.', 'StMarys gives 2, zzzz gives an explicit empty message, clear restores 13.', async (page, r) => {
    await list(page, r);
    const search = page.getByPlaceholder('Search by title or filename');
    for (const [query, count] of [['StMarys', 2], ['zzzz', 0], ['', 13]]) await probe(page, r, query || 'clear', async () => {
      await search.fill(query); await expect(page.locator(ROWS)).toHaveCount(count);
      if (!count) await expect(page.getByText('No recordings match your filters.')).toBeVisible();
      note(r, { query, count: await page.locator(ROWS).count() });
    });
  }),
  check('REC-03', 'The player filter returns incorrect recordings.', 'Aileena gives 1, Unassigned matches the API, All players restores the complete list.', async (page, r) => {
    await list(page, r);
    const recordings = await getData(page, '/api/recordings', 'recordings');
    for (const [name, count] of [['Aileena Hu', 1], ['Unassigned', recordings.filter((x) => !x.player_id).length], ['All players', recordings.length]]) await probe(page, r, name, async () => {
      await choosePlayer(page, name); await expect(page.locator(ROWS)).toHaveCount(count); note(r, { name, count });
    });
  }),
  check('REC-04', 'Favorite persistence or filtering fails.', 'Favorite/unfavorite PATCH succeeds and filtering tracks the API; restore original favorite.', async (page, r, shared) => {
    await list(page, r);
    const original = await remember(shared, page, path, 'recording', ['favorited']);
    const fixture = await getData(page, path, 'recording');
    const all = await getData(page, '/api/recordings', 'recordings');
    const otherFavorites = all.filter((x) => x.id !== fixture.id && x.favorited).length;
    try {
      if (original.favorited) {
        await actionResponse(page, path, 'PATCH', () => recordingRow(page, fixture).getByLabel('Remove from favorites').click());
      }
      const res = await actionResponse(page, path, 'PATCH', () => recordingRow(page, fixture).getByLabel('Add to favorites').click());
      assert(res.status() === 200 && res.request().postDataJSON().favorited === true, 'Favorite PATCH did not return 200/true');
      await page.getByRole('button', { name: 'Favorites', exact: true }).click();
      await expect(page.locator(ROWS)).toHaveCount(otherFavorites + 1);
      const off = await actionResponse(page, path, 'PATCH', () => recordingRow(page, fixture).getByLabel('Remove from favorites').click());
      assert(off.status() === 200 && off.request().postDataJSON().favorited === false, 'Unfavorite PATCH failed');
      await expect(page.locator(ROWS)).toHaveCount(otherFavorites);
      if (!otherFavorites) await expect(page.getByText('No recordings match your filters.')).toBeVisible();
      note(r, { otherFavorites, filteredAfterFavorite: otherFavorites + 1, filteredAfterRemoval: otherFavorites });
      await page.getByRole('button', { name: 'Favorites', exact: true }).click();
    } finally { await restore(page, path, 'recording', original); }
  }),
  check('REC-05', 'Date and Title sorting controls do not reorder recordings.', 'Click Date and Title, and compare full row order against date/title sort.', async (page, r) => {
    await list(page, r);
    const recordings = await getData(page, '/api/recordings', 'recordings');
    const titles = () => page.locator(`${ROWS} span.font-display.truncate`).allTextContents();
    const before = await titles();
    await page.getByText('Date', { exact: true }).click();
    const dateOrder = await titles();
    await page.getByText('Title', { exact: true }).click();
    const titleOrder = await titles();
    note(r, { before, dateOrder, titleOrder });
    const alphabetical = recordings.map((x) => cleanTitle(x.name)).sort((a, b) => a.localeCompare(b));
    const chronological = [...recordings].sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt)).map((x) => cleanTitle(x.name));
    assert(JSON.stringify(dateOrder) === JSON.stringify(chronological) || JSON.stringify(dateOrder) === JSON.stringify([...chronological].reverse()), 'Date order is incorrect');
    assert(JSON.stringify(titleOrder) === JSON.stringify(alphabetical) || JSON.stringify(titleOrder) === JSON.stringify([...alphabetical].reverse()), 'Title order is incorrect');
    assert(dateOrder[0] !== titleOrder[0], 'First row never changed after sorting');
  }),
  check('REC-06', 'Bulk selection or cancellation does not work.', 'Select two, see count, open confirmation, Cancel and clear without DELETE.', async (page, r) => {
    await list(page, r);
    const count = await page.locator(ROWS).count();
    let deletes = 0;
    const observe = (req) => { if (req.method() === 'DELETE') deletes++; };
    page.on('request', observe);
    try {
      for (const i of [0, 1]) await page.locator(ROWS).nth(i).locator(':scope > div').first().click();
      await expect(page.getByRole('button', { name: 'Delete 2', exact: true })).toBeVisible();
      await expect(page.getByLabel('Clear selection')).toBeVisible();
      await page.getByRole('button', { name: 'Delete 2', exact: true }).click();
      await expect(page.getByText('Delete 2 recordings?', { exact: true })).toBeVisible();
      await page.getByRole('button', { name: 'Cancel', exact: true }).click();
      await page.getByLabel('Clear selection').click();
      await expect(page.locator(ROWS)).toHaveCount(count);
      assert(deletes === 0, `${deletes} DELETE requests fired`);
      note(r, 'Two selected; Delete 2 confirmation canceled; no DELETE requests.');
    } finally { page.off('request', observe); }
  }),
  check('REC-07', 'Recording rows cannot be opened by mouse and keyboard.', 'Click and real Tab/Enter both open the fixture recording.', async (page, r) => {
    await list(page, r);
    const fixture = await getData(page, path, 'recording');
    await navigateClick(page, recordingRow(page, fixture), `/recordings/${fixture.id}`);
    await list(page, r);
    await page.evaluate(() => document.activeElement?.blur());
    let reached = false;
    for (let i = 0; i < 180; i++) {
      await page.keyboard.press('Tab');
      if (await recordingRow(page, fixture).evaluate((el) => el === document.activeElement)) { reached = true; break; }
    }
    assert(reached, 'Fixture row never received focus in 180 Tab presses');
    await page.keyboard.press('Enter');
    await expect(page).toHaveURL((url) => url.pathname === `/recordings/${fixture.id}`);
  }, 'P2'),
  check('REC-08', 'The recordings Upload action is broken.', 'Upload CTA opens /upload.', async (page, r) => {
    await list(page, r); await navigateClick(page, page.locator('main a[href="/upload"]').first(), '/upload');
  }),
  check('REC-09', 'Recording rows overflow or hide mobile deletion controls.', 'No overflow at all widths; every phone delete control receives pointer events.', (page, r) => viewportChecks(page, r, ['/recordings'], { extra: async (p, width) => {
    if (width !== 390) return;
    const buttons = p.locator(`${ROWS} [aria-label="Delete recording"]`);
    assert(await buttons.count() > 0, 'No row delete controls');
    for (let i = 0; i < await buttons.count(); i++) await buttons.nth(i).click({ trial: true });
  } })),
];
export default async function (args) { await runModule(args, checks, area, { drifts }); }
