import { check, assert, expect, note, probe, visit, navigateClick, getData, garbageText, viewportChecks, runModule } from '../lib.mjs';

export const area = 'dashboard';
// Drift: TeamStrip has four team totals; the plan's three labels describe neither this strip nor its values.
const drifts = [{ id: 'DRIFT-DASH', evidence: 'components/dashboard/TeamStrip.tsx: Recordings this season, Players, Recordings analyzed, Hours recorded. app/page.tsx: WatchList CTAs link to /players/<id>; latest dates are on roster cards, not a Last recording team metric.', expected: 'Plan specifies In bounds, Recordings, Last recording and watch links to recording/upload.' }];

export const checks = [
  check('DASH-01', 'Team metrics differ from the required labels or show invalid values.', 'Exactly In bounds, Recordings, Last recording, each with a valid number/date.', async (page, r) => {
    await visit(page, '/', r);
    const strip = page.locator('section[aria-label="Team metrics"]');
    await expect(strip).toBeVisible();
    const cells = await strip.locator('[data-count-card]').evaluateAll((els) => els.map((el) => ({ label: el.children[0]?.textContent.trim(), value: el.children[1]?.textContent.trim() })));
    note(r, { cells, garbage: await garbageText(page) });
    assert(cells.every((cell) => /\d/.test(cell.value)), 'Missing numeric/date metric values');
    assert((await garbageText(page)).length === 0, 'Invalid visible values');
    assert(JSON.stringify(cells.map((c) => c.label)) === JSON.stringify(['In bounds', 'Recordings', 'Last recording']), `Observed labels: ${cells.map((c) => c.label).join(', ')}`);
  }),
  check('DASH-02', 'Dashboard roster is incomplete or a player card does not open.', 'Nine API-backed player cards, each linking to its player.', async (page, r) => {
    await visit(page, '/', r);
    const players = await getData(page, '/api/players', 'players');
    const cards = page.locator('[aria-label="Roster"] a[href^="/players/"]');
    await expect(cards).toHaveCount(players.length);
    assert(players.length === 9, `Fixture drift: expected 9 players, API has ${players.length}`);
    for (const player of players) await expect(cards.filter({ hasText: player.name.split(' ').at(-1) })).toHaveCount(1);
    const href = await cards.first().getAttribute('href');
    await navigateClick(page, cards.first(), href);
    note(r, `${players.length} roster cards; first card opened ${href}`);
  }),
  check('DASH-03', 'A dashboard watch-list action is missing or dead.', 'Every watch-list item has a working recording/upload/player CTA.', async (page, r) => {
    await visit(page, '/', r);
    const list = page.locator('[aria-label="Watch list"]');
    await expect(list).toBeVisible();
    const items = list.locator('article');
    assert(await items.count() > 0, 'Watch list is empty');
    const hrefs = await items.locator('a').evaluateAll((els) => els.map((el) => el.getAttribute('href')));
    assert(hrefs.length === await items.count(), 'A watch-list item lacks its CTA');
    for (const href of hrefs) await probe(page, r, href, async () => {
      await visit(page, '/', r);
      assert(/^\/(recordings\/|players\/|upload$)/.test(href), `Unexpected CTA ${href}`);
      await navigateClick(page, list.locator(`a[href="${href}"]`).first(), href);
      note(r, `Watch CTA opened ${href}`);
    });
  }),
  check('DASH-04', 'Dashboard shows an empty state or invalid summary totals despite live recordings.', 'Populated dashboard with all numeric totals matching the recording API.', async (page, r) => {
    await visit(page, '/', r);
    const summary = await getData(page, '/api/dashboard/summary');
    const recordings = await getData(page, '/api/recordings', 'recordings');
    note(r, { totals: summary.totals, recordingCount: recordings.length });
    for (const key of ['total', 'done', 'processing', 'failed']) assert(Number.isFinite(summary.totals?.[key]), `totals.${key} is not a number`);
    assert(recordings.length > 0 && summary.totals.total === recordings.length, 'Summary/list totals disagree');
    await expect(page.locator('[aria-label="Team metrics"]')).toBeVisible();
    // Roster cards legitimately say No recordings yet; only test the page-level empty state.
    await expect(page.getByRole('heading', { name: 'Your first recording.', exact: true })).toHaveCount(0);
    await expect(page.getByRole('link', { name: 'Upload your first recording', exact: true })).toHaveCount(0);
  }),
  check('DASH-05', 'Dashboard overflows or logs errors on a supported viewport.', 'Zero horizontal overflow and console/page errors at all four widths.', (page, r) => viewportChecks(page, r, ['/'])),
];
export default async function (args) { await runModule(args, checks, area, { drifts }); }
