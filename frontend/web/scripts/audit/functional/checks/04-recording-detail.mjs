import { FIXTURES, BOGUS, DOTS, RECORDING_VIDEO, check, assert, expect, note, probe, visit, getData, actionResponse, remember, restore, cleanTitle, garbageText, viewportChecks, videoReady, seekControl, seekAway, assertSeek, runModule } from '../lib.mjs';

export const area = 'recording-detail';
const id = FIXTURES.recordingAssigned;
const path = `/recordings/${id}`;
const api = `/api/recordings/${id}`;
const drifts = [{ id: 'DRIFT-DET', evidence: 'app/recordings/[id]/page.tsx: h1 normalizes filename; no player badge/Unassigned label and only left-handed pill. Stats are Winners, Unforced errors, Avg rally length, Rallies won. VizPanel.tsx: position in Coverage; Marker key in Spacing; tabs have no arrow handler; Play from here closes selection. VideoPlayer.tsx: two Play buttons when paused, scope video to .cc-video-hero to exclude brand video. Not-found text is Recording not found.', expected: 'Plan names raw API title, assigned/right-handed badges, four different totals, both legends in Shot map and a different not-found sentence.' }];
async function detail(page, r, recordingId = id) {
  await visit(page, `/recordings/${recordingId}`, r);
  await expect(page.locator('main h1')).toBeVisible();
}
async function setName(page, value) {
  await page.getByLabel('Rename recording', { exact: true }).click();
  await page.getByLabel('Recording name', { exact: true }).fill(value);
  const res = await actionResponse(page, api, 'PATCH', () => page.getByLabel('Recording name', { exact: true }).press('Enter'));
  assert(res.status() === 200 && res.request().postDataJSON().name === value, `Rename returned ${res.status()}`);
  await expect(page.locator('main h1')).toHaveText(cleanTitle(value));
}

export const checks = [
  check('DET-01', 'Recording identity or player handedness is missing or incorrect.', 'Correct title, Recordings breadcrumb, Aileena badge and right/left handedness.', async (page, r) => {
    await detail(page, r);
    const data = await getData(page, api, 'recording');
    note(r, { name: data.name, title: await page.locator('main h1').innerText(), playerId: data.playerId, playerHandedness: data.playerHandedness });
    await expect(page.locator('nav[aria-label="Breadcrumb"] a[href="/recordings"]')).toBeVisible();
    await expect(page.locator('main h1')).toHaveText(cleanTitle(data.name));
    await probe(page, r, 'assigned identity', async () => {
      await expect(page.locator('main').getByText('Aileena Hu', { exact: true })).toBeVisible();
      // Fix branch: the header names the player (linked); only lefties get a handedness pill.
    });
    const recordings = await getData(page, '/api/recordings', 'recordings');
    const left = recordings.find((rec) => rec.player_id === FIXTURES.playerLeft && rec.status === 'done');
    if (left) await probe(page, r, 'left-handed pill', async () => {
      await detail(page, r, left.id); await expect(page.getByText('Left-handed', { exact: true })).toBeVisible();
    });
    else note(r, 'Left-handed recording subcheck skipped: no done recording assigned to Kaia.');
  }),
  check('DET-02', 'Recording video playback or custom controls fail.', 'Metadata, play/pause, seek, 1.5× speed, volume and prevented native context menu.', async (page, r) => {
    await detail(page, r); await videoReady(page);
    const video = page.locator(RECORDING_VIDEO);
    const initial = await video.evaluate((v) => ({ readyState: v.readyState, duration: v.duration, paused: v.paused }));
    note(r, initial);
    await page.getByLabel('Play', { exact: true }).last().click();
    await expect.poll(() => video.evaluate((v) => v.paused)).toBe(false);
    await page.getByLabel('Pause', { exact: true }).click();
    await expect.poll(() => video.evaluate((v) => v.paused)).toBe(true);
    const beforeSeek = await video.evaluate((v) => v.currentTime);
    const target = await seekControl(page, 0.45);
    assert(Math.abs(target - beforeSeek) > 0.2, 'Seek control did not change currentTime');
    await page.getByLabel('Playback speed', { exact: true }).click();
    await expect(page.getByRole('menu')).toBeVisible();
    assert(await page.getByRole('menuitemradio').count() >= 2, 'Speed choices missing');
    await page.getByRole('menuitemradio', { name: '1.5×', exact: true }).click();
    await expect.poll(() => video.evaluate((v) => v.playbackRate)).toBe(1.5);
    await expect(page.getByLabel('Volume', { exact: true })).toBeVisible();
    const prevented = await video.evaluate((v) => !v.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true, button: 2 })));
    assert(prevented, 'contextmenu default was not prevented');
  }, 'P0'),
  check('DET-03', 'Court visualization tabs do not switch by click or arrow keys.', 'Three tabs show their respective panels and update selection; arrows move focus.', async (page, r) => {
    await detail(page, r);
    const tabs = page.getByRole('tablist', { name: 'Court visualization mode' });
    await expect(tabs.getByRole('tab')).toHaveCount(3);
    for (const [name, title] of [['Shot map', 'Where shots landed'], ['Spacing', 'Contact spacing'], ['Coverage', 'Court coverage']]) {
      const tab = tabs.getByRole('tab', { name, exact: true });
      await tab.click(); await expect(tab).toHaveAttribute('aria-selected', 'true');
      await expect(page.getByRole('heading', { name: title, exact: true })).toBeVisible();
      await expect(tabs.locator('[aria-selected="true"]')).toHaveCount(1);
    }
    await tabs.getByRole('tab', { name: 'Shot map', exact: true }).click();
    await page.keyboard.press('ArrowRight');
    await expect(tabs.getByRole('tab', { name: 'Spacing', exact: true })).toBeFocused();
  }),
  check('DET-04', 'Shot-map bounce selection, keyboard access or video seeking fails.', 'Real focusable dots open stroke/result/time details, seek video and clear; both mode-specific legends render.', async (page, r) => {
    await detail(page, r); await videoReady(page);
    const dots = page.locator(DOTS);
    assert(await dots.count() > 0, 'No focusable real shot-map dots');
    note(r, { dotCount: await dots.count() });
    const dot = dots.first();
    const label = await dot.getAttribute('aria-label');
    const target = Number(label.match(/at ([\d.]+)s/)[1]);
    await seekAway(page, target);
    await dot.click();
    const panel = page.getByRole('region', { name: 'Selected bounce detail' });
    await expect(panel).toBeVisible();
    const text = await panel.innerText();
    note(r, { label, panel: text });
    assert(/Forehand|Backhand|Serve\/Overhead/.test(text) && /\b(In|Out)\b/.test(text) && /\d{2}:\d{2}/.test(text), 'Bounce metadata incomplete');
    await panel.getByRole('button', { name: 'Play from here' }).click();
    await assertSeek(page, target);
    await expect.poll(() => page.locator(RECORDING_VIDEO).evaluate((v) => { const b = v.getBoundingClientRect(); return b.top >= 0 && b.bottom <= innerHeight; })).toBe(true);
    await dot.click(); await panel.getByLabel('Clear selection').click(); await expect(panel).toHaveCount(0);
    await dot.focus(); await page.keyboard.press('Enter'); await expect(panel).toBeVisible();
    await panel.getByLabel('Clear selection').click();
    await expect(page.getByLabel('Bounce in/out key')).toBeVisible();
    await page.getByRole('tab', { name: 'Spacing', exact: true }).click();
    await expect(page.getByLabel('Marker key')).toBeVisible();
  }),
  check('DET-05', 'Coach insight tiles contain missing or invalid metrics.', 'Court position, net game and errors render real numeric values/explicit zero states; percentages 0–100.', async (page, r) => {
    await detail(page, r);
    for (const label of ['Your errors', 'Net game']) await probe(page, r, label, async () => {
      const tile = page.locator('.cc-coach-tile').filter({ hasText: label });
      await expect(tile).toBeVisible();
      const text = await tile.innerText(); note(r, { label, text });
      assert(/\d|No out-of-bounds bounces|No net approaches detected/.test(text), 'No numeric/explicit-zero insight');
      assert(!/No net approach data|Reprocess/.test(text), 'Required insight data unavailable');
    });
    await page.getByRole('tab', { name: 'Coverage', exact: true }).click();
    const position = page.locator('.cc-coach-tile').filter({ hasText: 'Court position' });
    await expect(position).toBeVisible();
    assert(/\d/.test(await position.innerText()), 'No numeric court position');
    const text = await page.locator('main').innerText();
    const percentages = [...text.matchAll(/(-?\d+(?:\.\d+)?)%/g)].map((m) => Number(m[1]));
    note(r, { percentages, garbage: await garbageText(page) });
    assert(percentages.length > 0 && percentages.every((n) => n >= 0 && n <= 100), 'Percentage outside 0–100');
    assert((await garbageText(page)).length === 0, 'Garbage text in detail');
  }),
  check('DET-06', 'Rally rows cannot expand or seek, or expose backend labels.', 'Click and Enter expand shot sequence; timestamps seek; human-readable reasons.', async (page, r) => {
    await detail(page, r); await videoReady(page);
    const row = page.locator('tr[role="button"]').first();
    await expect(row).toBeVisible();
    await row.click(); await expect(row).toHaveAttribute('aria-expanded', 'true');
    assert(await page.locator('main ol li').count() > 0, 'Expanded shot sequence empty');
    await row.focus(); await page.keyboard.press('Enter'); await expect(row).toHaveAttribute('aria-expanded', 'false');
    await page.keyboard.press('Enter'); await expect(row).toHaveAttribute('aria-expanded', 'true');
    const chip = row.locator('button[title^="Jump to"]');
    const parts = (await chip.innerText()).trim().split(':').map(Number);
    const target = parts[0] * 60 + parts[1];
    await seekAway(page, target); await chip.click(); await assertSeek(page, target);
    const reasons = await page.locator('tr[role="button"] td:nth-child(5)').allTextContents();
    note(r, { reasons, timestamp: target });
    assert(reasons.length > 0 && reasons.every((text) => text.trim() && !text.includes('_')), 'Raw or missing end-reason labels');
  }),
  check('DET-07', 'Scouting report shows raw markdown or lacks recovery guidance.', 'Rendered headings/prose, or unavailable state instructing reprocessing.', async (page, r) => {
    await detail(page, r);
    const report = page.getByLabel('Scouting report', { exact: true });
    if (await report.count()) {
      const text = await report.innerText(); note(r, text);
      assert(await report.locator('h2,h3').count() > 0 && await report.locator('p').count() > 0, 'Report is not structured prose');
      assert(!/\*\*|##/.test(text), 'Raw markdown in report');
    } else {
      const unavailable = page.getByLabel('Scouting report unavailable', { exact: true });
      await expect(unavailable).toBeVisible();
      const text = await unavailable.innerText(); note(r, text);
      assert(/reprocess|upload.*again|try again/i.test(text), 'No actionable recovery guidance');
    }
  }),
  check('DET-08', 'Detail statistics are missing or disagree with recording data.', 'In-bounds, shots, rallies and average rally length agree with API fields.', async (page, r) => {
    await detail(page, r);
    const data = await getData(page, api, 'recording');
    const tiles = await page.locator('.cc-stat-tile').evaluateAll((els) => els.map((el) => ({ label: el.children[0]?.textContent.trim(), value: el.children[1]?.textContent.trim() })));
    note(r, { tiles, inBoundsBounces: data.inBoundsBounces, shots: data.shots?.length, rallySummary: data.rallySummary });
    const totals = page.locator('.cc-card').filter({ has: page.getByRole('heading', { name: 'Totals', exact: true }) });
    await expect(totals).toContainText(`${data.shots.length.toLocaleString()} shots`);
    const bounceTotal = (data.inBoundsBounces ?? 0) + (data.outBoundsBounces ?? 0);
    const inPct = bounceTotal > 0 ? Math.round((data.inBoundsBounces / bounceTotal) * 100) : NaN;
    for (const [label, expected] of [['In bounds', inPct], ['Rallies', data.rallySummary?.total], ['Avg rally length', data.rallySummary?.avg_length]]) await probe(page, r, label, async () => {
      const tile = tiles.find((x) => x.label.toLowerCase() === label.toLowerCase());
      assert(tile && Number.isFinite(expected), `Missing ${label} tile or API value`);
      const leading = tile.value.replace(/,/g, '').match(/^-?\d+(?:\.\d+)?/)?.[0];
      assert(leading !== undefined && Math.abs(Number(leading) - expected) < 0.11, `${label}: UI ${tile.value}, API ${expected}`);
      if (label === 'In bounds') assert(tile.value.includes(`${data.inBoundsBounces} of ${bounceTotal}`), `In bounds unit missing: ${tile.value}`);
    });
  }, 'P0'),
  check('DET-09', 'Rename does not persist, restore or cancel correctly.', 'Rename via Enter returns PATCH 200; restore original; Escape sends no PATCH.', async (page, r, shared) => {
    await detail(page, r);
    const original = await remember(shared, page, api, 'recording', ['name']);
    assert(original.name.length + 5 <= 100, 'Fixture name cannot accommodate the requested (qa) suffix');
    try {
      await setName(page, `${original.name} (qa)`);
      assert((await getData(page, api, 'recording')).name === `${original.name} (qa)`, 'Renamed title did not persist');
      await setName(page, original.name);
      let requests = 0;
      const listener = (req) => { if (req.method() === 'PATCH' && new URL(req.url()).pathname === api) requests++; };
      page.on('request', listener);
      try {
        await page.getByLabel('Rename recording').click(); await page.getByLabel('Recording name').fill('QA canceled edit');
        await page.getByLabel('Recording name').press('Escape');
        await expect(page.getByLabel('Recording name')).toHaveCount(0);
        await page.waitForTimeout(1000); assert(requests === 0, `Escape sent ${requests} PATCH requests`);
      } finally { page.off('request', listener); }
    } finally { await restore(page, api, 'recording', original); }
  }),
  check('DET-10', 'Timed notes fail to save, seek or delete.', 'QA note appears at video time, seeks on click, and deletion restores the original notes array.', async (page, r, shared) => {
    await detail(page, r); await videoReady(page);
    const original = await remember(shared, page, api, 'recording', ['notes']);
    try {
      const target = await page.locator(RECORDING_VIDEO).evaluate((v) => { v.pause(); v.currentTime = Math.min(5, v.duration / 3); return v.currentTime; });
      await page.getByLabel('Add a note', { exact: true }).fill('QA note');
      const res = await actionResponse(page, api, 'PATCH', () => page.getByLabel('Add a note', { exact: true }).press('Enter'));
      assert(res.status() === 200, `Notes save returned ${res.status()}`);
      const updated = await getData(page, api, 'recording');
      assert(updated.notes.length === original.notes.length + 1, 'Note count did not increase');
      const added = updated.notes.at(-1);
      assert(added.text === 'QA note' && Math.abs(added.timestamp_sec - target) < 0.2, 'Note timestamp/text not persisted');
      const rows = page.locator('.cc-note-row').filter({ hasText: 'QA note' });
      const row = rows.filter({ hasText: `${String(Math.floor(target / 60)).padStart(2, '0')}:${String(Math.floor(target % 60)).padStart(2, '0')}` }).last();
      await expect(row).toBeVisible(); await seekAway(page, added.timestamp_sec); await row.locator('span').first().click(); await assertSeek(page, added.timestamp_sec);
      const deleted = await actionResponse(page, api, 'PATCH', () => row.getByLabel('Delete note').click());
      assert(deleted.status() === 200, 'Delete note PATCH failed');
      assert(JSON.stringify((await getData(page, api, 'recording')).notes) === JSON.stringify(original.notes), 'Original notes were not restored');
      note(r, { originalCount: original.notes.length, addedTimestamp: added.timestamp_sec });
    } finally { await restore(page, api, 'recording', original); }
  }),
  check('DET-11', 'Reprocessing an expired upload lacks a friendly error or leaves it processing.', 'Old raw returns 409, visible recovery text, recording remains done.', async (page, r) => {
    await detail(page, r);
    const control = page.getByRole('button', { name: 'Reprocess', exact: true });
    if (!await control.count()) return { skip: 'No reprocess control is present on this recording.' };
    await control.click(); await expect(page.getByText('Reprocess this recording?')).toBeVisible();
    const res = await actionResponse(page, '/api/trigger-process', 'POST', () => control.click(), 30000);
    const body = await res.json(); note(r, { status: res.status(), body });
    assert(res.status() === 409 && /original upload is no longer available/i.test(body.error), 'Old upload did not return expected 409');
    await expect(page.getByText(body.error, { exact: true })).toBeVisible();
    assert((await getData(page, api, 'recording')).status === 'done', 'Recording left in processing');
    await expect(page.getByText('Analyzing your recording.', { exact: true })).toHaveCount(0);
    await page.getByRole('button', { name: 'Cancel', exact: true }).click();
  }),
  check('DET-12', 'An unknown recording does not show a friendly recovery state.', 'Recording not found with Back to recordings and no console/page error.', async (page, r) => {
    const data = await visit(page, `/recordings/${BOGUS}`, r);
    await expect(page.getByText(/Recording not found|We couldn't load that recording/)).toBeVisible();
    await expect(page.locator('main a[href="/recordings"]')).toBeVisible();
    note(r, { text: await page.locator('main').innerText(), errors: data.errors });
    assert(data.errors.length === 0, data.errors.join('; '));
  }),
  check('DET-13', 'An unassigned recording has missing or misleading player identity.', 'Unassigned shown, no linked player badge, no invalid visible values.', async (page, r) => {
    await detail(page, r, FIXTURES.recordingUnassigned);
    const data = await getData(page, `/api/recordings/${FIXTURES.recordingUnassigned}`, 'recording');
    assert(data.playerId === null, `Fixture is assigned to ${data.playerId}`);
    await expect(page.locator('main a[href^="/players/"]')).toHaveCount(0);
    note(r, { garbage: await garbageText(page), main: (await page.locator('main').innerText()).slice(0, 700) });
    assert((await garbageText(page)).length === 0, 'Invalid text on unassigned recording');
    await expect(page.getByText('Unassigned', { exact: true })).toBeVisible();
  }),
  check('DET-14', 'Recording detail overflows or logs errors on a supported viewport.', 'Zero overflow/errors and full-page screenshots at all four widths.', (page, r) => viewportChecks(page, r, [path])),
];
export default async function (args) { await runModule(args, checks, area, { drifts }); }
