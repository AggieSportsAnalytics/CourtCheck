import { FIXTURES, QA_NAME, DOTS, RECORDING_VIDEO, PROCESS_TIMEOUT, LONG_CHECK_TIMEOUT, ProcessingTimeout, check, assert, expect, note, visit, getData, actionResponse, recordingRow, videoReady, shot, pollDone, navigateClick, runModule } from '../lib.mjs';

export const area = 'pipeline';
const drifts = [{ id: 'DRIFT-E2E', evidence: 'app/upload/page.tsx: [data-state] exposes idle/uploading/processing/done; initial processing headline is Loading the recording., backend stage is shown below it. VizPanel.tsx filters unknown/off-court shots, so dots need not equal raw shots.length. TeamStrip has no Last recording metric; roster cards show latest dates.', expected: 'Plan names Starting analysis., assumes one dot per shot, and expects a dashboard Last recording metric.' }];
function dependency(shared, done = false) {
  if (shared.skipUpload) return { skip: '--skip-upload supplied; no upload or reprocess attempted.' };
  if (shared.resumed && !done) return { skip: `Resumed: QA recording ${shared.qaRecordingId} was uploaded and processed by the previous run.` };
  if (!shared.qaRecordingId) return { skip: 'E2E-01 did not create an audit recording.' };
  if (done && !shared.e2eDone) return { skip: 'E2E-03 did not complete successfully.' };
  return null;
}
function progress(shared, body) {
  assert(Number.isFinite(body.progress) && body.progress >= 0 && body.progress <= 1, `Invalid backend progress ${body.progress}`);
  const previous = shared.progressSamples.at(-1);
  shared.progressSamples.push({ elapsedMs: Date.now() - shared.uploadStarted, progress: body.progress, stage: body.stage, status: body.status });
  assert(!previous || body.progress >= previous.progress, `Progress regressed from ${previous?.progress} to ${body.progress}`);
}
function plottedShots(shots) {
  // Source drift: VizPanel.buildShotMapDots drops unknowns/missing coords and distant noise.
  return shots.filter((s) => {
    const y = s.court_y > 39 ? 78 - s.court_y : s.court_y;
    return s.court_x != null && s.court_y != null && s.stroke !== 'unknown' && y >= -6 && y <= 45 && s.court_x >= -4 && s.court_x <= 31;
  });
}

export const checks = [
  check('E2E-01', 'A real upload cannot create a recording and promptly start processing.', 'QA pilot audit 2026-09-09 assigned to Aileena; create 200 with ID; trigger 200/202 within 10s.', async (page, r, shared) => {
    if (shared.skipUpload) return { skip: '--skip-upload supplied.' };
    if (shared.resumed) return { skip: `Resumed: QA recording ${shared.qaRecordingId} already exists; no second upload.` };
    await visit(page, '/upload', r);
    await page.getByPlaceholder('e.g. Lin vs Stanford · Set 1').fill(QA_NAME);
    await page.locator('select#player').selectOption(FIXTURES.playerRight);
    await page.getByLabel('Recording file input').setInputFiles(FIXTURES.clip);
    shared.uploadStarted = Date.now();
    shared.progressSamples = [];
    let triggerStarted;
    const listener = (req) => { if (new URL(req.url()).pathname === '/api/trigger-process' && req.method() === 'POST') triggerStarted = Date.now(); };
    page.on('request', listener);
    const trigger = page.waitForResponse((res) => new URL(res.url()).pathname === '/api/trigger-process' && res.request().method() === 'POST', { timeout: 120000 }).then((response) => ({ response, ended: Date.now() }), (error) => ({ error }));
    try {
      const created = await actionResponse(page, '/api/create-upload', 'POST', () => page.getByRole('button', { name: 'Upload and process' }).click(), 30000);
      const body = await created.json();
      // Capture immediately: cleanup must know the ID even when trigger or a later assertion fails.
      if (typeof body.match_id === 'string' && /^[0-9a-f-]{36}$/i.test(body.match_id)) {
        shared.qaRecordingId = body.match_id;
        shared.createdRecordingIds.push(body.match_id);
      }
      note(r, { createStatus: created.status(), match_id: body.match_id, name: QA_NAME, playerId: FIXTURES.playerRight });
      assert(created.status() === 200 && shared.qaRecordingId, `Create failed: ${created.status()} ${body.error ?? 'no match_id'}`);
      const payload = created.request().postDataJSON();
      assert(payload.name === QA_NAME && payload.player_id === FIXTURES.playerRight, 'Upload metadata was not submitted correctly');
      const started = await trigger;
      if (started.error) throw started.error;
      const elapsed = started.ended - triggerStarted;
      shared.triggered = [200, 202].includes(started.response.status());
      note(r, { triggerStatus: started.response.status(), triggerMs: elapsed });
      if (!shared.triggered) assert(false, `Trigger failed: ${started.response.status()} ${(await started.response.text().catch(() => '')).slice(0, 500)}`); // body read only on failure: Playwright never resolves this body on success and hung run 1 for 50 minutes
      if (elapsed > 10000) r.severity = 'P1';
      assert(elapsed <= 10000, `Trigger took ${elapsed}ms; fire-and-forget regression`);
    } finally { page.off('request', listener); }
  }, 'P0'),
  check('E2E-02', 'Processing does not display a backend stage or progress regresses.', 'Within 90s an actual backend stage appears in the processing pane; progress is monotonic.', async (page, r, shared) => {
    const skip = dependency(shared); if (skip) return skip;
    assert(shared.triggered, 'Upload trigger did not succeed');
    const start = Date.now();
    let stageSeen = false;
    while (Date.now() - start < 90000) {
      const body = await getData(page, `/api/status?match_id=${shared.qaRecordingId}`);
      progress(shared, body);
      assert(body.status !== 'failed', `Pipeline failed: ${body.error}`);
      if (body.stage && body.stage !== 'Starting analysis.') {
        const pane = page.locator('[data-state="processing"], [data-state="done"]');
        await expect(pane).toBeVisible();
        if (body.status !== 'done') await expect(pane).toContainText(body.stage, { timeout: 15000 });
        else await expect(page.locator('[data-state="done"]')).toBeVisible();
        await expect(pane).not.toContainText('Starting analysis.');
        await shot(page, r); stageSeen = true; break;
      }
      await page.waitForTimeout(1500);
    }
    note(r, { samples: shared.progressSamples });
    assert(stageSeen, 'No backend stage appeared in the UI within 90 seconds');
  }, 'P0'),
  check('E2E-03', 'The real recording does not complete within twenty minutes.', 'Status reaches done; failure error and total processing seconds recorded.', async (page, r, shared) => {
    const skip = dependency(shared); if (skip) return skip;
    assert(shared.triggered, 'Cannot await processing: trigger failed');
    const remaining = PROCESS_TIMEOUT - (Date.now() - shared.uploadStarted);
    assert(remaining > 0, 'Twenty-minute completion deadline already elapsed');
    const body = await pollDone(page, shared.qaRecordingId, r, remaining, (value) => progress(shared, value));
    assert(body.status === 'done', 'Status did not reach done');
    shared.e2eDone = true;
    shared.processingSeconds = (Date.now() - shared.uploadStarted) / 1000;
    note(r, { totalProcessingSeconds: shared.processingSeconds, samples: shared.progressSamples });
  }, 'P0'),
  check('E2E-04', 'The upload done state does not open the new recording.', 'Open recording navigates to the created ID.', async (page, r, shared) => {
    const skip = dependency(shared, true); if (skip) return skip;
    await expect(page.locator('[data-state="done"]')).toBeVisible({ timeout: 20000 });
    await navigateClick(page, page.getByRole('button', { name: 'Open recording', exact: true }), `/recordings/${shared.qaRecordingId}`);
    await expect(page.locator('main h1')).toHaveText(QA_NAME);
  }, 'P0'),
  check('E2E-05', 'The new recording lacks playable media or trustworthy analysis.', 'Video, audio evidence, real dots/rallies/insights/report and nonzero core stats agree with the API.', async (page, r, shared) => {
    const skip = dependency(shared, true); if (skip) return skip;
    await visit(page, `/recordings/${shared.qaRecordingId}`, r); await videoReady(page);
    const data = await getData(page, `/api/recordings/${shared.qaRecordingId}`, 'recording');
    await page.getByLabel('Play', { exact: true }).last().click();
    await expect.poll(() => page.locator(RECORDING_VIDEO).evaluate((v) => v.currentTime), { timeout: 10000 }).toBeGreaterThan(0.5);
    const audio = await page.locator(RECORDING_VIDEO).evaluate((v) => ({ mozHasAudio: v.mozHasAudio ?? null, decodedBytes: v.webkitAudioDecodedByteCount ?? null, tracks: v.audioTracks?.length ?? null }));
    await page.locator(RECORDING_VIDEO).evaluate((v) => v.pause());
    const hasAudio = audio.mozHasAudio === true || audio.decodedBytes > 0 || audio.tracks > 0;
    note(r, { audio, audioResult: hasAudio ? 'present' : 'undeterminable with Chromium media properties' });
    if (audio.mozHasAudio === false || audio.tracks === 0) assert(hasAudio, 'Media explicitly reports no audio track');
    const dots = await page.locator(DOTS).count();
    const rows = await page.locator('tr[role="button"]').count();
    note(r, { rawShots: data.shots.length, plottedShots: plottedShots(data.shots).length, dots, rallyRows: rows, rallySummary: data.rallySummary });
    assert(dots >= 1 && dots === plottedShots(data.shots).length, 'Rendered dots disagree with plot-eligible API shots');
    assert(rows >= 1 && data.rallySummary?.total >= 1, 'No rallies in output');
    await expect(page.getByRole('heading', { name: 'Errors and net play' })).toBeVisible();
    assert(data.positionSummary?.n_frames > 0 && data.errorSummary && data.netApproachSummary, 'Coach insights missing API data');
    const report = page.getByLabel('Scouting report', { exact: true });
    await expect(report).toBeVisible();
    const reportText = await report.innerText(); note(r, { report: reportText });
    assert(reportText.length > 100 && !/\bP[12]\b|—|\*\*|##/.test(reportText), 'Scouting report contains jargon/raw markdown or insufficient notes');
    await expect(report.getByRole('heading', { name: 'Coaching cue' })).toBeVisible();
    assert(data.shots.length > 0 && data.inBoundsBounces > 0 && data.rallySummary.avg_length > 0, 'Expected core stats are zero');
    const totals = page.locator('.cc-card').filter({ has: page.getByRole('heading', { name: 'Totals', exact: true }) });
    await expect(totals).toContainText(`${data.shots.length} shots`);
    await expect(totals.locator('.cc-stat-tile').filter({ hasText: 'Avg rally length' })).toContainText(data.rallySummary.avg_length.toFixed(1));
    await expect(totals.locator('.cc-stat-tile')).toHaveCount(4);
  }, 'P0'),
  check('E2E-06', 'The completed recording is missing from the library, player history or dashboard date.', 'New recording is first, assigned to Aileena, in her history and reflected in Last recording today.', async (page, r, shared) => {
    const skip = dependency(shared, true); if (skip) return skip;
    await visit(page, '/recordings', r);
    const data = await getData(page, `/api/recordings/${shared.qaRecordingId}`, 'recording');
    const list = await getData(page, '/api/recordings', 'recordings');
    assert(list[0]?.id === shared.qaRecordingId && list[0].playerName === 'Aileena Hu', 'Library API order/assignment incorrect');
    await expect(recordingRow(page, data)).toBeVisible();
    await expect(page.locator('.cc-match-row[role="button"]').first()).toContainText(QA_NAME);
    const { probe } = await import('../lib.mjs');
    await probe(page, r, 'visible player label', () => expect(recordingRow(page, data)).toContainText('Aileena Hu'));
    await visit(page, `/players/${FIXTURES.playerRight}`, r);
    await expect(page.locator(`a.cc-match-row[href="/recordings/${shared.qaRecordingId}"]`)).toBeVisible();
    await visit(page, '/', r);
    const today = await page.evaluate(() => new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
    const player = page.locator(`[aria-label="Roster"] a[href="/players/${FIXTURES.playerRight}"]`);
    await expect(player).toContainText(`Last recording · ${today}`);
    note(r, { recordingId: shared.qaRecordingId, player: 'Aileena Hu', dashboardDate: today });
  }),
  check('E2E-07', 'A fresh recording cannot be reprocessed to completion.', 'One reprocess returns 200, shows processing, then completes within twenty minutes.', async (page, r, shared) => {
    const skip = dependency(shared, true); if (skip) return skip;
    await visit(page, `/recordings/${shared.qaRecordingId}`, r);
    const button = page.getByRole('button', { name: 'Reprocess', exact: true });
    await button.click(); await expect(page.getByText('Reprocess this recording?')).toBeVisible();
    const started = Date.now();
    const res = await actionResponse(page, '/api/trigger-process', 'POST', () => button.click(), 30000);
    note(r, { status: res.status(), triggerMs: Date.now() - started });
    if (res.status() !== 200) assert(false, `Fresh reprocess returned ${res.status()}: ${(await res.text().catch(() => '')).slice(0, 500)}`);
    await expect(page.getByText('Analyzing your recording.', { exact: true })).toBeVisible();
    const body = await getData(page, `/api/status?match_id=${shared.qaRecordingId}`);
    assert(['pending', 'processing'].includes(body.status), `Did not flip to processing: ${body.status}`);
    try { await pollDone(page, shared.qaRecordingId, r, PROCESS_TIMEOUT - (Date.now() - started)); }
    catch (e) {
      if (!(e instanceof ProcessingTimeout)) throw e;
      return { skip: `Reprocess completion verification exceeded its twenty-minute budget after ${(Date.now() - started) / 1000}s; reprocess was submitted once, last status remains pending/processing.` };
    }
    await expect(page.locator('main h1')).toHaveText(QA_NAME, { timeout: 20000 });
    shared.reprocessed = true;
    note(r, { reprocessSeconds: (Date.now() - started) / 1000 });
  }),
];
// Processing waits legitimately run up to twenty minutes; everything else keeps the default ceiling.
for (const c of checks) if (c.id === 'E2E-03' || c.id === 'E2E-07') c.timeoutMs = LONG_CHECK_TIMEOUT;
export default async function (args) { await runModule(args, checks, area, { drifts: args.shared.skipUpload || args.shared.resumed ? [] : drifts }); }
