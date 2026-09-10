import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { mkdir } from 'node:fs/promises';
import { TMP, FIXTURES, BOGUS, check, assert, expect, note, probe, visit, getData, apiJson, jsonInit, remember, restore, viewportChecks, runModule } from '../lib.mjs';

export const area = 'upload';
const fileInput = 'input[aria-label="Recording file input"]';
const drifts = [{ id: 'DRIFT-UPL', evidence: 'app/upload/page.tsx: [data-state="idle"] stages a file; Upload and process calls the hook duration probe. Dropzone has no rejection callback and the idle pane does not render hook error. Player selector id=player includes year suffixes.', expected: 'Plan implies duration validation on selection and visible type/duration error text.' }];
async function noUploadRequest(page, r, action) {
  const requests = [];
  const listener = (req) => { if (new URL(req.url()).pathname === '/api/create-upload') requests.push(req.method()); };
  const block = (route) => route.abort('blockedbyclient');
  // A failed negative test must not create an unlabelled production recording.
  await page.route('**/api/create-upload', block);
  page.on('request', listener);
  try { await action(); await page.waitForTimeout(1500); }
  finally {
    page.off('request', listener); await page.unroute('**/api/create-upload', block);
    note(r, { createUploadRequests: requests });
    assert(requests.length === 0, 'Validation unexpectedly sent /api/create-upload');
  }
}
async function validationCases(page, r) {
  const rec = `/api/recordings/${FIXTURES.recordingAssigned}`;
  const player = `/api/players/${FIXTURES.playerRight}`;
  const cases = [
    ['/api/create-upload', 'POST', { filename: 'x.txt' }, 400, /Invalid file type/i],
    ['/api/trigger-process', 'POST', {}, 400, /Missing match_id/],
    ['/api/trigger-process', 'POST', { match_id: BOGUS }, 403],
    ['/api/status', 'GET', null, 400, /missing match_id/],
    [`/api/status?match_id=${BOGUS}`, 'GET', null, 404],
    [rec, 'PATCH', { name: '' }, 400],
    [rec, 'PATCH', { name: 'a'.repeat(101) }, 400],
    [rec, 'PATCH', {}, 400, /Nothing to update/],
    [rec, 'PATCH', { notes: 'bad' }, 400],
    [player, 'PATCH', { name: '' }, 400],
    [player, 'PATCH', { photo_url: 'http://x' }, 400],
    ['/api/proxy-image?url=http://example.com/a.png', 'GET', null, 403],
    ['/api/proxy-image?url=https://evil.example/a.png', 'GET', null, 403],
    [`/api/recordings/${BOGUS}`, 'GET', null, 404],
  ];
  for (const [path, method, body, status, error] of cases) await probe(page, r, `${method}-${path}-${JSON.stringify(body)}`, async () => {
    const res = await apiJson(page, path, method === 'GET' ? {} : jsonInit(method, body));
    note(r, { path, method, request: body, status: res.status, body: res.body, parseError: res.parseError });
    assert(res.status === status, `Expected ${status}, observed ${res.status}`);
    assert(!res.parseError && typeof res.body?.error === 'string', 'Expected JSON error response');
    if (error) assert(error.test(res.body.error), `Unexpected error: ${res.body.error}`);
  });
}

export const checks = [
  check('UPL-01', 'Upload form is missing player choices, metadata or size guidance.', 'Dropzone, name input, Unknown / not assigned plus nine API players, MP4/MOV/AVI max 500 MB.', async (page, r) => {
    await visit(page, '/upload', r);
    await expect(page.locator('[data-state="idle"]')).toBeVisible();
    await expect(page.locator(fileInput)).toHaveCount(1);
    await expect(page.getByPlaceholder('e.g. Lin vs Stanford · Set 1')).toBeVisible();
    await expect(page.getByText('MP4 · MOV · AVI · max 500 MB', { exact: true })).toBeVisible();
    const players = await getData(page, '/api/players', 'players');
    await expect(page.locator('select#player option')).toHaveCount(10);
    assert(players.length === 9, `Expected nine players, got ${players.length}`);
    const options = await page.locator('select#player option').evaluateAll((els) => els.map((el) => ({ value: el.value, text: el.textContent })));
    note(r, { options });
    assert(options[0].value === '' && options[0].text === 'Unknown / not assigned', 'Unassigned choice missing');
    for (const player of players) assert(options.some((option) => option.value === player.id && option.text.includes(player.name)), `Missing ${player.name}`);
  }),
  check('UPL-02', 'Selecting a non-video file gives no rejection message or starts an upload.', 'Small .txt rejected visibly and no create-upload request.', async (page, r) => {
    await visit(page, '/upload', r);
    await noUploadRequest(page, r, async () => {
      await page.locator(fileInput).setInputFiles({ name: 'audit-invalid.txt', mimeType: 'text/plain', buffer: Buffer.from('not a video') });
      await expect(page.locator('main').getByText(/not a video we can read|MP4, MOV, or AVI/i)).toBeVisible();
    });
  }),
  check('UPL-03', 'A recording over fifteen minutes is not visibly rejected before upload.', '16-minute generated video returns idle with please trim or split and no create-upload.', async (page, r) => {
    await mkdir(TMP, { recursive: true });
    const long = `${TMP}/long.mp4`;
    await promisify(execFile)('ffmpeg', ['-f', 'lavfi', '-i', 'color=c=black:s=64x64:r=1:d=960', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-y', long], { timeout: 60000 });
    await visit(page, '/upload', r);
    await noUploadRequest(page, r, async () => {
      await page.locator(fileInput).setInputFiles(long);
      await page.getByRole('button', { name: 'Upload and process' }).click();
      await expect(page.getByText(/15 minutes or shorter/i)).toBeVisible({ timeout: 20000 });
      await expect(page.locator('[data-state="idle"]')).toBeVisible();
      note(r, await page.locator('main').innerText());
    });
  }),
  check('UPL-04', 'API boundary validation accepts invalid input or returns the wrong error.', 'Fourteen malformed/unauthorized API calls return their specified JSON 400/403/404 errors.', async (page, r, shared) => {
    await visit(page, '/upload', r);
    const recPath = `/api/recordings/${FIXTURES.recordingAssigned}`;
    const playerPath = `/api/players/${FIXTURES.playerRight}`;
    const recOriginal = await remember(shared, page, recPath, 'recording', ['name', 'notes']);
    const playerOriginal = await remember(shared, page, playerPath, 'player', ['name', 'photo_url']);
    try { await validationCases(page, r); }
    finally {
      await probe(page, r, 'restore recording after validation', () => restore(page, recPath, 'recording', recOriginal));
      await probe(page, r, 'restore player after validation', () => restore(page, playerPath, 'player', playerOriginal));
    }
  }),
  check('UPL-05', 'Upload controls are clipped on phone or portrait iPad.', 'Dropzone and staged Upload and process control are reachable without horizontal overflow at 390/820.', (page, r) => viewportChecks(page, r, ['/upload'], { widths: [[390, 844], [820, 1180]], extra: async (p) => {
    await expect(p.locator('[data-state="idle"]')).toBeVisible();
    await p.locator(fileInput).setInputFiles(FIXTURES.clip);
    const button = p.getByRole('button', { name: 'Upload and process' });
    await expect(button).toBeVisible(); await button.click({ trial: true });
    const { overflowPx, shot } = await import('../lib.mjs');
    assert(await overflowPx(p) === 0, 'Staged upload state overflows');
    await shot(p, r, `${r.id}-staged`);
  } })),
];
export default async function (args) { await runModule(args, checks, area, { drifts }); }
