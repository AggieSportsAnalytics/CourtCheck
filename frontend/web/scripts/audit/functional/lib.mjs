import { chromium } from 'playwright';
import { expect } from '@playwright/test';
import { mkdir, writeFile } from 'node:fs/promises';

export { expect };
export const BASE = process.env.CC_AUDIT_BASE ?? 'https://courtcheck-rho.vercel.app';
export const STORAGE = process.env.CC_AUDIT_STORAGE ?? 'scripts/audit/storageState.json'; // override for a re-domained copy (local builds)
export const SHOTS = 'scripts/audit/screenshots/functional';
export const TMP = '/private/tmp/courtcheck-audit';
export const QA_PREFIX = 'QA pilot audit';
export const QA_NAME = `${QA_PREFIX} 2026-09-09`;
export const VIEWPORTS = { desktop: [1440, 900], ipadL: [1024, 768], ipadP: [820, 1180], phone: [390, 844] };
export const FIXTURES = {
  recordingAssigned: 'eeaac0eb-d0ee-46aa-975d-3826b2caa7f0',
  recordingUnassigned: '7c98190b-4920-47a5-9e8b-ef5704a913a4',
  playerRight: '306e89a0-a75a-4965-abc7-5344fb42c297',
  playerLeft: '237515ce-c905-4754-99d7-9df8182c9ec9',
  clip: '../../data/StMarys_Court2_Aileena_vs_StMarys.mp4',
};
export const BOGUS = '00000000-0000-0000-0000-000000000000';
export const APP_PAGES = ['/', '/recordings', '/players', '/upload', '/profile', '/settings'];
export const REVIEW_PAGES = ['/', '/recordings', `/recordings/${FIXTURES.recordingAssigned}`, `/players/${FIXTURES.playerLeft}`, '/upload', '/settings'];
export const RECORDING_VIDEO = 'main .cc-video-hero video'; // Sidebar/wordmark also contain videos.
export const ROWS = '.cc-match-row[role="button"]';
export const DOTS = 'svg .shot-dot[tabindex="0"]';
export const PROCESS_TIMEOUT = 20 * 60 * 1000;
export const CHECK_TIMEOUT = 5 * 60 * 1000;           // default per-check ceiling
export const LONG_CHECK_TIMEOUT = PROCESS_TIMEOUT + 3 * 60 * 1000; // E2E processing waits
const PAGE_TIMEOUT = 60000;
const ACTION_TIMEOUT = 10000;
const pageInfo = new WeakMap();
const deletePermits = new WeakMap();

export class SessionDied extends Error {}
export class ProcessingTimeout extends Error {}
export function assert(condition, message) { if (!condition) throw new Error(message); }
export function note(r, value) {
  r.evidence += `${r.evidence ? '\n' : ''}${typeof value === 'string' ? value : JSON.stringify(value)}`;
}
export function cleanTitle(name) { return name.replace(/\.[a-z0-9]+$/i, '').replace(/_/g, ' ').trim(); }
export function check(id, title, expected, fn, severity = 'P1') { return { id, title, expected, fn, severity }; }

export function checkSession(page) {
  const info = pageInfo.get(page);
  if (!info?.authed) return;
  const path = page.url() === 'about:blank' ? '' : new URL(page.url()).pathname;
  if (info.died || /^\/(landing|auth)(\/|\.|$)/.test(path)) {
    throw new SessionDied(info.died || `Session expired: landed on ${path}`);
  }
}

function watchPage(page, authed) {
  const info = { authed, errors: [], failed: [], died: null };
  pageInfo.set(page, info);
  page.on('pageerror', (e) => info.errors.push(e.message));
  page.on('console', (m) => { if (m.type() === 'error') info.errors.push(m.text()); });
  page.on('framenavigated', (frame) => {
    if (!authed || frame !== page.mainFrame() || frame.url() === 'about:blank') return;
    const path = new URL(frame.url()).pathname;
    if (/^\/(landing|auth)(\/|\.|$)/.test(path)) info.died = `Session expired: landed on ${path}`;
  });
  page.on('response', (res) => {
    const url = new URL(res.url());
    if (res.status() >= 400 && !url.pathname.startsWith('/_vercel/')) {
      info.failed.push(`${res.status()} ${res.request().method()} ${url.origin}${url.pathname}`);
    }
    if (authed && url.origin === new URL(BASE).origin && url.pathname.startsWith('/api/') && res.status() === 401) {
      info.died = `Authenticated API returned 401: ${url.pathname}`;
    }
  });
  return info;
}

export async function openAuthed() {
  const browser = await chromium.launch();
  try {
    const ctx = await browser.newContext({ storageState: STORAGE, viewport: { width: 1440, height: 900 } });
    ctx.setDefaultTimeout(ACTION_TIMEOUT);
    await ctx.addInitScript(() => {
      try { sessionStorage.setItem('ccDashSplashSeen', '1'); }
      catch (e) { console.warn(`Audit splash flag unavailable: ${e.message}`); }
    });
    // Defense in depth: even a mistaken UI click cannot delete team data.
    await ctx.route('**/api/**', async (route) => {
      const req = route.request();
      const path = new URL(req.url()).pathname;
      const permitted = deletePermits.get(ctx);
      const forbiddenDelete = req.method() === 'DELETE' && (!permitted || path !== `/api/recordings/${permitted}`);
      if (forbiddenDelete || (req.method() === 'POST' && path === '/api/onboarding')) {
        console.error(`AUDIT SAFETY BLOCK: ${req.method()} ${path}`);
        await route.abort('blockedbyclient');
      } else await route.continue();
    });
    return { browser, ctx };
  } catch (e) { await browser.close(); throw e; }
}

export async function openAnon() {
  const browser = await chromium.launch();
  try {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    ctx.setDefaultTimeout(ACTION_TIMEOUT);
    return { browser, ctx };
  } catch (e) { await browser.close(); throw e; }
}
export async function saveState(ctx) { await ctx.storageState({ path: STORAGE }); }

export async function gotoAuthed(page, path, { expectAuthed = true } = {}) {
  const info = pageInfo.get(page) ?? watchPage(page, expectAuthed);
  info.errors.length = 0;
  info.failed.length = 0;
  const start = Date.now();
  await page.goto(new URL(path, BASE).href, { waitUntil: 'domcontentloaded', timeout: PAGE_TIMEOUT });
  let settle = 'networkidle';
  try { await page.waitForLoadState('networkidle', { timeout: 15000 }); }
  catch (e) {
    if (e.name !== 'TimeoutError') throw e;
    settle = 'networkidle timed out after 15000ms (page may be polling)';
  }
  checkSession(page);
  return { landed: new URL(page.url()).pathname, errors: info.errors, failed: info.failed, loadMs: Date.now() - start, settle };
}
export async function visit(page, path, r, authed = true) {
  const data = await gotoAuthed(page, path, { expectAuthed: authed });
  note(r, { url: page.url(), loadMs: data.loadMs, settle: data.settle });
  return data;
}

export async function shot(page, r, name = r.id) {
  await mkdir(SHOTS, { recursive: true });
  const width = page.viewportSize()?.width ?? 0;
  const file = `${SHOTS}/${name.replace(/[^a-zA-Z0-9_-]/g, '_')}-${width}.png`;
  await page.screenshot({ path: file, fullPage: true, timeout: ACTION_TIMEOUT });
  if (!r.screenshots.includes(file)) r.screenshots.push(file);
  return file;
}
async function failureShot(page, r, name) {
  if (!page || page.isClosed()) { note(r, 'Screenshot unavailable: page was not open.'); return; }
  try { await shot(page, r, name); }
  catch (e) { note(r, `Screenshot failed: ${e.message}`); }
}

export function createCollector() {
  const results = [];
  return {
    results,
    async run(id, area, title, fn, { severity = 'P1', page, expected = '', timeoutMs = CHECK_TIMEOUT } = {}) {
      const start = Date.now();
      const r = { id, area, title, status: 'pass', severity, evidence: '', expected, durationMs: 0, screenshots: [] };
      let fatal;
      try {
        if (page) checkSession(page);
        // Hard ceiling per check: run 1 hung for 50 minutes inside one step whose waits were all
        // individually bounded, so every check now races a timer and the suite moves on.
        let timer;
        const ceiling = new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(`Check exceeded its ${Math.round(timeoutMs / 1000)}s ceiling`)), timeoutMs); });
        const out = await Promise.race([fn(r), ceiling]).finally(() => clearTimeout(timer));
        if (page) checkSession(page);
        if (out?.skip) { r.status = 'skip'; note(r, out.skip); }
        assert(!r.problems?.length, r.problems?.join('\n'));
      } catch (e) {
        if (page) {
          try { checkSession(page); }
          catch (sessionError) { e = sessionError; }
        }
        r.status = 'fail';
        note(r, e.message ?? String(e));
        await failureShot(page, r, id);
        if (e instanceof SessionDied) { r.severity = 'P0'; fatal = e; }
      }
      delete r.problems;
      r.durationMs = Date.now() - start;
      results.push(r);
      console.log(`${r.status.toUpperCase().padEnd(4)} ${id} ${title} (${r.durationMs}ms)${r.status !== 'pass' ? ` :: ${r.evidence}` : ''}`);
      if (fatal) throw fatal;
      return r;
    },
  };
}

// Soft subchecks allow every viewport/route/validation case to run, with evidence for each.
export async function probe(page, r, label, fn) {
  try { await fn(); checkSession(page); }
  catch (e) {
    checkSession(page);
    if (e instanceof SessionDied) throw e;
    r.problems ??= [];
    r.problems.push(`${label}: ${e.message}`);
    note(r, `${label}: ${e.message}`);
    await failureShot(page, r, `${r.id}-${label}`);
  }
}

export async function runModule({ ctx, collector, shared }, checks, area, { authed = true, drifts = [] } = {}) {
  const page = await ctx.newPage();
  watchPage(page, authed);
  try {
    for (const drift of drifts) {
      await collector.run(drift.id, area, 'The plan selector or display contract differs from the component source.', async (r) => {
        note(r, drift.evidence);
        assert(false, drift.evidence);
      }, { severity: 'P3', page, expected: drift.expected });
    }
    for (const c of checks) {
      await collector.run(c.id, c.area ?? area, c.title, (r) => c.fn(page, r, shared), { severity: c.severity, expected: c.expected, page, timeoutMs: c.timeoutMs });
    }
  } finally { await page.close(); }
}

export async function overflowPx(page) {
  return page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
}

/** Actual painted text only: omit script/style, hidden controls and sr-only labels. */
export async function visibleText(page) {
  return page.evaluate(() => {
    const out = [];
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {
      const el = walker.currentNode.parentElement;
      const text = walker.currentNode.textContent?.trim();
      if (!el || !text || el.closest('script,style,noscript,[aria-hidden="true"],.sr-only')) continue;
      if (!el.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true })) continue;
      const range = document.createRange();
      range.selectNodeContents(walker.currentNode);
      if (![...range.getClientRects()].some((box) => box.width > 0 && box.height > 0)) continue;
      const style = getComputedStyle(el);
      out.push({ text, fontSize: parseFloat(style.fontSize), italic: style.fontStyle === 'italic', wordmark: Boolean(el.closest('.brand-mark,[aria-label="CourtCheck home"]')), element: el.tagName.toLowerCase() });
    }
    return out;
  });
}
export async function garbageText(page) {
  return (await visibleText(page)).filter(({ text }) => /\b(?:NaN|undefined|null|Infinity)\b|\[object Object\]|%%|—/.test(text)).map(({ text }) => text.slice(0, 120));
}

export async function apiJson(page, path, init = {}) {
  assert(init.method !== 'DELETE', 'Use deleteQaRecording; unrestricted deletion is forbidden.');
  assert(!(path === '/api/onboarding' && init.method === 'POST'), 'Onboarding POST is forbidden.');
  checkSession(page);
  const out = await page.evaluate(async ({ path, init }) => {
    const res = await fetch(path, { ...init, cache: 'no-store', signal: AbortSignal.timeout(30000) });
    const raw = await res.text();
    let body = null;
    let parseError = null;
    try { body = raw ? JSON.parse(raw) : null; }
    catch (e) { parseError = e.message; }
    return { status: res.status, body, parseError, contentType: res.headers.get('content-type'), landed: new URL(res.url).pathname };
  }, { path, init });
  if (pageInfo.get(page)?.authed && (out.status === 401 || /^\/(landing|auth)/.test(out.landed))) {
    throw new SessionDied(`Authenticated ${path} returned ${out.status}, landed on ${out.landed}`);
  }
  return out;
}
export function jsonInit(method, body) { return { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }; }
export async function getData(page, path, key) {
  const res = await apiJson(page, path);
  assert(res.status === 200 && !res.parseError, `GET ${path}: ${res.status}, ${res.parseError ?? JSON.stringify(res.body?.error)}`);
  const data = key ? res.body?.[key] : res.body;
  assert(data != null, `GET ${path}: missing ${key ?? 'body'}`);
  return data;
}
export async function patch(page, path, body) {
  const res = await apiJson(page, path, jsonInit('PATCH', body));
  assert(res.status === 200, `PATCH ${path}: ${res.status} ${JSON.stringify(res.body)}`);
}
export async function actionResponse(page, path, method, action, timeout = ACTION_TIMEOUT) {
  const pending = page.waitForResponse((res) => new URL(res.url()).pathname === path && res.request().method() === method, { timeout });
  // Attach a rejection handler immediately, even if the action itself fails.
  const outcome = pending.then((response) => ({ response }), (error) => ({ error }));
  await action();
  const result = await outcome;
  checkSession(page);
  if (result.error) throw result.error;
  return result.response;
}
export async function navigateClick(page, locator, path) {
  await locator.click();
  await expect(page).toHaveURL((url) => url.pathname === path, { timeout: ACTION_TIMEOUT });
  checkSession(page);
  await expect(page.locator('main')).toBeVisible();
  assert((await page.locator('main').innerText()).trim().length > 0, `${path}: blank main`);
}
export function recordingRow(page, recording) {
  return page.locator(ROWS).filter({ has: page.getByText(cleanTitle(recording.name), { exact: true }) });
}
export async function remember(shared, page, path, key, fields) {
  const data = await getData(page, path, key);
  shared.originals ??= [];
  const values = Object.fromEntries(fields.map((field) => [field, data[field]]));
  shared.originals.push({ path, key, values: structuredClone(values) });
  return values;
}
export async function restore(page, path, key, original) {
  const current = await getData(page, path, key);
  const changed = Object.fromEntries(Object.entries(original).filter(([k, v]) => JSON.stringify(v) !== JSON.stringify(current[k])));
  if (Object.keys(changed).length) await patch(page, path, changed);
}

export async function viewportChecks(page, r, paths, { widths = Object.values(VIEWPORTS), authed = true, extra } = {}) {
  for (const path of paths) {
    for (const [width, height] of widths) {
      await probe(page, r, `${path}-${width}`, async () => {
        await page.setViewportSize({ width, height });
        const data = await visit(page, path, r, authed);
        await shot(page, r, `${r.id}-${paths.length > 1 ? path : ''}`);
        const overflow = await overflowPx(page);
        note(r, { path, width, overflow, errors: data.errors });
        assert(overflow === 0, `Horizontal overflow ${overflow}px`);
        assert(data.errors.length === 0, `Console/page errors: ${data.errors.join('; ')}`);
        if (extra) await extra(page, width);
      });
    }
  }
  await page.setViewportSize({ width: 1440, height: 900 });
}

export async function videoReady(page) {
  await expect(page.locator(RECORDING_VIDEO)).toHaveCount(1);
  await expect.poll(() => page.locator(RECORDING_VIDEO).evaluate((v) => v.readyState >= 1 && Number.isFinite(v.duration) && v.duration > 0), { timeout: 20000 }).toBe(true);
}
export async function seekControl(page, fraction) {
  const slider = page.getByLabel('Seek', { exact: true });
  await slider.scrollIntoViewIfNeeded();
  const box = await slider.boundingBox();
  assert(box && box.width > 0, 'Seek slider has no clickable bounds');
  // range inputs cannot be filled with Playwright fill(); use a real pointer click.
  await slider.click({ position: { x: box.width * fraction, y: box.height / 2 } });
  const target = Number(await slider.inputValue());
  assert(target > 0 && Number.isFinite(target), `Seek slider value did not change: ${target}`);
  await assertSeek(page, target);
  return target;
}
export async function seekAway(page, target) {
  await videoReady(page);
  await page.locator(RECORDING_VIDEO).evaluate((v, t) => { v.pause(); v.currentTime = t < v.duration / 2 ? v.duration * 0.8 : 0; }, target);
}
export async function assertSeek(page, target) {
  await expect.poll(() => page.locator(RECORDING_VIDEO).evaluate((v) => v.currentTime), { timeout: 10000 }).toBeGreaterThanOrEqual(Math.max(0, target - 0.3));
  const actual = await page.locator(RECORDING_VIDEO).evaluate((v) => { v.pause(); return v.currentTime; });
  assert(Math.abs(actual - target) < 2, `Seek expected ${target}s, observed ${actual}s`);
}

export async function pollDone(page, id, r, timeout = PROCESS_TIMEOUT, onPoll) {
  const start = Date.now();
  let lastLog = 0;
  while (Date.now() - start < timeout) {
    const body = await getData(page, `/api/status?match_id=${id}`);
    if (onPoll) onPoll(body);
    assert(['pending', 'processing', 'done', 'failed'].includes(body.status), `Invalid status: ${JSON.stringify(body)}`);
    if (body.status === 'failed') throw new Error(`Pipeline failed: ${body.error ?? 'no error provided'}`);
    if (body.status === 'done') { note(r, { id, processingSeconds: (Date.now() - start) / 1000 }); return body; }
    if (Date.now() - lastLog > 30000) {
      console.log(`${r.id}: ${(Date.now() - start) / 1000}s, ${body.status}, ${body.stage}, progress=${body.progress}`);
      await saveState(page.context());
      lastLog = Date.now();
    }
    await page.waitForTimeout(5000);
  }
  throw new ProcessingTimeout(`Pipeline did not finish after ${(Date.now() - start) / 1000}s`);
}

/** Never delete an existing row, even if its title happens to use the QA prefix. */
export async function deleteQaRecording(page, shared, r) {
  const id = shared.qaRecordingId;
  assert(id && shared.createdRecordingIds?.includes(id), 'Delete refused: ID was not created by this audit run.');
  const recording = await getData(page, `/api/recordings/${id}`, 'recording');
  assert(typeof recording.name === 'string' && recording.name.startsWith(QA_PREFIX), `Delete refused: recording title must start with "${QA_PREFIX}".`);
  await visit(page, '/recordings', r);
  const row = recordingRow(page, recording);
  await expect(row).toHaveCount(1);
  const title = await row.locator('span.font-display.truncate').innerText();
  assert(title.startsWith(QA_PREFIX), `Delete refused: row title must start with "${QA_PREFIX}".`);
  await row.getByLabel('Delete recording', { exact: true }).click();
  // Drift: confirmation replaces .cc-match-row and omits the title; scope through Cancel.
  const confirmation = page.getByRole('button', { name: 'Cancel', exact: true }).locator('..').locator('..');
  await expect(confirmation).toContainText('Delete this recording?');
  const fresh = await getData(page, `/api/recordings/${id}`, 'recording');
  assert(typeof fresh.name === 'string' && fresh.name.startsWith(QA_PREFIX), 'Delete refused: title changed before confirmation.');
  deletePermits.set(page.context(), id);
  try {
    const res = await actionResponse(page, `/api/recordings/${id}`, 'DELETE', () => confirmation.getByRole('button', { name: 'Delete', exact: true }).click());
    note(r, `DELETE /api/recordings/${id}: ${res.status()}`);
    assert(res.status() === 204, `Expected DELETE 204, got ${res.status()}`);
  } finally { deletePermits.delete(page.context()); }
  await expect(recordingRow(page, recording)).toHaveCount(0);
  const res = await apiJson(page, `/api/recordings/${id}`);
  assert(res.status === 404, `Deleted recording GET returned ${res.status}`);
}

export async function writeReport(results, meta) {
  const dir = 'scripts/audit/functional';
  await mkdir(dir, { recursive: true });
  await writeFile(`${dir}/report.json`, JSON.stringify({ ...meta, results }, null, 2) + '\n');
  const flat = (s) => String(s).replace(/\u001b\[[0-9;]*m/g, '').replace(/\s+/g, ' ').trim();
  const failures = results.filter((r) => r.status === 'fail').sort((a, b) => a.severity.localeCompare(b.severity));
  const lines = ['# CourtCheck functional audit', '', `Base: ${meta.base}  Started: ${meta.startedAt}  Finished: ${meta.finishedAt}`, '', '## Findings', ''];
  for (const f of failures) lines.push(`- [${f.severity}] [area: ${f.area}] ${flat(f.title)}`, `  Evidence: ${flat(f.evidence)}`, `  Expected: ${flat(f.expected)}`, `  Repro: check ${f.id}${f.screenshots.length ? ` (${f.screenshots.join(', ')})` : ''}`);
  if (!failures.length) lines.push(results.length ? 'No failures in the executed checks.' : 'Offline smoke only: no functional checks executed; production was not contacted.');
  lines.push('', '## Checks run', '', '| check id | result | duration | evidence / skip reason |', '|---|---|---|---|');
  for (const r of results) lines.push(`| ${r.id} | ${r.status === 'skip' ? 'skipped' : r.status} | ${r.durationMs}ms | ${flat(r.evidence).replace(/\|/g, '&#124;')} |`);
  await writeFile(`${dir}/REPORT.md`, lines.join('\n') + '\n');
}
