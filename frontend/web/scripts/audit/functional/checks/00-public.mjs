import { open, mkdir } from 'node:fs/promises';
import { BASE, TMP, FIXTURES, VIEWPORTS, check, assert, expect, note, probe, visit, viewportChecks, openAnon, runModule } from '../lib.mjs';

export const area = 'auth';
const anonVisit = (page, path, r) => visit(page, path, r, false);
const authPages = ['/auth/login', '/auth/signup', '/auth/forgot-password', '/auth/update-password'];
const landingPath = (url) => ['/landing', '/landing.html'].includes(new URL(url).pathname);
const drifts = [{ id: 'DRIFT-PUB', evidence: 'app/auth/login/page.tsx: signup link reads "Create an account."; app/auth/signup/page.tsx: only one password input, no confirmation. Query-string error is not read by LoginPage.', expected: 'The plan names a Sign up link, password-confirmation input and visible callback errors.' }];

async function inlineError(page) {
  await expect(page.locator('[role="alert"]').filter({ visible: true }).first()).toBeVisible({ timeout: 10000 });
  assert((await page.locator('[role="alert"]').filter({ visible: true }).first().innerText()).trim().length > 0, 'Inline error is empty');
}
async function callbackError(page, r, path, expected) {
  await anonVisit(page, path, r);
  const url = new URL(page.url());
  const error = url.searchParams.get('error');
  note(r, { landed: url.pathname, error });
  assert(url.pathname === '/auth/login' && error, 'Missing login redirect or error parameter');
  if (expected) assert(error === expected, `Expected ${expected}; observed ${error}`);
  await expect(page.getByText(error, { exact: true })).toBeVisible();
}

export const checks = [
  check('PUB-01', 'Protected pages do not consistently redirect anonymous coaches to landing.', 'All seven protected URLs end on /landing or /landing.html.', async (page, r) => {
    for (const path of ['/', '/recordings', '/players', '/upload', '/profile', '/settings', `/recordings/${FIXTURES.recordingAssigned}`]) {
      await probe(page, r, path, async () => { await anonVisit(page, path, r); assert(landingPath(page.url()), `${path} landed on ${page.url()}`); });
    }
  }, 'P0'),
  check('PUB-02', 'Anonymous API requests do not return the Unauthorized JSON contract.', 'Every protected API returns 401 application/json with error Unauthorized, without following redirects.', async (page, r) => {
    for (const path of ['/api/recordings', '/api/players', '/api/dashboard/summary', '/api/status?match_id=x', '/api/heatmaps/latest', '/api/onboarding']) {
      await probe(page, r, path, async () => {
        const res = await page.request.get(`${BASE}${path}`, { maxRedirects: 0 });
        const text = await res.text();
        note(r, { path, status: res.status(), contentType: res.headers()['content-type'], body: text.slice(0, 200) });
        assert(res.status() === 401 && /application\/json/.test(res.headers()['content-type']), 'Expected 401 JSON');
        assert(JSON.parse(text).error === 'Unauthorized', 'Expected error Unauthorized');
      });
    }
  }),
  check('PUB-03', 'The landing page has missing content, broken video assets or viewport errors.', 'Title, login/signup links, hero asset 200 and zero errors/overflow at all four widths.', async (page, r) => {
    await anonVisit(page, '/landing.html', r);
    assert((await page.title()).trim(), 'Empty title');
    await expect(page.locator('a[href="/auth/login"]').first()).toBeVisible();
    await expect(page.locator('a[href="/auth/signup"]').first()).toBeVisible();
    const assets = await page.locator('video[src], video source[src]').evaluateAll((els) => els.map((el) => el.getAttribute('src')).filter((s) => /Bounce_Animated\.(webm|mp4)/.test(s)));
    assert(assets.length > 0, 'No Bounce_Animated hero video source');
    for (const asset of new Set(assets)) {
      const res = await page.request.get(new URL(asset, BASE).href);
      note(r, `${asset}: ${res.status()}`);
      assert(res.status() === 200, `Hero asset returned ${res.status()}`);
    }
    await viewportChecks(page, r, ['/landing.html'], { authed: false });
  }),
  check('PUB-04', 'Legacy routes do not permanently redirect to the dashboard.', '308 Location / for each legacy URL.', async (page, r) => {
    for (const path of ['/overall-stats', '/opponents', '/match-stats/x']) {
      await probe(page, r, path, async () => {
        const res = await page.request.get(`${BASE}${path}`, { maxRedirects: 0 });
        const location = res.headers().location;
        note(r, { path, status: res.status(), location });
        assert(res.status() === 308 && location && new URL(location, BASE).pathname === '/', 'Expected 308 to /');
      });
    }
  }),
  check('PUB-05', 'An unknown route exposes a raw application error.', 'A friendly not-found page or the anonymous landing redirect.', async (page, r) => {
    const data = await anonVisit(page, '/this-does-not-exist', r);
    const text = await page.locator('body').innerText();
    assert(!/Application error|Unhandled Runtime Error|NEXT_NOT_FOUND|stack trace/i.test(text), text.slice(0, 400));
    assert(landingPath(page.url()) || /not found|404|doesn.t exist/i.test(text), `Unrecognized unknown-route state: ${data.landed}`);
    note(r, { landingRedirect: landingPath(page.url()), text: text.slice(0, 400) });
  }),
  check('PUB-06', 'Public pages are missing required security headers.', 'HSTS, DENY frame policy, nosniff and CSP on landing and login.', async (page, r) => {
    for (const path of ['/landing.html', '/auth/login']) {
      await probe(page, r, path, async () => {
        const res = await page.request.get(`${BASE}${path}`);
        const h = res.headers();
        const observed = Object.fromEntries(['strict-transport-security', 'x-frame-options', 'x-content-type-options', 'content-security-policy'].map((k) => [k, h[k] ?? null]));
        note(r, { path, headers: observed });
        assert(h['strict-transport-security'] && h['x-frame-options']?.toUpperCase() === 'DENY' && h['x-content-type-options'] === 'nosniff' && h['content-security-policy'], 'Missing/incorrect security headers');
      });
    }
  }),
  check('PUB-07', 'Auth pages overflow, log errors or use zoom-triggering input text.', 'Four auth pages render at all four widths with input font sizes at least 16px.', async (page, r) => {
    await viewportChecks(page, r, authPages, { authed: false, extra: async (p) => {
      await expect(p.locator('form')).toBeVisible();
      const sizes = await p.locator('input').evaluateAll((els) => els.filter((el) => el.checkVisibility()).map((el) => ({ id: el.id, size: parseFloat(getComputedStyle(el).fontSize) })));
      assert(sizes.length > 0 && sizes.every((x) => x.size >= 16), JSON.stringify(sizes));
    } });
  }),
  check('PUB-08', 'Login validation or recovery links fail.', 'Empty and invalid credentials show inline errors; recovery/signup links navigate correctly.', async (page, r) => {
    await probe(page, r, 'empty login', async () => {
      await anonVisit(page, '/auth/login', r);
      let dialogs = 0;
      const listener = async (dialog) => { dialogs++; await dialog.dismiss(); };
      page.on('dialog', listener);
      try { await page.getByRole('button', { name: 'Sign in', exact: true }).click(); await inlineError(page); assert(dialogs === 0, 'Browser dialog appeared'); }
      finally { page.off('dialog', listener); }
    });
    await probe(page, r, 'wrong password', async () => {
      await anonVisit(page, '/auth/login', r);
      await page.locator('#email').fill('coach@school.edu');
      await page.locator('#password').fill('wrongpassword');
      await page.getByRole('button', { name: 'Sign in', exact: true }).click();
      await inlineError(page);
      note(r, await page.getByRole('alert').innerText());
      assert(new URL(page.url()).pathname === '/auth/login', 'Wrong-password submission left login');
    });
    for (const path of ['/auth/forgot-password', '/auth/signup']) {
      await probe(page, r, path, async () => {
        await anonVisit(page, '/auth/login', r);
        await page.locator(`a[href="${path}"]`).click();
        await expect(page).toHaveURL((url) => url.pathname === path);
      });
    }
  }, 'P0'),
  check('PUB-09', 'Signup client validation or keyboard role selection is incomplete.', 'Short and mismatched passwords show inline errors without any signup request; role changes with keyboard.', async (page, r) => {
    let signupRequests = 0;
    const blockSignup = async (route) => { signupRequests++; await route.abort('blockedbyclient'); };
    await page.route('**/auth/v1/signup**', blockSignup);
    try {
      await anonVisit(page, '/auth/signup', r);
      await probe(page, r, 'short password', async () => {
        await page.locator('#name').fill('QA Pilot');
        await page.locator('#email').fill('coach@school.edu');
        await page.locator('#password').fill('abc');
        await page.getByRole('button', { name: 'Sign up', exact: true }).click();
        await expect(page.getByText('At least 10 characters.', { exact: true })).toBeVisible();
      });
      await probe(page, r, 'keyboard role', async () => {
        const player = page.getByRole('radiogroup', { name: 'Role' }).getByRole('radio', { name: 'Player', exact: true });
        await player.focus(); await page.keyboard.press('Space');
        await expect(player).toHaveAttribute('aria-checked', 'true');
      });
      await probe(page, r, 'mismatched passwords', async () => {
        const passwords = page.locator('input[type="password"]');
        assert(await passwords.count() === 2, 'Signup has no confirmation input; mismatch validation cannot be exercised. No valid signup submitted.');
        // The mismatched form is invalid; the signup route is also blocked above.
        await passwords.nth(0).fill('audit-password-A'); await passwords.nth(1).fill('audit-password-B');
        await page.getByRole('button', { name: 'Sign up', exact: true }).click();
        await expect(page.getByText(/passwords?.*(match|same)/i)).toBeVisible();
      });
      assert(signupRequests === 0, `${signupRequests} unexpected signup requests were blocked`);
    } finally { await page.unroute('**/auth/v1/signup**', blockSignup); }
  }),
  check('PUB-10', 'Password reset does not show success or inline email validation.', 'One reset email is requested and success is visible; invalid email produces an inline error.', async (page, r) => {
    let emailSkipped = false;
    await probe(page, r, 'one reset email', async () => {
      await anonVisit(page, '/auth/forgot-password', r);
      await mkdir(TMP, { recursive: true });
      let marker;
      try { marker = await open(`${TMP}/forgot-password-sent`, 'wx'); }
      catch (e) {
        if (e.code !== 'EEXIST') throw e;
        emailSkipped = true;
        note(r, 'Reset email subcheck skipped: one attempt already recorded in /private/tmp/courtcheck-audit/forgot-password-sent.');
        return;
      }
      await marker.close();
      await page.locator('#email').fill('brile761@gmail.com');
      await page.getByRole('button', { name: 'Send reset link', exact: true }).click();
      await expect(page.getByRole('heading', { name: 'Check your email.' })).toBeVisible({ timeout: 15000 });
      note(r, 'One reset email requested for brile761@gmail.com; success state visible.');
    });
    await probe(page, r, 'invalid email', async () => {
      await anonVisit(page, '/auth/forgot-password', r);
      await page.locator('#email').fill('not-an-email');
      await page.getByRole('button', { name: 'Send reset link', exact: true }).click();
      await inlineError(page);
    });
    if (emailSkipped) return { skip: 'Email-success subcheck already attempted (persistent one-email guard); invalid-email subcheck executed above.' };
  }),
  check('PUB-11', 'Missing confirmation tokens leave the coach without a visible error.', 'Redirect to login with Missing token displayed.', (page, r) => callbackError(page, r, '/auth/confirm', 'Missing token')),
  check('PUB-12', 'Missing reset tokens leave the coach without a visible error.', 'Redirect to login with the invalid/expired-link error displayed.', (page, r) => callbackError(page, r, '/auth/reset-password', 'Invalid or expired reset link')),
];

export default async function ({ collector, shared }) {
  const { browser, ctx } = await openAnon();
  try { await runModule({ ctx, collector, shared }, checks, area, { authed: false, drifts }); }
  finally { await browser.close(); }
}
