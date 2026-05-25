import { chromium } from 'playwright';

const EMAIL = process.env.CC_AUDIT_EMAIL;
const PASSWORD = process.env.CC_AUDIT_PASSWORD;
const BASE = process.env.CC_AUDIT_BASE ?? 'http://localhost:3000';

if (!EMAIL || !PASSWORD) {
  console.error('CC_AUDIT_EMAIL and CC_AUDIT_PASSWORD must be set');
  process.exit(1);
}

const browser = await chromium.launch();
const ctx = await browser.newContext();
const page = await ctx.newPage();

await page.goto(`${BASE}/auth/login`, { waitUntil: 'networkidle' });

// Try common selectors — login forms vary.
const emailInput = page.locator('input[type="email"], input[name="email"], input[id*="email"]').first();
const passwordInput = page.locator('input[type="password"], input[name="password"]').first();
const submit = page.locator('button[type="submit"]').first();

await emailInput.fill(EMAIL);
await passwordInput.fill(PASSWORD);
await submit.click();

try {
  await page.waitForURL((url) => !url.pathname.startsWith('/auth'), { timeout: 15000 });
} catch {
  console.error('Sign-in did not redirect off /auth within 15s. Current URL:', page.url());
  console.error('Possible causes: wrong credentials, email verification required, captcha, or selector mismatch.');
  await page.screenshot({ path: 'scripts/audit/sign-in-failed.png', fullPage: true });
  console.error('Screenshot saved to scripts/audit/sign-in-failed.png');
  await browser.close();
  process.exit(2);
}

await ctx.storageState({ path: 'scripts/audit/storageState.json' });
await browser.close();

console.log('Saved storageState.json');
