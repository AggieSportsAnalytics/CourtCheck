import { chromium } from 'playwright';

const BASE = process.env.CC_AUDIT_BASE ?? 'http://localhost:3000';

// Interactive sign-in: launches a visible Chromium window. Brian signs in
// manually. Script waits for any navigation off /auth and then persists the
// storage state.
const browser = await chromium.launch({ headless: false });
const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 } });
const page = await ctx.newPage();

await page.goto(`${BASE}/auth/login`);

console.log('Sign in inside the open Chromium window. Storage state will save automatically after redirect.');

await page.waitForURL((url) => !url.pathname.startsWith('/auth'), { timeout: 5 * 60 * 1000 });

await ctx.storageState({ path: 'scripts/audit/storageState.json' });
console.log('Saved scripts/audit/storageState.json. Closing browser.');

await browser.close();
