import { REVIEW_PAGES, check, assert, note, probe, visit, visibleText, openAnon, runModule } from '../lib.mjs';

export const area = 'copy';
// The plan supplies no COPY ids. These seven stable ids cover every requested rule.
const rules = [
  ['COPY-01', 'UI copy contains em dashes.', 'No em dashes in visible copy.', ({ text }) => text.includes('—')],
  ['COPY-02', 'UI uses noncanonical words for recordings.', 'Use recording, with actual match-play terminology and source filenames distinguished in evidence.', ({ text }) => {
    // Whitelist only source filenames, not an entire ancestor containing one.
    if (/St\s*Mary|StMarys/i.test(text) && (/\.(mp4|mov|avi)\b/i.test(text) || /court\s*2/i.test(text))) return false;
    if (/^\/api\//.test(text)) return false;
    if (/\b(?:clip(?:s)?|film(?:s)?)\b/i.test(text)) return true;
    // Authentication sessions and the verb "match" are not names for a recording.
    const rest = text.replace(/(?:end|expired?|signed? out|revok\w*|auth\w*|device)[^.]*session(?:s)?[^.]*|session(?:s)?[^.]*(?:device|expired?|signed? out|revok\w*|auth\w*)/gi, '')
      .replace(/\bmatch(?:es)? (?:your|those|the) filters\b|\bmatch (?:point|score|result|play)\b/gi, '');
    return /\b(?:match(?:es)?|session(?:s)?)\b/i.test(rest);
  }],
  ['COPY-03', 'UI exposes tracker player abbreviations.', 'You / Opponent, never P1/P2/Opp.', ({ text }) => /\b(?:P1|P2|Opp)\b/.test(text)],
  ['COPY-04', 'UI uses the obsolete serve label.', 'Serve/Overhead, never Serve / Smash.', ({ text }) => /Serve\s*\/\s*Smash/i.test(text)],
  ['COPY-05', 'UI uses British analysis spelling.', 'American spelling: Analyzing.', ({ text }) => /\bAnalysing\b/i.test(text)],
  ['COPY-06', 'UI uses italics outside the wordmark.', 'Only the wordmark may be italic.', ({ italic, wordmark }) => italic && !wordmark],
  ['COPY-07', 'Visible text is smaller than thirteen pixels.', 'Painted UI text has computed font-size at least 13px.', ({ fontSize }) => fontSize < 13],
];

async function scan(page, r, predicate, shared) {
  for (const path of REVIEW_PAGES) await probe(page, r, path, async () => {
    // Cache observations, not conclusions, so all seven rules assess the same render.
    if (!shared.copyPages?.[path]) {
      await visit(page, path, r);
      shared.copyPages ??= {};
      shared.copyPages[path] = await visibleText(page);
    } else await visit(page, path, r);
    const bad = shared.copyPages[path].filter(predicate);
    note(r, { path, snippets: bad });
    assert(bad.length === 0, `${path}: ${bad.length} copy/style violations: ${bad.map((x) => x.text).join(' | ')}`);
  });
  // Never open public auth/landing pages using the authenticated session.
  const { browser, ctx } = await openAnon();
  const anon = await ctx.newPage();
  try {
    await probe(anon, r, 'landing', async () => {
      await visit(anon, '/landing.html', r, false);
      const bad = (await visibleText(anon)).filter(predicate);
      note(r, { path: '/landing.html', snippets: bad });
      assert(!bad.length, `/landing.html: ${bad.length} violations: ${bad.map((x) => x.text).join(' | ')}`);
    });
  } finally { await anon.close(); await browser.close(); }
}

export const checks = rules.map(([id, title, expected, predicate]) => check(id, title, expected, (page, r, shared) => scan(page, r, predicate, shared), 'P2'));
export default async function (args) { await runModule(args, checks, area); }
