// From frontend/web: node scripts/audit/functional/run.mjs [--only 03,04] [--skip-upload]
import { mkdir, readdir } from 'node:fs/promises';
import { BASE, TMP, createCollector, openAuthed, saveState, writeReport, SessionDied } from './lib.mjs';

function parseArgs(args) {
  let only = null;
  let skipUpload = false;
  let qaRecording = null;
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--skip-upload') skipUpload = true;
    else if (args[i] === '--only' && args[i + 1] && !args[i + 1].startsWith('--')) only = args[++i].split(',');
    // Resume against a QA recording that an earlier run already uploaded and processed to done.
    else if (args[i] === '--qa-recording' && /^[0-9a-f-]{36}$/i.test(args[i + 1] ?? '')) qaRecording = args[++i];
    else throw new Error(`Unknown or incomplete argument: ${args[i]}`);
  }
  return { only, skipUpload, qaRecording };
}
const { only, skipUpload, qaRecording } = parseArgs(process.argv.slice(2));
const startedAt = new Date().toISOString();
const collector = createCollector();
const shared = { qaRecordingId: null, qaPlayerId: null, createdRecordingIds: [], originals: [], skipUpload };
if (qaRecording) {
  Object.assign(shared, { qaRecordingId: qaRecording, createdRecordingIds: [qaRecording], triggered: true, e2eDone: true, resumed: true, uploadStarted: Date.now(), progressSamples: [] });
  console.log(`Resuming with QA recording ${qaRecording}: E2E-01..04 are skipped, E2E-05..07 and cleanup run against it.`);
}
const files = (await readdir(new URL('./checks/', import.meta.url))).filter((f) => /^\d\d-.*\.mjs$/.test(f)).sort();
const selected = files.filter((f) => !only || only.some((p) => f.startsWith(p)));
const modules = [];
let browser;
let ctx;
let abortReason;

function harnessFailure(error) {
  const message = error.message ?? String(error);
  abortReason = message;
  console.error(error instanceof SessionDied
    ? `ABORT: ${message}. Brian must run node scripts/audit/sign-in.mjs.`
    : `ABORT (harness error): ${message}`);
  collector.results.push({ id: `harness-${collector.results.filter((r) => r.id.startsWith('harness')).length + 1}`, area: 'auth', title: 'The audit could not complete or persist its session.', status: 'fail', severity: 'P0', evidence: message, expected: 'All selected checks finish and the rotated session is saved.', durationMs: 0, screenshots: [] });
}

try {
  await mkdir(TMP, { recursive: true });
  for (const file of selected) modules.push({ file, mod: await import(new URL(`./checks/${file}`, import.meta.url)) });
  ({ browser, ctx } = await openAuthed());
  for (const { file, mod } of modules) {
    console.log(`\n== ${file}`);
    try { await mod.default({ ctx, collector, shared }); }
    finally {
      await saveState(ctx);
      // Incremental report so a hang or kill never loses the modules already finished.
      await writeReport(collector.results, { startedAt, finishedAt: new Date().toISOString(), base: BASE, partial: true });
    }
  }
} catch (error) { harnessFailure(error); }
finally {
  if (ctx) {
    try { await saveState(ctx); }
    catch (error) { harnessFailure(error); }
  }
  if (browser) {
    try { await browser.close(); }
    catch (error) { harnessFailure(error); }
  }
}
if (abortReason) {
  for (const { mod } of modules) {
    for (const c of mod.checks) {
      if (collector.results.some((r) => r.id === c.id)) continue;
      collector.results.push({ id: c.id, area: c.area ?? mod.area, title: c.title, status: 'skip', evidence: `Run aborted: ${abortReason}`, expected: c.expected, durationMs: 0, screenshots: [] });
    }
  }
}
await writeReport(collector.results, { startedAt, finishedAt: new Date().toISOString(), base: BASE });
const fails = collector.results.filter((r) => r.status === 'fail').length;
console.log(`\n${collector.results.length} checks, ${fails} failed. Report: scripts/audit/functional/REPORT.md`);
process.exitCode = fails ? 1 : 0;
