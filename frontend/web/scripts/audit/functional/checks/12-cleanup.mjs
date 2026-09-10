import { check, assert, expect, note, probe, visit, getData, deleteQaRecording, runModule } from '../lib.mjs';

export const area = 'recordings';
export const checks = [
  check('CLN-01', 'The audit-created recording could not be safely removed.', 'Only this run’s QA-prefix recording is deleted via UI; DELETE 204, row absent and GET 404.', async (page, r, shared) => {
    if (!shared.qaRecordingId) return { skip: 'This run created no recording; no existing recording will be deleted.' };
    await visit(page, '/recordings', r);
    await deleteQaRecording(page, shared, r);
    note(r, `Removed only audit-created ID ${shared.qaRecordingId}.`);
    shared.qaRecordingDeleted = true;
  }),
  check('CLN-02', 'Temporary audit edits were not fully restored.', 'Favorites, handedness, recording name, notes and display name equal their captured original values.', async (page, r, shared) => {
    if (!shared.originals.length && shared.displayNameOriginal === undefined) return { skip: 'No mutation modules ran; no original-value snapshots were captured.' };
    await visit(page, '/settings', r);
    for (const original of shared.originals) await probe(page, r, original.path, async () => {
      const current = await getData(page, original.path, original.key);
      for (const [key, value] of Object.entries(original.values)) {
        const equal = JSON.stringify(current[key]) === JSON.stringify(value);
        note(r, { path: original.path, field: key, restored: equal });
        assert(equal, `${original.path} ${key}: original ${JSON.stringify(value)}, now ${JSON.stringify(current[key])}`);
      }
    });
    if (shared.displayNameOriginal !== undefined) await probe(page, r, 'display name', async () => {
      await expect(page.getByPlaceholder('Your name', { exact: true })).toHaveValue(shared.displayNameOriginal);
      note(r, 'Display name equals its original value after reloading settings.');
    });
    if (shared.qaPlayerId) {
      note(r, `Permanent QA player ID: ${shared.qaPlayerId}; no player DELETE endpoint exists.`);
      assert(false, 'Audit-created player cannot be removed because the application has no player DELETE endpoint.');
    }
  }),
];
export default async function (args) { await runModule(args, checks, area); }
