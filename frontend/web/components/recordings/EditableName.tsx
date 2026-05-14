'use client';

import { useEffect, useRef, useState } from 'react';

type Props = {
  recordingId: string;
  initialName: string;
  /** Where this is rendered controls its style:
   *   - 'title': inline next to the h1 on the detail page
   *   - 'row':   compact icon button in the recordings-list row
   *   - 'menu':  full-width pill (e.g., inside a row that has already opened
   *              into a rename-confirm pane) */
  variant?: 'title' | 'row';
  /** Receives the saved name so the parent can sync local state. */
  onSaved?: (newName: string) => void;
};

const MAX_LEN = 100;

/**
 * Pencil button that toggles into an inline input. Saves via PATCH
 * /api/recordings/[id]. Local state owns the input value during editing so
 * arrow keys don't accidentally fire route-level seek shortcuts.
 */
export default function EditableName({ recordingId, initialName, variant = 'title', onSaved }: Props) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(initialName);
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (editing) {
      setValue(initialName);
      setErr(null);
      requestAnimationFrame(() => {
        inputRef.current?.focus();
        inputRef.current?.select();
      });
    }
  }, [editing, initialName]);

  const submit = async () => {
    const trimmed = value.trim();
    if (!trimmed) {
      setErr('Name cannot be empty.');
      return;
    }
    if (trimmed.length > MAX_LEN) {
      setErr(`Name too long (max ${MAX_LEN}).`);
      return;
    }
    if (trimmed === initialName) {
      setEditing(false);
      return;
    }
    setSaving(true);
    setErr(null);
    try {
      const res = await fetch(`/api/recordings/${recordingId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: trimmed }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || 'Failed to rename');
      }
      onSaved?.(trimmed);
      setEditing(false);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setSaving(false);
    }
  };

  if (!editing) {
    return (
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          setEditing(true);
        }}
        aria-label="Rename recording"
        className={
          variant === 'title'
            ? 'inline-flex items-center justify-center w-8 h-8 rounded-full border border-line bg-paper text-ink-mute hover:text-ink hover:border-ink-mute cursor-pointer'
            : 'inline-flex items-center justify-center w-7 h-7 rounded-full border border-line bg-paper text-ink-mute hover:text-court hover:border-court cursor-pointer'
        }
        style={{ transition: 'border-color var(--duration-quick) var(--ease-out), color var(--duration-quick) var(--ease-out)' }}
      >
        <PencilIcon size={variant === 'title' ? 14 : 13} />
      </button>
    );
  }

  return (
    <div
      className="inline-flex items-center gap-2"
      onClick={(e) => e.stopPropagation()}
    >
      <input
        ref={inputRef}
        type="text"
        value={value}
        maxLength={MAX_LEN}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter') {
            e.preventDefault();
            void submit();
          } else if (e.key === 'Escape') {
            e.preventDefault();
            setEditing(false);
          }
          e.stopPropagation();
        }}
        disabled={saving}
        className={
          variant === 'title'
            ? 'w-[min(420px,60vw)] rounded-[10px] border border-line bg-paper px-3 py-2 text-[0.95rem] text-ink outline-none focus:border-ink'
            : 'w-[220px] rounded-[10px] border border-line bg-paper px-3 py-1.5 text-[0.9rem] text-ink outline-none focus:border-ink'
        }
        aria-label="Recording name"
      />
      <button
        type="button"
        onClick={() => void submit()}
        disabled={saving}
        className="inline-flex items-center px-3 py-1.5 rounded-full bg-ink text-cream text-[0.82rem] font-medium hover:-translate-y-px disabled:opacity-60 disabled:cursor-not-allowed cursor-pointer dark:bg-court-deep"
        style={{ transition: 'transform var(--duration-quick) var(--ease-spring)' }}
      >
        {saving ? 'Saving…' : 'Save'}
      </button>
      <button
        type="button"
        onClick={() => setEditing(false)}
        disabled={saving}
        className="inline-flex items-center px-3 py-1.5 rounded-full border border-line bg-paper text-ink-soft hover:text-ink hover:border-ink-mute text-[0.82rem] font-medium cursor-pointer"
      >
        Cancel
      </button>
      {err && <span className="text-[0.78rem] text-clay">{err}</span>}
    </div>
  );
}

function PencilIcon({ size = 14 }: { size?: number }) {
  return (
    <svg
      viewBox="0 0 24 24"
      width={size}
      height={size}
      fill="none"
      stroke="currentColor"
      strokeWidth={1.75}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d="M12 20h9" />
      <path d="M16.5 3.5a2.121 2.121 0 1 1 3 3L7 19l-4 1 1-4Z" />
    </svg>
  );
}
