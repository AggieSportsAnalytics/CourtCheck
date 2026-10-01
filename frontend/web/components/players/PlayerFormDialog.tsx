'use client'

import { useEffect, useState } from 'react'
import { toast } from 'sonner'

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'

export interface RosterPlayer {
  id: string
  name: string
  year: string | null
  position: string | null
  photo_url: string | null
  handedness?: string | null
  created_at?: string
}

const YEAR_OPTIONS = [
  { value: '', label: 'No year' },
  { value: 'Fr', label: 'Freshman' },
  { value: 'So', label: 'Sophomore' },
  { value: 'Jr', label: 'Junior' },
  { value: 'Sr', label: 'Senior' },
  { value: 'Gr', label: 'Graduate' },
] as const

type Props = {
  mode: 'add' | 'edit'
  open: boolean
  onOpenChange: (open: boolean) => void
  /** Existing player when mode === 'edit'. */
  initial?: RosterPlayer | null
  /** Called with the created/updated player after a successful save. */
  onSaved: (player: RosterPlayer) => void
}

export function PlayerFormDialog({ mode, open, onOpenChange, initial, onSaved }: Props) {
  const [name, setName] = useState('')
  const [year, setYear] = useState('')
  const [position, setPosition] = useState('')
  const [photoUrl, setPhotoUrl] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Reset the form whenever the dialog opens so stale values from a previous
  // edit don't leak into a new add (and vice versa).
  useEffect(() => {
    if (!open) return
    setName(initial?.name ?? '')
    setYear(initial?.year ?? '')
    setPosition(initial?.position ?? '')
    setPhotoUrl(initial?.photo_url ?? '')
    setError(null)
  }, [open, initial])

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    const trimmedName = name.trim()
    if (!trimmedName) {
      setError('Name is required.')
      return
    }
    if (trimmedName.length > 100) {
      setError('Name is too long (max 100 characters).')
      return
    }
    const trimmedPhoto = photoUrl.trim()
    if (trimmedPhoto && !/^https:\/\//i.test(trimmedPhoto)) {
      setError('Photo URL must start with https://')
      return
    }

    const payload = {
      name: trimmedName,
      year: year || null,
      position: position.trim() || null,
      photo_url: trimmedPhoto || null,
    }

    setSaving(true)
    setError(null)
    try {
      const url = mode === 'add' ? '/api/players' : `/api/players/${initial!.id}`
      const method = mode === 'add' ? 'POST' : 'PATCH'
      const res = await fetch(url, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        const message = typeof body?.error === 'string' ? body.error.trim() : ''
        throw new Error(
          message && message !== 'Internal server error'
            ? message
            : "Couldn't save the player. Try again.",
        )
      }

      if (mode === 'add') {
        const body = await res.json().catch(() => ({}))
        const created = body?.player as RosterPlayer | undefined
        if (created) onSaved(created)
        toast.success(`${trimmedName} added to your roster.`)
      } else {
        // PATCH returns only the changed fields; merge onto the known player.
        onSaved({ ...(initial as RosterPlayer), ...payload })
        toast.success('Player updated.')
      }
      onOpenChange(false)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong.')
    } finally {
      setSaving(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{mode === 'add' ? 'Add player' : 'Edit player'}</DialogTitle>
          <DialogDescription>
            {mode === 'add'
              ? 'Add a player to your roster. You can assign recordings to them later.'
              : 'Update this player’s details.'}
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="player-name">Name</Label>
            <Input
              id="player-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Sophie Luescher"
              maxLength={100}
              autoFocus
              required
            />
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="player-year">Year</Label>
              <select
                id="player-year"
                value={year}
                onChange={(e) => setYear(e.target.value)}
                className="h-11 rounded-[12px] border border-line bg-paper px-3 text-[1.02rem] text-ink transition-colors hover:border-ink-mute focus:border-court focus:outline-none focus-visible:ring-2 focus-visible:ring-court/20"
              >
                {YEAR_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
            </div>

            <div className="flex flex-col gap-1.5">
              <Label htmlFor="player-position">Position</Label>
              <Input
                id="player-position"
                value={position}
                onChange={(e) => setPosition(e.target.value)}
                placeholder="Optional"
                maxLength={50}
              />
            </div>
          </div>

          <div className="flex flex-col gap-1.5">
            <Label htmlFor="player-photo">Photo URL</Label>
            <Input
              id="player-photo"
              type="url"
              value={photoUrl}
              onChange={(e) => setPhotoUrl(e.target.value)}
              placeholder="https://… (optional)"
            />
          </div>

          {error && (
            <div
              role="alert"
              className="rounded-[10px] border border-clay bg-[color-mix(in_srgb,var(--color-clay)_8%,transparent)] px-3.5 py-2.5 text-[0.95rem] leading-[1.45] text-clay"
            >
              {error}
            </div>
          )}

          <DialogFooter>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={() => onOpenChange(false)}
              disabled={saving}
            >
              Cancel
            </Button>
            <Button type="submit" variant="primary" size="sm" disabled={saving}>
              {saving
                ? mode === 'add'
                  ? 'Adding…'
                  : 'Saving…'
                : mode === 'add'
                  ? 'Add player'
                  : 'Save changes'}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
