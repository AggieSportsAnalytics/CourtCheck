'use client'

import { useState } from 'react'
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
import type { RosterPlayer } from '@/components/players/PlayerFormDialog'

type Props = {
  open: boolean
  onOpenChange: (open: boolean) => void
  player: RosterPlayer | null
  /** Recordings currently assigned to this player — deleted alongside them. */
  recordingCount: number
  /** Called after a successful delete with the removed player's id. */
  onDeleted: (playerId: string) => void
}

export function DeletePlayerDialog({
  open,
  onOpenChange,
  player,
  recordingCount,
  onDeleted,
}: Props) {
  const [deleting, setDeleting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleDelete() {
    if (!player) return
    setDeleting(true)
    setError(null)
    try {
      const res = await fetch(`/api/players/${player.id}`, { method: 'DELETE' })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        const message = typeof body?.error === 'string' ? body.error.trim() : ''
        throw new Error(
          message && message !== 'Internal server error'
            ? message
            : "Couldn't delete the player. Try again.",
        )
      }
      onDeleted(player.id)
      toast.success(`${player.name} removed from your roster.`)
      onOpenChange(false)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong.')
    } finally {
      setDeleting(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Delete {player?.name ?? 'player'}?</DialogTitle>
          <DialogDescription>
            This permanently removes the player
            {recordingCount > 0 ? (
              <>
                {' '}
                and their{' '}
                <span className="font-semibold text-ink">
                  {recordingCount} recording{recordingCount === 1 ? '' : 's'}
                </span>
                , including the processed videos and heatmaps.
              </>
            ) : (
              '.'
            )}{' '}
            This can’t be undone.
          </DialogDescription>
        </DialogHeader>

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
            disabled={deleting}
          >
            Cancel
          </Button>
          <Button
            type="button"
            variant="accent"
            size="sm"
            onClick={handleDelete}
            disabled={deleting}
          >
            {deleting ? 'Deleting…' : 'Delete player'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
