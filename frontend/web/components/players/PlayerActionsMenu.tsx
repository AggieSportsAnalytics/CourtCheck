'use client'

import { MoreVertical, Pencil, Trash2 } from 'lucide-react'

import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from '@/components/ui/dropdown-menu'

type Props = {
  playerName: string
  onEdit: () => void
  onDelete: () => void
}

export function PlayerActionsMenu({ playerName, onEdit, onDelete }: Props) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        aria-label={`Actions for ${playerName}`}
        // Sits over the card's <Link>; stop the click/navigation from bubbling.
        onClick={(e) => {
          e.preventDefault()
          e.stopPropagation()
        }}
        className="flex size-8 items-center justify-center rounded-full border border-line bg-paper/80 text-ink-mute backdrop-blur-sm transition-colors hover:border-ink hover:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-court/30"
      >
        <MoreVertical className="size-[18px]" strokeWidth={1.75} />
      </DropdownMenuTrigger>
      <DropdownMenuContent
        align="end"
        className="min-w-[9rem] rounded-[12px] border-line bg-paper p-1 shadow-[var(--shadow-card)]"
        onClick={(e) => e.stopPropagation()}
      >
        <DropdownMenuItem
          className="gap-2.5 rounded-[8px] px-2.5 py-2 text-ink focus-visible:bg-shade focus:bg-shade"
          onSelect={(e) => {
            e.preventDefault()
            onEdit()
          }}
        >
          <Pencil className="size-4" strokeWidth={1.75} />
          Edit
        </DropdownMenuItem>
        <DropdownMenuItem
          className="gap-2.5 rounded-[8px] px-2.5 py-2 text-clay focus-visible:bg-[color-mix(in_srgb,var(--color-clay)_10%,transparent)] focus:bg-[color-mix(in_srgb,var(--color-clay)_10%,transparent)]"
          onSelect={(e) => {
            e.preventDefault()
            onDelete()
          }}
        >
          <Trash2 className="size-4" strokeWidth={1.75} />
          Delete
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
