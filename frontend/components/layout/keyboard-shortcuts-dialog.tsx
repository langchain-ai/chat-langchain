'use client'

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { useAvailableShortcuts, formatShortcut, type KeyboardShortcut } from '@/hooks/useKeyboardShortcuts'

interface KeyboardShortcutsDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function KeyboardShortcutsDialog({
  open,
  onOpenChange,
}: KeyboardShortcutsDialogProps) {
  const shortcuts = useAvailableShortcuts()

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-2xl">Keyboard Shortcuts</DialogTitle>
        </DialogHeader>

        <div className="space-y-6 py-4">
          {Object.entries(shortcuts).map(([category, categoryShortcuts]) => (
            <div key={category}>
              <h3 className="text-base font-bold text-foreground mb-4 pb-2 border-b border-border/50">
                {category}
              </h3>
              <div className="space-y-3">
                {categoryShortcuts.map((shortcut: KeyboardShortcut, index: number) => (
                  <div
                    key={`${category}-${index}`}
                    className="flex items-center justify-between py-3 px-4 rounded-lg bg-muted/30 hover:bg-muted/50 transition-colors border border-border/30"
                  >
                    <span className="text-sm font-medium text-foreground">
                      {shortcut.description}
                    </span>
                    <kbd className="inline-flex items-center gap-1 rounded-md border-2 border-border bg-card px-3 py-1.5 font-mono text-sm font-bold text-foreground shadow-sm min-w-[60px] justify-center">
                      {formatShortcut(shortcut)}
                    </kbd>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  )
}
