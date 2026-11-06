"use client"

import { useRef } from "react"
import { Dialog, DialogContent } from "@langchain/macaw-components/Dialog"
import { Kbd } from "@langchain/macaw-components/Kbd"
import { Text } from "@langchain/macaw-components/Text"
import {
  useAvailableShortcuts,
  formatShortcut,
  type KeyboardShortcut,
} from "@/hooks/useKeyboardShortcuts"

interface KeyboardShortcutsDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function KeyboardShortcutsDialog({
  open,
  onOpenChange,
}: KeyboardShortcutsDialogProps) {
  const returnFocusRef = useRef<HTMLElement | null>(null)
  const shortcuts = useAvailableShortcuts()
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        title="Keyboard shortcuts"
        description="Navigate conversations and compose messages with your keyboard."
        className="w-[calc(100vw-2rem)] max-w-xl"
        onOpenAutoFocus={() => {
          returnFocusRef.current =
            document.activeElement instanceof HTMLElement
              ? document.activeElement
              : null
        }}
        onCloseAutoFocus={(event) => {
          if (returnFocusRef.current?.isConnected) {
            event.preventDefault()
            returnFocusRef.current.focus()
          }
        }}
      >
        <div className="space-y-6">
          {Object.entries(shortcuts).map(([category, categoryShortcuts]) => (
            <section key={category} aria-label={category} className="space-y-3">
              <Text as="h3" variant="sm" weight="semibold">
                {category}
              </Text>
              {categoryShortcuts.map(
                (shortcut: KeyboardShortcut, index: number) => (
                  <div
                    key={`${category}-${index}`}
                    className="flex items-center justify-between gap-4"
                  >
                    <Text variant="sm" color="secondary">
                      {shortcut.description}
                    </Text>
                    <Kbd className="shrink-0">{formatShortcut(shortcut)}</Kbd>
                  </div>
                )
              )}
            </section>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  )
}
