"use client"

import { useLayoutEffect, useRef } from "react"
import { MessageComposer, type MessageComposerProps } from "./message-composer"

/** The same composer used by the welcome screen, docked below the transcript. */
export function ChatInput(props: MessageComposerProps) {
  const dockRef = useRef<HTMLDivElement>(null)

  useLayoutEffect(() => {
    const dock = dockRef.current
    const shell = dock?.closest<HTMLElement>("[data-chat-shell]")
    if (!dock || !shell) return

    // Keep the sidebar divider aligned as drafts, attachments, and errors resize the dock.
    const syncHeight = () => {
      shell.style.setProperty(
        "--chat-input-height",
        `${dock.getBoundingClientRect().height}px`
      )
    }
    syncHeight()
    const observer = new ResizeObserver(syncHeight)
    observer.observe(dock)
    return () => {
      observer.disconnect()
      shell.style.removeProperty("--chat-input-height")
    }
  }, [])

  return (
    <div
      ref={dockRef}
      className="shrink-0 border-t border-subtle bg-surface-level-1 px-3 py-3 sm:px-6"
    >
      <div className="mx-auto w-full max-w-3xl">
        <MessageComposer {...props} />
      </div>
    </div>
  )
}
