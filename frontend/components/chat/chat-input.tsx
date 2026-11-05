"use client"

import { MessageComposer, type MessageComposerProps } from "./message-composer"

/** The same composer used by the welcome screen, docked below the transcript. */
export function ChatInput(props: MessageComposerProps) {
  return (
    <div className="shrink-0 border-t border-subtle bg-surface-level-1 px-3 py-3 sm:px-6">
      <div className="mx-auto w-full max-w-3xl">
        <MessageComposer {...props} />
      </div>
    </div>
  )
}
