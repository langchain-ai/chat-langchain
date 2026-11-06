"use client"

import { memo, useEffect, useRef, useState } from "react"
import { ChatConversation } from "@langchain/macaw-components/ChatConversation"
import { MessageItem, type MessageItemProps } from "./message-item"
import type { Message } from "@/lib/types"
import { messageStatus } from "@/lib/utils/chat/presentation"

interface MessageListProps extends Omit<
  MessageItemProps,
  "message" | "isLastAssistant"
> {
  messages: Message[]
}

export const MessageList = memo(function MessageList({
  messages,
  ...props
}: MessageListProps) {
  const lastAssistant = messages.findLast(
    (message) => message.role === "assistant"
  )
  const previousStatus = useRef<string | undefined>(undefined)
  const [announcement, setAnnouncement] = useState("")
  const status = lastAssistant ? messageStatus(lastAssistant) : undefined

  // Announce the settled reply once, never every token or historical message.
  useEffect(() => {
    if (
      previousStatus.current === "streaming" &&
      status &&
      status !== "streaming"
    ) {
      setAnnouncement(
        status === "failed"
          ? "Response failed."
          : status === "interrupted"
            ? "Response stopped."
            : "Response complete."
      )
    } else if (status === "streaming") {
      setAnnouncement("")
    }
    previousStatus.current = status
  }, [status])

  return (
    <>
      <ChatConversation
        key={messages[0]?.id ?? "empty"}
        label="Conversation"
        className="min-h-0 flex-1"
      >
        {messages.map((message) => (
          <MessageItem
            key={message.id}
            message={message}
            isLastAssistant={message.id === lastAssistant?.id}
            {...props}
          />
        ))}
      </ChatConversation>
      <span
        className="sr-only"
        role="status"
        aria-live="polite"
        aria-atomic="true"
      >
        {announcement}
      </span>
    </>
  )
})
