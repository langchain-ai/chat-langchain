"use client"

import { memo, useEffect, useRef, useState } from "react"
import { ChatConversation } from "@langchain/macaw-components/ChatConversation"
import { Skeleton } from "@langchain/macaw-components/Skeleton"
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

// Widths vary so the placeholder reads as prose rather than a uniform block.
const SKELETON_TURNS = [
  { question: "w-2/5", answer: ["w-full", "w-11/12", "w-4/5", "w-3/5"] },
  { question: "w-1/3", answer: ["w-full", "w-5/6", "w-2/3"] },
]

/** Holds the conversation's shape inside the same column while a thread loads. */
export function MessageListSkeleton() {
  return (
    <>
      <ChatConversation label="Conversation" className="min-h-0 flex-1">
        {SKELETON_TURNS.flatMap((turn, index) => [
          <Skeleton
            key={`question-${index}`}
            className={`h-10 shrink-0 self-end rounded-xl motion-reduce:animate-none ${turn.question}`}
          />,
          <div
            key={`answer-${index}`}
            aria-hidden="true"
            className="flex flex-col gap-space-2"
          >
            {turn.answer.map((width, line) => (
              <Skeleton
                key={line}
                className={`motion-reduce:animate-none ${width}`}
              />
            ))}
          </div>,
        ])}
      </ChatConversation>
      <span className="sr-only" role="status">
        Loading conversation
      </span>
    </>
  )
}
