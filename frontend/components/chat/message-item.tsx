"use client"

import { memo, useCallback, useEffect, useMemo, useState } from "react"
import {
  ChatMessage,
  ChatMessageActions,
  ChatMessageBody,
  type ChatMessageContentBlock,
} from "@langchain/macaw-components/ChatMessage"
import { ChatRichContent } from "@langchain/macaw-components/ChatRichContent"
import { ChatState } from "@langchain/macaw-components/ChatState"
import {
  ChatToolCall,
  ChatToolCallGroup,
  ChatToolCallReasoning,
} from "@langchain/macaw-components/ChatToolCall"
import {
  Attachment,
  AttachmentList,
} from "@langchain/macaw-components/Attachment"
import { Button } from "@langchain/macaw-components/Button"
import { Icon, type IconComponent } from "@langchain/macaw-components/Icon"
import { IconButton } from "@langchain/macaw-components/IconButton"
import { ImagePreviewBlock } from "@langchain/macaw-components/ImagePreviewBlock"
import { Link } from "@langchain/macaw-components/Link"
import { MarkdownDisplay } from "@langchain/macaw-components/MarkdownDisplay"
import { Text } from "@langchain/macaw-components/Text"
import { Textarea } from "@langchain/macaw-components/Textarea"
import { ChatTextIcon } from "@phosphor-icons/react/dist/ssr/ChatText"
import { CoinsIcon } from "@phosphor-icons/react/dist/ssr/Coins"
import { ThumbsDownIcon } from "@phosphor-icons/react/dist/ssr/ThumbsDown"
import { ThumbsUpIcon } from "@phosphor-icons/react/dist/ssr/ThumbsUp"
import { TimerIcon } from "@phosphor-icons/react/dist/ssr/Timer"
import type { Message } from "@/lib/types"
import {
  INPUT_TOO_LONG_MESSAGE,
  MAX_INPUT_CHARS,
} from "@/lib/constants/features"
import {
  messageStatus,
  subgraphStatus,
  toolStatus,
} from "@/lib/utils/chat/presentation"

export interface MessageItemProps {
  message: Message
  showToolCalls?: boolean
  isLastAssistant: boolean
  isRegenerating: boolean
  copiedId: string | null
  onCopy: (content: string, messageId: string) => void
  onRegenerate: () => void
  onEditAndRerun?: (messageId: string, newContent: string) => void
  feedbackComment: { [messageId: string]: string }
  showCommentInput: string | null
  onFeedback: (
    messageId: string,
    feedbackType: "positive" | "negative",
    comment?: string
  ) => void
  onSubmitComment: (messageId: string) => void
  onCancelComment: (messageId: string) => void
  onToggleComment: (messageId: string) => void
  setFeedbackComment: React.Dispatch<
    React.SetStateAction<{ [messageId: string]: string }>
  >
}

export const MessageItem = memo(function MessageItem({
  message,
  showToolCalls,
  isLastAssistant,
  isRegenerating,
  copiedId,
  onCopy,
  onRegenerate,
  onEditAndRerun,
  feedbackComment,
  showCommentInput,
  onFeedback,
  onSubmitComment,
  onCancelComment,
  onToggleComment,
  setFeedbackComment,
}: MessageItemProps) {
  const [isEditing, setIsEditing] = useState(false)
  const editLimit = Math.max(
    0,
    MAX_INPUT_CHARS -
      (message.images ?? []).reduce(
        (length, file) =>
          length +
          (file.mimeType.startsWith("image/") ? 0 : (file.textLength ?? 0)),
        0
      )
  )
  const [editContent, setEditContent] = useState(message.content)
  const [editError, setEditError] = useState<string | null>(null)
  const status = messageStatus(message)
  const isAssistant = message.role === "assistant"
  const isStreaming = status === "streaming"
  const hasFeedback =
    isAssistant &&
    Boolean(message.runId && (message.shareUrl || message.usageMetadata))
  const steps = message.thinkingSteps ?? []
  const metadata: { label: string; value: string; icon?: IconComponent }[] = []
  if (isAssistant) {
    if (message.thinkingDuration != null)
      metadata.push({
        label: "Response time",
        value: `${(message.thinkingDuration / 1000).toFixed(1)}s`,
        icon: TimerIcon,
      })
    // The word "tokens" stays visible: a coin alone would read as cost.
    if (message.usageMetadata?.total_tokens != null)
      metadata.push({
        label: "Total tokens",
        value: `${(message.usageMetadata.total_tokens / 1000).toFixed(1)}k tokens`,
        icon: CoinsIcon,
      })
    const cost = message.usageMetadata?.total_cost
    if (typeof cost === "number" && cost > 0)
      metadata.push({ label: "Cost", value: `$${cost.toFixed(4)}` })
  }

  useEffect(() => {
    if (!isEditing) setEditContent(message.content)
  }, [message.content, isEditing])

  const startEdit = useCallback(() => {
    setEditContent(message.content)
    setEditError(
      message.content.length > editLimit ? INPUT_TOO_LONG_MESSAGE : null
    )
    setIsEditing(true)
  }, [message.content, editLimit])

  const saveEdit = useCallback(() => {
    if (
      !editContent.trim() ||
      isRegenerating ||
      editContent.length > editLimit
    ) {
      setEditError(
        isRegenerating
          ? "Wait for the current response to finish."
          : editContent.length > editLimit
            ? INPUT_TOO_LONG_MESSAGE
            : "Enter a message before sending."
      )
      return false
    }
    onEditAndRerun?.(message.id, editContent.trim())
    setIsEditing(false)
    setEditError(null)
    return true
  }, [editContent, editLimit, isRegenerating, message.id, onEditAndRerun])

  const blocks = useMemo<ChatMessageContentBlock[]>(() => {
    const content: ChatMessageContentBlock[] = []
    if (
      isAssistant &&
      (isStreaming ||
        status === "failed" ||
        status === "interrupted" ||
        steps.length > 0)
    ) {
      content.push({
        id: `${message.id}-progress`,
        type: "custom",
        content:
          steps.length > 0 ? (
            <ChatToolCallReasoning
              label={`Agent steps (${steps.length})`}
              status={status}
              elapsedMs={message.thinkingDuration}
            >
              <ol className="list-decimal space-y-1 pl-4 text-xs text-tertiary">
                {steps.map((step, index) => (
                  <li key={`${message.id}-step-${index}`}>
                    <Text variant="xs" color="tertiary">
                      {step}
                    </Text>
                  </li>
                ))}
              </ol>
            </ChatToolCallReasoning>
          ) : (
            <ChatState
              status={status}
              label={
                isStreaming
                  ? "Working on your answer"
                  : status === "failed"
                    ? "Response failed"
                    : "Response stopped"
              }
              showElapsed={isStreaming}
              elapsedMs={message.thinkingDuration}
            />
          ),
      })
    }
    const images: { id: string; name: string; src: string }[] = []
    const files: NonNullable<Message["images"]> = []
    for (const file of message.images ?? []) {
      const src =
        file.url ||
        (file.base64 ? `data:${file.mimeType};base64,${file.base64}` : "")
      if (file.mimeType.startsWith("image/") && src)
        images.push({ id: file.id, name: file.name || "Attached image", src })
      else files.push(file)
    }
    if (images.length) {
      content.push({
        id: `${message.id}-images`,
        type: "custom",
        content: (
          <div className="flex flex-col items-end gap-2">
            {images.map((image) => (
              <ImagePreviewBlock
                key={image.id}
                src={image.src}
                altText={image.name}
                caption={image.name}
                className="w-72 max-w-full"
              />
            ))}
          </div>
        ),
      })
    }
    if (files.length) {
      content.push({
        id: `${message.id}-attachments`,
        type: "custom",
        content: (
          <AttachmentList layout="wrap">
            {files.map((file) => (
              <li key={file.id}>
                <Attachment
                  name={file.name || "File"}
                  contentType={file.mimeType}
                  size="lg"
                  variant="file"
                  metadata={
                    file.size ? `${Math.round(file.size / 1024)} KB` : undefined
                  }
                />
              </li>
            ))}
          </AttachmentList>
        ),
      })
    }
    if (message.content && !isEditing) {
      content.push(
        isAssistant
          ? {
              id: `${message.id}-content`,
              type: "custom",
              content: (
                <MarkdownDisplay
                  content={message.content}
                  isStreaming={isStreaming}
                />
              ),
            }
          : {
              id: `${message.id}-content`,
              type: "text",
              text: message.content,
              appearance: "bubble",
            }
      )
    }
    const toolCalls = showToolCalls ? (message.toolCalls ?? []) : []
    const subgraphs = message.subgraphOutputs ?? []
    if (toolCalls.length || subgraphs.length) {
      content.push({
        id: `${message.id}-activity`,
        type: "custom",
        // One block keeps both groups on the row rhythm of a single
        // ChatToolCallGroup instead of the larger gap between message blocks.
        content: (
          <div className="flex min-w-0 flex-col">
            {toolCalls.length > 0 && (
              <ChatToolCallGroup label={`Tool calls (${toolCalls.length})`}>
                {toolCalls.map((tool) => (
                  <ChatToolCall
                    key={tool.id}
                    toolCall={{
                      id: tool.id,
                      name: tool.name,
                      status: toolStatus(tool, message),
                      args: tool.args,
                      result: tool.output,
                      error: tool.error,
                    }}
                  />
                ))}
              </ChatToolCallGroup>
            )}
            {subgraphs.length > 0 && (
              <ChatToolCallGroup
                label={`Subagent outputs (${subgraphs.length})`}
              >
                {subgraphs.map((subgraph) => {
                  const id =
                    subgraph.toolCallId ??
                    `${subgraph.name}-${subgraph.timestamp}`
                  return (
                    <ChatToolCall
                      key={id}
                      label={subgraph.name}
                      summary={null}
                      toolCall={{
                        id,
                        name: subgraph.name,
                        status: subgraphStatus(subgraph, message),
                        result: subgraph.output || undefined,
                        error: subgraph.error,
                      }}
                    />
                  )
                })}
              </ChatToolCallGroup>
            )}
          </div>
        ),
      })
    }
    return content
  }, [
    message,
    status,
    isAssistant,
    isStreaming,
    isEditing,
    showToolCalls,
    steps,
  ])

  return (
    <ChatMessage
      message={{ id: message.id, role: message.role, status, content: blocks }}
      isLatest={isLastAssistant || isEditing || showCommentInput === message.id}
      footer={
        <>
          <ChatMessageActions
            status={status}
            actions={{
              copy: message.content
                ? {
                    onClick: () => onCopy(message.content, message.id),
                    label: copiedId === message.id ? "Copied" : "Copy message",
                  }
                : undefined,
              edit:
                !isAssistant && onEditAndRerun && !isRegenerating
                  ? { onClick: startEdit, label: "Edit and rerun" }
                  : undefined,
              retry:
                isAssistant && isLastAssistant && !isRegenerating
                  ? { onClick: onRegenerate, label: "Regenerate response" }
                  : undefined,
            }}
          >
            {/* Match ChatMessageActions' built-in xs icon actions. */}
            {hasFeedback && (
              <>
                <IconButton
                  icon={ThumbsUpIcon}
                  label="Helpful"
                  size="xs"
                  variant="plain"
                  color={
                    message.feedback === "positive" ? "primary" : "secondary"
                  }
                  iconWeight={
                    message.feedback === "positive" ? "fill" : "regular"
                  }
                  aria-pressed={message.feedback === "positive"}
                  onClick={() =>
                    onFeedback(
                      message.id,
                      "positive",
                      feedbackComment[message.id]
                    )
                  }
                />
                <IconButton
                  icon={ThumbsDownIcon}
                  label="Not helpful"
                  size="xs"
                  variant="plain"
                  color={
                    message.feedback === "negative" ? "primary" : "secondary"
                  }
                  iconWeight={
                    message.feedback === "negative" ? "fill" : "regular"
                  }
                  aria-pressed={message.feedback === "negative"}
                  onClick={() =>
                    onFeedback(
                      message.id,
                      "negative",
                      feedbackComment[message.id]
                    )
                  }
                />
                <IconButton
                  icon={ChatTextIcon}
                  label="Add feedback comment"
                  size="xs"
                  variant="plain"
                  color="secondary"
                  iconWeight="regular"
                  aria-expanded={showCommentInput === message.id}
                  onClick={() => onToggleComment(message.id)}
                />
              </>
            )}
          </ChatMessageActions>
          {(metadata.length > 0 || message.shareUrl) && (
            // Like earlier messages' footers, metadata appears on hover or
            // focus; touch screens, which cannot hover, always show it.
            <span className="inline-flex flex-wrap items-center gap-x-3 gap-y-1 pl-1 transition-opacity duration-fast group-focus-within/chat-message:transition-none motion-reduce:transition-none pointer-fine:opacity-0 pointer-fine:group-hover/chat-message:opacity-100 pointer-fine:group-focus-within/chat-message:opacity-100">
              {metadata.map((item) => (
                <Text
                  key={item.label}
                  as="span"
                  variant="xs"
                  color="quaternary"
                  title={item.label}
                  className="inline-flex items-center gap-1 tabular-nums"
                >
                  {item.icon && (
                    // Icon defaults to text-primary instead of inheriting.
                    <Icon
                      icon={item.icon}
                      size="xs"
                      className="text-quaternary"
                      aria-hidden
                    />
                  )}
                  <span className="sr-only">{item.label}: </span>
                  {item.value}
                </Text>
              ))}
              {message.shareUrl && (
                <Link
                  href={message.shareUrl}
                  variant="xs"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  View trace
                </Link>
              )}
            </span>
          )}
        </>
      }
    >
      {/* ChatMessage only end-aligns the body it renders itself. */}
      <ChatMessageBody
        blocks={blocks}
        className={isAssistant ? undefined : "items-end"}
      />
      {isEditing && (
        <ChatRichContent
          title="Edit message"
          description="Send an updated message to rerun the conversation from here."
          state={editError ? "validation-error" : "unanswered"}
          message={editError ?? undefined}
          submitLabel="Save and rerun"
          skipLabel="Cancel"
          onSubmit={() => {
            if (!saveEdit()) throw new Error("Message validation failed")
          }}
          onSkip={() => setIsEditing(false)}
        >
          {({ disabled, isError, errorId }) => (
            <Textarea
              aria-label="Edit message"
              value={editContent}
              onChange={(value) => {
                setEditError(
                  value.length > editLimit ? INPUT_TOO_LONG_MESSAGE : null
                )
                setEditContent(value)
              }}
              maxLength={editLimit}
              rows={4}
              autoFocus
              disabled={disabled}
              isError={isError}
              aria-describedby={errorId}
              onBeforeInput={(event) => {
                const inserted = (event.nativeEvent as InputEvent).data
                const target = event.currentTarget
                if (
                  inserted &&
                  target.value.length -
                    (target.selectionEnd - target.selectionStart) +
                    inserted.length >
                    editLimit
                ) {
                  event.preventDefault()
                  setEditError(INPUT_TOO_LONG_MESSAGE)
                }
              }}
              onPaste={(event) => {
                const target = event.currentTarget
                if (
                  target.value.length -
                    (target.selectionEnd - target.selectionStart) +
                    event.clipboardData.getData("text").length >
                  editLimit
                ) {
                  event.preventDefault()
                  setEditError(INPUT_TOO_LONG_MESSAGE)
                }
              }}
              onKeyDown={(event) => {
                if (
                  event.key === "Enter" &&
                  !event.shiftKey &&
                  !event.nativeEvent.isComposing
                ) {
                  event.preventDefault()
                  saveEdit()
                } else if (event.key === "Escape") setIsEditing(false)
              }}
            />
          )}
        </ChatRichContent>
      )}
      {showCommentInput === message.id && (
        <div className="w-full space-y-2 rounded-xl border border-subtle bg-elevated p-4">
          <Textarea
            label="Feedback about this response"
            value={feedbackComment[message.id] || ""}
            onChange={(value) =>
              setFeedbackComment((previous) => ({
                ...previous,
                [message.id]: value,
              }))
            }
            placeholder="What worked, or what could be better?"
            hintText={
              !message.feedback
                ? "Choose Helpful or Not helpful before submitting."
                : undefined
            }
            rows={3}
            autoFocus
            onKeyDown={(event) => {
              if (
                event.key === "Enter" &&
                !event.shiftKey &&
                !event.nativeEvent.isComposing
              ) {
                event.preventDefault()
                if (message.feedback && feedbackComment[message.id]?.trim())
                  onSubmitComment(message.id)
              } else if (event.key === "Escape") onCancelComment(message.id)
            }}
          />
          <div className="flex gap-2">
            <Button
              size="sm"
              disabled={
                !message.feedback || !feedbackComment[message.id]?.trim()
              }
              onClick={() => onSubmitComment(message.id)}
            >
              Submit feedback
            </Button>
            <Button
              size="sm"
              color="secondary"
              variant="plain"
              onClick={() => onCancelComment(message.id)}
            >
              Cancel
            </Button>
          </div>
        </div>
      )}
    </ChatMessage>
  )
})
