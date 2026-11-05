"use client"

import {
  Children,
  isValidElement,
  memo,
  useCallback,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react"
import ReactMarkdown, { type Components } from "react-markdown"
import remarkGfm from "remark-gfm"
import {
  ChatMessage,
  ChatMessageActions,
  ChatMessageBody,
  ChatMessageHeader,
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
import { IconButton } from "@langchain/macaw-components/IconButton"
import {
  CodeLite,
  type CodeLiteCodeProps,
} from "@langchain/macaw-components/Code/CodeLite"
import { resolveCodeLanguage } from "@langchain/macaw-components/Code/codeLanguages"
import { Link } from "@langchain/macaw-components/Link"
import { Textarea } from "@langchain/macaw-components/Textarea"
import { ChatTextIcon } from "@phosphor-icons/react/dist/ssr/ChatText"
import { ThumbsDownIcon } from "@phosphor-icons/react/dist/ssr/ThumbsDown"
import { ThumbsUpIcon } from "@phosphor-icons/react/dist/ssr/ThumbsUp"
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

// CodeLite bundles TypeScript (a JavaScript superset), Python, shell, YAML, and JSON grammars.
function codeLanguage(language: string): CodeLiteCodeProps["language"] {
  const resolved = resolveCodeLanguage(language)
  if (resolved === "javascript") return "typescript"
  return resolved === "typescript" ||
    resolved === "python" ||
    resolved === "shell" ||
    resolved === "yaml" ||
    resolved === "json"
    ? resolved
    : "plaintext"
}

// Headings, paragraphs, lists, tables, and inline code are styled by `.prose`
// in globals.css. Override the block wrapper so CodeLite never sits inside a
// native <pre>. React keeps each Markdown node mounted as its streamed text changes.
const markdownComponents: Components = {
  a: ({ href, children }) =>
    href ? (
      <Link href={href} target="_blank" rel="noopener noreferrer">
        {children}
      </Link>
    ) : (
      <>{children}</>
    ),
  pre: ({ children }) => {
    const code = Children.toArray(children).find(isValidElement)
    const props = code?.props as
      { className?: string; children?: ReactNode } | undefined
    const language =
      /language-([\w+-]+)/.exec(props?.className ?? "")?.[1] ?? "text"
    return (
      <CodeLite
        variant="block"
        label={language}
        language={codeLanguage(language)}
        value={String(props?.children ?? "").replace(/\n$/, "")}
        showGutter={false}
        virtualize={false}
        className="my-3"
      />
    )
  },
}

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
              <ol className="list-decimal space-y-2 pl-5 text-sm text-secondary">
                {steps.map((step, index) => (
                  <li key={`${message.id}-step-${index}`}>{step}</li>
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
    if (message.images?.length) {
      content.push({
        id: `${message.id}-attachments`,
        type: "custom",
        content: (
          <AttachmentList layout="wrap">
            {message.images.map((file) => {
              const imageSource =
                file.url ||
                (file.base64
                  ? `data:${file.mimeType};base64,${file.base64}`
                  : undefined)
              const isImage =
                file.mimeType.startsWith("image/") && Boolean(imageSource)
              return (
                <li key={file.id}>
                  <Attachment
                    name={file.name || "File"}
                    contentType={file.mimeType}
                    size="lg"
                    variant={isImage ? "thumbnail" : "file"}
                    metadata={
                      file.size
                        ? `${Math.round(file.size / 1024)} KB`
                        : undefined
                    }
                    preview={
                      isImage ? (
                        <img
                          src={imageSource}
                          alt={file.name || "Attached image"}
                          className="h-full w-full object-cover"
                        />
                      ) : undefined
                    }
                  />
                </li>
              )
            })}
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
                <div className="prose min-w-0 text-sm leading-relaxed [overflow-wrap:anywhere]">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={markdownComponents}
                  >
                    {message.content}
                  </ReactMarkdown>
                </div>
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
    if (showToolCalls && message.toolCalls?.length) {
      content.push({
        id: `${message.id}-tools`,
        type: "custom",
        content: (
          <ChatToolCallGroup label={`Tool calls (${message.toolCalls.length})`}>
            {message.toolCalls.map((tool) => (
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
        ),
      })
    }
    if (message.subgraphOutputs?.length) {
      content.push({
        id: `${message.id}-subagents`,
        type: "custom",
        content: (
          <ChatToolCallGroup
            label={`Subagent outputs (${message.subgraphOutputs.length})`}
          >
            {message.subgraphOutputs.map((subgraph) => {
              const id =
                subgraph.toolCallId ?? `${subgraph.name}-${subgraph.timestamp}`
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
          {isAssistant && (
            <div className="flex flex-wrap items-center gap-2 text-xs text-secondary">
              {message.thinkingDuration != null && (
                <span>{(message.thinkingDuration / 1000).toFixed(1)}s</span>
              )}
              {message.usageMetadata?.total_tokens != null && (
                <span>
                  {(message.usageMetadata.total_tokens / 1000).toFixed(1)}k
                  tokens
                </span>
              )}
              {typeof message.usageMetadata?.total_cost === "number" &&
                message.usageMetadata.total_cost > 0 && (
                  <span>${message.usageMetadata.total_cost.toFixed(4)}</span>
                )}
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
            </div>
          )}
        </>
      }
    >
      <ChatMessageHeader name={isAssistant ? "Chat LangChain" : "You"} />
      <ChatMessageBody blocks={blocks} />
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
