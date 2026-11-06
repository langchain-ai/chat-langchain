"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { Badge } from "@langchain/macaw-components/Badge"
import { Banner } from "@langchain/macaw-components/Banner"
import type { ChatComposerInputHandle } from "@langchain/macaw-components/ChatComposer"
import { Text } from "@langchain/macaw-components/Text"
import { ChatInput } from "@/components/chat/chat-input"
import { WelcomeScreen } from "@/components/chat/features/welcome-screen"
import { MessageList } from "@/components/chat/message-list"
import type { MessageComposerProps } from "@/components/chat/message-composer"
import { Header } from "@/components/layout/header"
import { Sidebar } from "@/components/layout/sidebar"
import { KeyboardShortcutsDialog } from "@/components/layout/keyboard-shortcuts-dialog"
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts"
import { useFileUpload } from "@/lib/hooks/files/use-file-upload"
import { useVoiceInput } from "@/lib/hooks/files/use-voice-input"
import {
  INPUT_TOO_LONG_MESSAGE,
  MAX_INPUT_CHARS,
} from "@/lib/constants/features"
import type { Thread } from "@/lib/hooks/threads"
import type { ImageAttachment, Message } from "@/lib/types"

type DemoThread = { thread: Thread; messages: Message[] }
type Draft = {
  threadId: string
  content: string
  images: ImageAttachment[]
  id: string
}
type Stream = { threadId: string; messageId: string; response: string }

const agentAnswer = `## Build your first agent

Start with a model, a small set of tools, and a clear system prompt.

\`\`\`python
from langchain.agents import create_agent

agent = create_agent(
    model="openai:gpt-4.1",
    tools=[search_docs],
    system_prompt="Help developers find the right documentation.",
)
\`\`\`

| Part | Purpose |
| --- | --- |
| Model | Chooses the next action |
| Tools | Retrieve information and take actions |
| State | Carries context between steps |

- Start with one useful tool.
- Inspect the execution in **LangSmith**.
- Add checkpoints when you need durable state.

Read the [LangChain agent guide](https://docs.langchain.com/oss/python/langchain/agents) for more examples.`

function sampleResponse(prompt: string) {
  if (/trace|langsmith/i.test(prompt))
    return `## See what your agent is doing

LangSmith brings the model calls, tool inputs, outputs, and timing together in a trace.

1. Enable tracing in your application's environment.
2. Run your agent with a representative question.
3. Open the trace to inspect each step and compare responses.

Use [the tracing guide](https://docs.langchain.com/langsmith/observability) to connect your app. This sample response is generated locally for the UI demo.`
  if (/human|approv/i.test(prompt))
    return `## Keep a person in the loop

Use an interrupt before a consequential tool action, persist the graph state, and resume after a decision.

- **Approve** continues with the proposed action.
- **Edit** updates the tool arguments before continuing.
- **Reject** gives the agent feedback and lets it choose another step.

See the [human-in-the-loop guide](https://docs.langchain.com/oss/python/langchain/human-in-the-loop). This demo shows sample content; it does not execute any tools.`
  return agentAnswer
}

function initialThreads(): DemoThread[] {
  const now = Date.now()
  return [
    {
      id: "demo-agent",
      title: "Build an agent with LangChain",
      age: 0,
      question: "How do I build an agent with LangChain?",
      answer: agentAnswer,
    },
    {
      id: "demo-tracing",
      title: "Trace and debug with LangSmith",
      age: 1.5,
      question: "How do I trace my app with LangSmith?",
      answer: sampleResponse("LangSmith"),
    },
    {
      id: "demo-approval",
      title: "Add human-in-the-loop approval",
      age: 4,
      question: "How do I add human-in-the-loop approval?",
      answer: sampleResponse("approval"),
    },
    {
      id: "demo-history",
      title: "Your first documentation assistant",
      age: 14,
      question: "Where should I start?",
      answer:
        "Start with a focused question, retrieve the relevant documentation, and evaluate the answer against an example you trust.",
    },
  ].map((sample) => {
    const date = new Date(now - sample.age * 86_400_000)
    return {
      thread: {
        thread_id: sample.id,
        created_at: date.toISOString(),
        updated_at: date.toISOString(),
        metadata: {
          user_id: "local-demo",
          title: sample.title,
          lastMessage: sample.answer,
        },
      },
      messages: [
        {
          id: `${sample.id}-user`,
          role: "user",
          content: sample.question,
          timestamp: date,
        },
        {
          id: `${sample.id}-assistant`,
          role: "assistant",
          content: sample.answer,
          timestamp: date,
          runId: `${sample.id}-sample-run`,
          status: "completed",
          thinkingDuration: 4200,
          thinkingSteps: [
            "Search the relevant documentation",
            "Check the example and prepare a response",
          ],
          toolCalls: [
            {
              id: `${sample.id}-search`,
              name: "search_docs",
              args: { query: sample.question },
              output: { sources: ["LangChain documentation"], matches: 3 },
              status: "succeeded",
            },
          ],
          subgraphOutputs: [
            {
              name: "Documentation researcher",
              output: "Found the relevant guide and a minimal example.",
              timestamp: date.getTime(),
              toolCallId: `${sample.id}-research`,
              isComplete: true,
              status: "succeeded",
            },
          ],
          usageMetadata: {
            total_tokens: 1240,
            input_tokens: 880,
            output_tokens: 360,
          },
        },
      ],
    }
  })
}

/** Opt-in local review fixture. Messages, responses, and feedback stay in memory. */
export function MacawChatDemo() {
  const [threads, setThreads] = useState<DemoThread[]>(initialThreads)
  const [selectedId, setSelectedId] = useState("demo-agent")
  const [collapsed, setCollapsed] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)
  const [shortcutsOpen, setShortcutsOpen] = useState(false)
  const [input, setInput] = useState("")
  const [inputError, setInputError] = useState<string | null>(null)
  const [stream, setStream] = useState<Stream | null>(null)
  const [queue, setQueue] = useState<Draft[]>([])
  const [copiedId, setCopiedId] = useState<string | null>(null)
  const [feedbackComment, setFeedbackComment] = useState<
    Record<string, string>
  >({})
  const [showCommentInput, setShowCommentInput] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const inputRef = useRef<ChatComposerInputHandle>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const files = useFileUpload({ getInputLength: () => input.length })
  const messages =
    threads.find((item) => item.thread.thread_id === selectedId)?.messages ?? []
  const attachmentLength = files.attachedFiles.reduce(
    (total, file) =>
      total + (file.mimeType.startsWith("image/") ? 0 : (file.textLength ?? 0)),
    0
  )
  const updateInput = useCallback(
    (value: string) => {
      const maximum = Math.max(0, MAX_INPUT_CHARS - attachmentLength)
      setInput(value.slice(0, maximum))
      setInputError(value.length > maximum ? INPUT_TOO_LONG_MESSAGE : null)
    },
    [attachmentLength]
  )
  const voice = useVoiceInput({
    onTranscript: (text) => updateInput(`${input}${input ? " " : ""}${text}`),
  })

  const startResponse = useCallback(
    (draft: Draft, previousMessages?: Message[]) => {
      const assistantId = crypto.randomUUID()
      const timestamp = new Date()
      const userMessage: Message = {
        id: draft.id,
        role: "user",
        content: draft.content,
        timestamp,
        images: draft.images,
      }
      const assistantMessage: Message = {
        id: assistantId,
        runId: `${assistantId}-sample-run`,
        role: "assistant",
        content: "",
        timestamp,
        status: "streaming",
        isThinking: true,
        thinkingStartTime: timestamp.getTime(),
        thinkingSteps: ["Search the documentation"],
        toolCalls: [
          {
            id: `${assistantId}-search`,
            name: "search_docs",
            args: { query: draft.content },
            status: "running",
          },
        ],
      }
      setThreads((current) =>
        current.map((item) =>
          item.thread.thread_id === draft.threadId
            ? {
                thread: {
                  ...item.thread,
                  updated_at: timestamp.toISOString(),
                  metadata: {
                    ...item.thread.metadata,
                    title: draft.content.slice(0, 60) || "Attached files",
                    lastMessage: draft.content,
                  },
                },
                messages: [
                  ...(previousMessages ?? item.messages),
                  userMessage,
                  assistantMessage,
                ],
              }
            : item
        )
      )
      setStream({
        threadId: draft.threadId,
        messageId: assistantId,
        response: sampleResponse(draft.content),
      })
    },
    []
  )

  useEffect(() => {
    if (!stream) return
    let length = 0
    const started = Date.now()
    const timer = window.setInterval(() => {
      length = Math.min(stream.response.length, length + 15)
      const done = length === stream.response.length
      setThreads((current) =>
        current.map((item) =>
          item.thread.thread_id === stream.threadId
            ? {
                ...item,
                messages: item.messages.map((message) =>
                  message.id === stream.messageId
                    ? {
                        ...message,
                        content: stream.response.slice(0, length),
                        isThinking: false,
                        status: done ? "completed" : "streaming",
                        thinkingDuration: Date.now() - started,
                        thinkingSteps: [
                          "Search the documentation",
                          "Prepare a clear example",
                        ],
                        toolCalls: message.toolCalls?.map((tool) => ({
                          ...tool,
                          status: "succeeded",
                          output: {
                            sources: ["LangChain documentation"],
                            matches: 3,
                          },
                        })),
                        usageMetadata: done
                          ? { total_tokens: 1240 }
                          : undefined,
                      }
                    : message
                ),
              }
            : item
        )
      )
      if (done) setStream(null)
    }, 55)
    return () => window.clearInterval(timer)
  }, [stream])

  useEffect(() => {
    if (stream || !queue.length) return
    const [next, ...remaining] = queue
    setQueue(remaining)
    startResponse(next)
  }, [queue, startResponse, stream])

  const stopResponse = () => {
    if (stream)
      setThreads((current) =>
        current.map((item) => ({
          ...item,
          messages: item.messages.map((message) =>
            message.id === stream.messageId
              ? {
                  ...message,
                  status: "interrupted",
                  wasInterrupted: true,
                  isThinking: false,
                }
              : message
          ),
        }))
      )
    setStream(null)
    setQueue([])
  }
  const send = () => {
    if (!input.trim() && !files.attachedFiles.length) return
    const draft = {
      id: crypto.randomUUID(),
      threadId: selectedId,
      content: input.trim(),
      images: files.attachedFiles,
    }
    if (stream) setQueue((current) => [...current, draft])
    else startResponse(draft)
    setInput("")
    setInputError(null)
    files.clearFiles()
    voice.stopListening()
  }
  const newChat = () => {
    stopResponse()
    const id = crypto.randomUUID()
    const date = new Date().toISOString()
    setThreads((current) => [
      {
        thread: {
          thread_id: id,
          created_at: date,
          updated_at: date,
          metadata: { user_id: "local-demo", title: "New conversation" },
        },
        messages: [],
      },
      ...current,
    ])
    setSelectedId(id)
    setInput("")
    setInputError(null)
    files.clearFiles()
    voice.stopListening()
    requestAnimationFrame(() => inputRef.current?.focus())
  }
  const editAndRerun = (messageId: string, content: string) => {
    stopResponse()
    const index = messages.findIndex((message) => message.id === messageId)
    const original = messages[index]
    if (index < 0 || !original) return
    startResponse(
      {
        id: original.id,
        threadId: selectedId,
        content,
        images: original.images ?? [],
      },
      messages.slice(0, index)
    )
  }
  const regenerate = () => {
    const lastUser = messages.findLast((message) => message.role === "user")
    if (lastUser) editAndRerun(lastUser.id, lastUser.content)
  }
  const feedback = (
    id: string,
    value: "positive" | "negative",
    comment?: string
  ) => {
    setThreads((current) =>
      current.map((item) => ({
        ...item,
        messages: item.messages.map((message) =>
          message.id === id
            ? { ...message, feedback: value, feedbackComment: comment }
            : message
        ),
      }))
    )
    setShowCommentInput(null)
    setNotice("Feedback saved for this demo session.")
  }
  useKeyboardShortcuts([
    {
      shortcut: {
        key: "/",
        metaKey: true,
        description: "Toggle keyboard shortcuts",
        category: "Navigation",
      },
      handler: () => setShortcutsOpen((value) => !value),
    },
    {
      shortcut: {
        key: "b",
        metaKey: true,
        description: "Toggle sidebar",
        category: "Navigation",
      },
      handler: () => setCollapsed((value) => !value),
    },
    {
      shortcut: {
        key: "i",
        metaKey: true,
        description: "Create new chat",
        category: "Navigation",
      },
      handler: newChat,
    },
  ])

  const composer: MessageComposerProps = {
    input,
    onInputChange: updateInput,
    onSend: send,
    isLoading: stream !== null,
    isStopping: false,
    onStop: stopResponse,
    userId: "local-demo",
    attachedFiles: files.attachedFiles,
    uploadError: files.uploadError,
    inputError,
    isDragging: files.isDragging,
    onDragOver: files.handleDragOver,
    onDragLeave: files.handleDragLeave,
    onDrop: files.handleDrop,
    onPaste: files.handlePaste,
    onRemoveFile: files.removeFile,
    onFileButtonClick: () => fileInputRef.current?.click(),
    fileInputRef,
    onFileSelect: files.handleFileSelect,
    textareaRef: inputRef,
    isVoiceListening: voice.isListening,
    isVoiceSupported: voice.isSupported,
    onVoiceToggle: voice.toggleListening,
    voiceError: voice.error,
    queuedMessages: queue.map((draft) => ({
      content: draft.content,
      id: draft.id,
    })),
  }

  return (
    <main className="flex h-dvh min-h-0 flex-col bg-background text-primary">
      <div className="flex shrink-0 flex-wrap items-center justify-between gap-2 border-b border-subtle px-4 py-2">
        <Badge color="special">Interactive demo · Sample responses</Badge>
        <Text variant="xs" color="tertiary">
          Local session · No model or tool calls · Refresh to reset
        </Text>
      </div>
      <div data-chat-shell className="flex min-h-0 flex-1">
        <Sidebar
          isCollapsed={collapsed}
          onToggle={() => setCollapsed((value) => !value)}
          threads={threads.map((item) => item.thread)}
          currentThreadId={selectedId}
          onSelectThread={(id) => {
            stopResponse()
            setSelectedId(id)
            setInput("")
            setInputError(null)
            files.clearFiles()
            voice.stopListening()
          }}
          onDeleteThread={(id) => {
            if (id === selectedId) newChat()
            setThreads((current) =>
              current.filter((item) => item.thread.thread_id !== id)
            )
          }}
          mobileOpen={mobileOpen}
          onMobileOpenChange={setMobileOpen}
        />
        <div className="flex min-w-0 flex-1 flex-col">
          <Header
            onNewChat={newChat}
            onOpenSidebar={() => setMobileOpen(true)}
            showInitialAuthPrompt={false}
          />
          {notice && (
            <Banner intent="info" dismissible onDismiss={() => setNotice(null)}>
              {notice}
            </Banner>
          )}
          {messages.length ? (
            <>
              <MessageList
                key={selectedId}
                messages={messages}
                showToolCalls
                isRegenerating={stream !== null}
                copiedId={copiedId}
                onCopy={async (content, id) => {
                  try {
                    await navigator.clipboard.writeText(content)
                    setCopiedId(id)
                  } catch {
                    setNotice(
                      "Clipboard access is unavailable in this browser."
                    )
                  }
                }}
                onRegenerate={regenerate}
                onEditAndRerun={editAndRerun}
                feedbackComment={feedbackComment}
                showCommentInput={showCommentInput}
                onFeedback={feedback}
                onSubmitComment={(id) =>
                  feedback(
                    id,
                    messages.find((message) => message.id === id)?.feedback ??
                      "positive",
                    feedbackComment[id]
                  )
                }
                onCancelComment={() => setShowCommentInput(null)}
                onToggleComment={(id) =>
                  setShowCommentInput((current) => (current === id ? null : id))
                }
                setFeedbackComment={setFeedbackComment}
              />
              <ChatInput {...composer} />
            </>
          ) : (
            <WelcomeScreen {...composer} />
          )}
        </div>
      </div>
      <KeyboardShortcutsDialog
        open={shortcutsOpen}
        onOpenChange={setShortcutsOpen}
      />
    </main>
  )
}
