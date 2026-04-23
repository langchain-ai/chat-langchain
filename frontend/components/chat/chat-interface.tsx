"use client"

import { useState, useEffect, useCallback, useRef, useMemo } from "react"
import type { ClientProfile } from "@/lib/hooks"
import { Client } from "@langchain/langgraph-sdk"
import { readRun, shareRun } from "@/lib/api/langsmith"
import type { Message, ImageAttachment } from "@/lib/types"
import { createUserMessage, generateMessageId, extractTextFromContent } from "@/lib/utils/chat"
import { truncate } from "@/lib/utils/string"
import { useStreamHandler, useFeedback, useChatState } from "@/lib/hooks/chat"
import { useUserId } from "@/lib/hooks/auth"
import { useFileUpload, useVoiceInput } from "@/lib/hooks/files"
import { MessageList } from "./message-list"
import { WelcomeScreen } from "./features/welcome-screen"
import { ChatInput } from "./chat-input"
import type { AgentConfig } from "@/components/layout/agent-settings"
import { LANGGRAPH_API_URL, LANGSMITH_API_KEY } from "@/lib/constants/api"

// Enhanced scrollbar styles with smooth transitions
const scrollbarStyles = `
  .custom-scrollbar {
    scroll-behavior: smooth;
    will-change: scroll-position;
  }
  .custom-scrollbar::-webkit-scrollbar {
    width: 6px;
  }
  .custom-scrollbar::-webkit-scrollbar-track {
    background: transparent;
  }
  .custom-scrollbar::-webkit-scrollbar-thumb {
    background: #7FC8FF;
    border-radius: 3px;
    transition: background 0.2s ease;
  }
  .custom-scrollbar::-webkit-scrollbar-thumb:hover {
    background: #7FC8FF;
  }
  .custom-scrollbar::-webkit-scrollbar-thumb:active {
    background: #7FC8FF;
  }
`

interface ChatInterfaceProps {
  showToolCalls?: boolean
  threadId: string
  onThreadUpdate?: (threadId: string, title: string, lastMessage: string, client?: ClientProfile, messageCount?: number) => void
  onThreadNotFound?: () => void
  agentConfig?: AgentConfig
  onAgentConfigChange?: (config: AgentConfig) => void
  isNewThread?: boolean
  customTitle?: string | null
  /** Pre-fill or auto-send a message. Use with autoSend to control behavior. */
  initialMessage?: string | null
  /** If true, initialMessage is sent immediately. If false, it just populates the input. */
  autoSend?: boolean
  /** Called after auto-send completes (use to clear URL params, etc.) */
  onInitialMessageSent?: () => void
}

interface QueuedMessage {
  content: string
  files: ImageAttachment[]
  userMessage: Message
}

export function ChatInterface({
  showToolCalls = false,
  threadId,
  onThreadUpdate,
  onThreadNotFound,
  initialMessage,
  customTitle,
  agentConfig,
  onAgentConfigChange,
  isNewThread = false,
  autoSend = false,
  onInitialMessageSent,
}: ChatInterfaceProps) {
  // ============================================================================
  // State Management
  // ============================================================================

  const [messages, setMessages] = useState<Message[]>([])

  // UI state with reducer
  const { state: uiState, dispatch: uiDispatch, setInput } = useChatState(threadId)

  // File upload state
  const {
    attachedFiles,
    uploadError,
    isDragging,
    handleFileSelect,
    handlePaste,
    handleDrop,
    handleDragOver,
    handleDragLeave,
    removeFile,
    clearFiles,
  } = useFileUpload()

  // Message queue for sending while AI is responding
  const messageQueueRef = useRef<QueuedMessage[]>([])
  const isProcessingQueueRef = useRef(false)
  const [queuedMessagesDisplay, setQueuedMessagesDisplay] = useState<{ content: string; id: string }[]>([])

  // Track the "base" input text (before voice input started + finalized transcripts)
  const baseInputRef = useRef(uiState.input)

  // Voice input - append transcribed text to current input
  const {
    isListening: isVoiceListening,
    isSupported: isVoiceSupported,
    error: voiceError,
    interimTranscript,
    toggleListening: handleVoiceToggle,
  } = useVoiceInput({
    onTranscript: (text) => {
      // When final transcript comes in, add it to the base and update input
      const newBase = baseInputRef.current ? `${baseInputRef.current} ${text}` : text
      baseInputRef.current = newBase
      setInput(newBase)
    },
  })

  // Update base input ref when user types manually (not from voice)
  useEffect(() => {
    // Only update base if we're not listening or interim hasn't changed
    // This prevents overwriting during voice input
    if (!isVoiceListening && !interimTranscript) {
      baseInputRef.current = uiState.input
    }
  }, [uiState.input, isVoiceListening, interimTranscript])

  // Combine base input with interim transcript for display
  const displayInput = isVoiceListening && interimTranscript
    ? (baseInputRef.current ? `${baseInputRef.current} ${interimTranscript}` : interimTranscript)
    : uiState.input

  // Custom toggle that captures current input as base when starting
  const toggleVoiceListening = useCallback(() => {
    if (!isVoiceListening) {
      // Starting - capture current input as base
      baseInputRef.current = uiState.input
    }
    handleVoiceToggle()
  }, [isVoiceListening, uiState.input, handleVoiceToggle])

  // ============================================================================
  // Refs
  // ============================================================================

  // Create a ref to control stream interruption
  const shouldInterruptRef = useRef(false)

  // File input ref for triggering file selection
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Textarea ref for auto-focus
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Track previous loading state to detect completion of AI response
  const prevIsLoadingRef = useRef(false)

  // ============================================================================
  // User Information
  // ============================================================================

  // Get user information for tracking in LangSmith
  const userId = useUserId()

  // Create stable client instance with user authentication
  // Recreate when userId changes to update auth headers
  const client = useMemo(() => {
    if (!userId) {
      // Don't create client until userId is available
      // This prevents creating a client without auth headers
      return null
    }

    const headers: Record<string, string> = {
      Authorization: `Bearer ${userId}`,
    }

    return new Client({
      apiUrl: LANGGRAPH_API_URL,
      apiKey: LANGSMITH_API_KEY,
      defaultHeaders: headers,
    })
  }, [userId])

  // Memoize user metadata to prevent unnecessary re-renders
  const userEmail = useMemo(
    () => userId || null,
    [userId]
  )
  const userName = useMemo(
    () => (userId ? `User ${userId.slice(0, 8)}` : null),
    [userId]
  )

  // ============================================================================
  // Custom Hooks
  // ============================================================================

  const { processStream } = useStreamHandler({
    client,
    threadId,
    setMessages,
    agentConfig,
    shouldInterruptRef,
    userId,
    userEmail,
    userName,
  })

  const {
    feedbackComment,
    showCommentInput,
    handleFeedback,
    handleSubmitComment,
    handleCancelComment,
    handleToggleComment,
    setFeedbackComment,
  } = useFeedback({
    messages,
    setMessages,
  })

  // ============================================================================
  // Effects
  // ============================================================================

  // Restore draft when switching threads
  useEffect(() => {
    if (typeof window === 'undefined') return

    // Check if there's an initial message (from ticket page, etc.)
    // If so, let that take precedence on first load
    if (initialMessage && !uiState.hasAutoSent) {
      return
    }

    const draft = localStorage.getItem(`draft-${threadId}`)
    if (draft) {
      setInput(draft)
    } else {
      // Clear input when switching to thread with no draft
      setInput('')
    }
  }, [threadId, initialMessage, uiState.hasAutoSent, setInput])

  // Track if we've sent a message on the current thread to skip unnecessary reloads
  const hasSentMessageRef = useRef<string | null>(null)

  // Load conversation history when threadId changes
  useEffect(() => {
    // Capture the current threadId to prevent race conditions
    const currentThreadId = threadId

    const loadThreadHistory = async () => {
      // Skip loading for new threads - they don't exist in backend yet
      if (isNewThread) {
        console.log('New thread detected - skipping backend load')
        setMessages([])
        uiDispatch({ type: 'SET_LOADING_THREAD', payload: false })
        return
      }

      // Skip reload if we just sent a message on this thread - client state is authoritative
      // This prevents race conditions where history reload overwrites trace URLs
      if (hasSentMessageRef.current === currentThreadId) {
        console.log('Skipping reload - we just sent a message on this thread')
        uiDispatch({ type: 'SET_LOADING_THREAD', payload: false })
        return
      }

      if (!LANGGRAPH_API_URL) {
        console.error("Missing NEXT_PUBLIC_LANGGRAPH_API_URL; cannot load thread history")
        uiDispatch({ type: 'SET_LOADING_THREAD', payload: false })
        return
      }

      // Wait for client to be ready (userId must be loaded first)
      if (!client) {
        console.log('Client not ready yet (waiting for userId)')
        uiDispatch({ type: 'SET_LOADING_THREAD', payload: false })
        return
      }

      try {
        console.log('Loading thread history for:', currentThreadId)
        const state = await client.threads.getState(currentThreadId).catch((err) => {
          // 403 means auth issue (shouldn't happen after our fixes, but handle gracefully)
          if (err?.response?.status === 403 || err?.status === 403) {
            console.error('Authorization error loading thread:', err)
            // Notify parent to navigate to most recent thread
            onThreadNotFound?.()
            return null
          }

          // 404 means thread doesn't exist OR user doesn't have access
          if (err?.response?.status === 404 || err?.status === 404) {
            console.log('Thread not found (404) - may not exist or access denied')
            // Notify parent to navigate to most recent thread
            onThreadNotFound?.()
            return null
          }

          // Other errors - log but continue with empty thread
          console.error('Error fetching thread state:', err)
          return null
        })

        if (!state) {
          // Thread doesn't exist or wasn't accessible - start fresh
          console.log('No thread state found - starting with empty thread')
          setMessages([])
          uiDispatch({ type: 'SET_LOADING_THREAD', payload: false })
          return
        }

        const threadMessages = (state.values as any)?.messages || []
        if (threadMessages.length === 0) {
          // Thread exists but has no messages - clear messages
          console.log('Thread exists but has no messages')
          setMessages([])
          uiDispatch({ type: 'SET_LOADING_THREAD', payload: false })
          return
        }

        const convertedMessages: Message[] = threadMessages
          .filter((msg: any) => {
            const msgType = msg.type || msg.role
            return ["human", "user", "ai", "assistant"].includes(msgType)
          })
          .map((msg: any, idx: number) => {
            // Create a stable ID for historical messages
            const messageId = msg.id || `history-${threadId}-${idx}-${msg.content?.slice(0, 20)}`

            const role = (msg.type === "ai" || msg.role === "assistant") ? "assistant" : "user"

            return {
              id: messageId,
              role,
              content: extractTextFromContent(msg.content),
              timestamp: msg.created_at ? new Date(msg.created_at) : new Date(),
              toolCalls: msg.tool_calls,
              runId: msg.run_id,
              thinkingDuration: msg.response_metadata?.thinking_duration,
              usageMetadata: msg.usage_metadata,
              shareUrl: msg.response_metadata?.share_url,
              // Preserve images/attachments
              images: msg.images,
            }
          })
          .filter((msg: Message) => msg.content.trim().length > 0)
          .reduce((acc: Message[], msg: Message, idx: number, arr: Message[]) => {
            // For consecutive AI messages, only keep the LAST one in the group
            if (msg.role === "assistant") {
              const nextMsg = arr[idx + 1]
              if (nextMsg && nextMsg.role === "assistant") {
                return acc
              }
            }
            return [...acc, msg]
          }, [])

        console.log(`SUCCESS: Loaded ${convertedMessages.length} messages from thread history`)

        // Only set messages if we're still on the same thread (prevent race conditions)
        if (currentThreadId === threadId) {
          // Merge with existing messages to preserve client-side metadata (shareUrl, usageMetadata, thinkingDuration)
          // This prevents race conditions where generateShareLink() updates get overwritten by history reload
          setMessages(prev => {
            // Build lookup maps by message ID, runId, and content hash (for matching when IDs differ)
            // Client-generated IDs differ from backend IDs, and backend doesn't always have runId
            const existingById = new Map(
              prev.map(m => [m.id, { shareUrl: m.shareUrl, usageMetadata: m.usageMetadata, thinkingDuration: m.thinkingDuration, runId: m.runId }])
            )
            const existingByRunId = new Map(
              prev.filter(m => m.runId).map(m => [m.runId!, { shareUrl: m.shareUrl, usageMetadata: m.usageMetadata, thinkingDuration: m.thinkingDuration, runId: m.runId }])
            )
            // Match by role + content prefix (first 100 chars) as fallback when IDs don't match
            const existingByContent = new Map(
              prev.filter(m => m.role === 'assistant').map(m => [
                `${m.role}:${m.content.slice(0, 100)}`,
                { shareUrl: m.shareUrl, usageMetadata: m.usageMetadata, thinkingDuration: m.thinkingDuration, runId: m.runId }
              ])
            )

            return convertedMessages.map((msg, idx) => {
              // Try to find existing metadata by ID first, then by runId, then by content
              const existingByIdMatch = existingById.get(msg.id)
              const existingByRunIdMatch = msg.runId ? existingByRunId.get(msg.runId) : undefined
              const contentKey = `${msg.role}:${msg.content.slice(0, 100)}`
              const existingByContentMatch = msg.role === 'assistant' ? existingByContent.get(contentKey) : undefined
              const existing = existingByIdMatch || existingByRunIdMatch || existingByContentMatch

              if (existing) {
                return {
                  ...msg,
                  // Preserve runId from existing if backend doesn't have it
                  runId: msg.runId || existing.runId,
                  shareUrl: msg.shareUrl || existing.shareUrl,
                  usageMetadata: msg.usageMetadata || existing.usageMetadata,
                  thinkingDuration: msg.thinkingDuration || existing.thinkingDuration,
                }
              }
              return msg
            })
          })
          uiDispatch({ type: 'SET_LOADING_THREAD', payload: false })
        } else {
          console.log(`Discarding messages for ${currentThreadId} - now on ${threadId}`)
          return
        }

        // Fetch metadata from LangSmith for messages that have runIds but missing metadata
        // Use Promise.all to batch all updates and prevent race conditions
        const messagesNeedingMetadata = convertedMessages.filter(
          (msg): msg is Message & { runId: string } =>
            !!msg.runId && msg.role === 'assistant' && (!msg.usageMetadata || !msg.shareUrl)
        )

        if (messagesNeedingMetadata.length > 0) {
          const metadataUpdates = await Promise.all(
            messagesNeedingMetadata.map(async (msg) => {
              try {
                console.log(`Fetching metadata from LangSmith for message ${msg.id} with runId ${msg.runId}`)

                let usageMetadata = msg.usageMetadata
                let thinkingDuration = msg.thinkingDuration
                let shareUrl = msg.shareUrl

                if (!msg.usageMetadata) {
                  const run = await readRun(msg.runId)

                  if (run) {
                    usageMetadata = {
                      input_tokens: run.prompt_tokens || 0,
                      output_tokens: run.completion_tokens || 0,
                      total_tokens: run.total_tokens || 0,
                      input_cost: (run as any).prompt_cost || 0,
                      output_cost: (run as any).completion_cost || 0,
                      total_cost: (run as any).total_cost || 0,
                    }

                    thinkingDuration = run.end_time && run.start_time
                      ? new Date(run.end_time).getTime() - new Date(run.start_time).getTime()
                      : undefined
                  }
                }

                if (!msg.shareUrl) {
                  try {
                    shareUrl = await shareRun(msg.runId)
                  } catch {
                    // Share URL might not exist yet, that's ok
                  }
                }

                if (!usageMetadata && !thinkingDuration && !shareUrl) {
                  return null
                }

                return {
                  messageId: msg.id,
                  usageMetadata,
                  thinkingDuration,
                  shareUrl,
                }
              } catch (error) {
                console.log(`Could not fetch metadata for run ${msg.runId}:`, error)
                return null
              }
            })
          )

          // Apply all metadata updates in a single state update
          // Check thread ID again to prevent race conditions
          if (currentThreadId === threadId) {
            const validUpdates = metadataUpdates.filter((u): u is NonNullable<typeof u> => u !== null)
            if (validUpdates.length > 0) {
              setMessages(prev => prev.map(m => {
                const update = validUpdates.find(u => u.messageId === m.id)
                return update
                  ? {
                      ...m,
                      usageMetadata: update.usageMetadata || m.usageMetadata,
                      thinkingDuration: update.thinkingDuration || m.thinkingDuration,
                      shareUrl: update.shareUrl || m.shareUrl,
                    }
                  : m
              }))
            }
          }
        }
      } catch (error) {
        console.error("Unexpected error loading thread history:", error)
        uiDispatch({ type: 'SET_LOADING_THREAD', payload: false })
      }
    }

    // Start loading state
    console.log('Thread ID changed to:', threadId)
    uiDispatch({ type: 'SET_LOADING_THREAD', payload: true })

    // Clear the "sent message" flag if we're switching to a completely different thread
    // (but keep it if it's the same thread - that's the case we want to skip reload)
    if (hasSentMessageRef.current && hasSentMessageRef.current !== threadId) {
      hasSentMessageRef.current = null
    }

    // Load new thread immediately
    loadThreadHistory()
  }, [threadId, client, uiDispatch, isNewThread])


  // Auto-focus textarea when loading completes and userId is available
  useEffect(() => {
    if (!uiState.isLoadingThread && userId && textareaRef.current) {
      // Small delay to ensure DOM is ready
      const timeoutId = setTimeout(() => {
        textareaRef.current?.focus()
      }, 100)
      return () => clearTimeout(timeoutId)
    }
  }, [uiState.isLoadingThread, userId])

  // Auto-focus textarea after AI finishes responding
  useEffect(() => {
    // Detect transition from loading (true) to not loading (false)
    const wasLoading = prevIsLoadingRef.current
    const isCurrentlyLoading = uiState.isLoading || uiState.isRegenerating

    // Update the ref for next render
    prevIsLoadingRef.current = isCurrentlyLoading

    // Focus only when transitioning from loading to not loading
    if (wasLoading && !isCurrentlyLoading && userId && textareaRef.current && messages.length > 0) {
      // Small delay to ensure DOM is ready and smooth transition
      const timeoutId = setTimeout(() => {
        textareaRef.current?.focus()
      }, 100)
      return () => clearTimeout(timeoutId)
    }
  }, [uiState.isLoading, uiState.isRegenerating, userId, messages.length])

  // ============================================================================
  // Event Handlers
  // ============================================================================

  // Process a single message (used for both immediate send and queue processing)
  const processMessage = useCallback(async (content: string, files: ImageAttachment[], userMessage: Message) => {
    uiDispatch({ type: 'START_SEND' })
    shouldInterruptRef.current = false
    hasSentMessageRef.current = threadId

    try {
      const assistantMessageId = generateMessageId()
      const { assistantContent } = await processStream(content, assistantMessageId, files)

      if (onThreadUpdate && assistantContent) {
        const firstUserMsg = messages.find((m) => m.role === "user") || userMessage
        const title = customTitle || truncate(firstUserMsg.content, 60) || "New conversation"
        const messageCount = messages.length + 2
        onThreadUpdate(threadId, title, truncate(assistantContent, 100), undefined, messageCount)
      }
    } catch (error) {
      console.error("Error streaming from LangGraph:", error)
      const errorMessage = createUserMessage(`Error: ${error instanceof Error ? error.message : "Failed to connect to the agent"}`)
      errorMessage.role = "assistant"
      setMessages((prev) => [...prev, errorMessage])

      if (onThreadUpdate) {
        const messageCount = messages.length + 2
        onThreadUpdate(threadId, customTitle || truncate(userMessage.content, 60) || "New conversation", truncate(errorMessage.content, 100), undefined, messageCount)
      }
    } finally {
      uiDispatch({ type: 'FINISH_SEND' })
    }
  }, [threadId, onThreadUpdate, processStream, messages, customTitle, uiDispatch])

  // Process queued messages one by one
  const processQueue = useCallback(async () => {
    if (isProcessingQueueRef.current || messageQueueRef.current.length === 0) return

    isProcessingQueueRef.current = true
    const nextMessage = messageQueueRef.current.shift()!

    // Remove from queue display and add to chat
    setQueuedMessagesDisplay(prev => prev.filter(m => m.id !== nextMessage.userMessage.id))
    setMessages((prev) => [...prev, nextMessage.userMessage])

    await processMessage(nextMessage.content, nextMessage.files, nextMessage.userMessage)

    isProcessingQueueRef.current = false

    // Process next in queue if any
    if (messageQueueRef.current.length > 0) {
      processQueue()
    }
  }, [processMessage])

  // Process queue when AI finishes responding
  useEffect(() => {
    const wasLoading = prevIsLoadingRef.current
    const isCurrentlyLoading = uiState.isLoading || uiState.isRegenerating

    // When loading finishes and there are queued messages, process them
    if (wasLoading && !isCurrentlyLoading && messageQueueRef.current.length > 0) {
      processQueue()
    }
  }, [uiState.isLoading, uiState.isRegenerating, processQueue])

  // Auto-send initial message (for ?q= URL param)
  useEffect(() => {
    const trimmedMessage = initialMessage?.trim()
    if (!trimmedMessage || uiState.hasAutoSent || uiState.isLoadingThread || !userId || !client) {
      return
    }

    uiDispatch({ type: 'SET_AUTO_SENT', payload: true })

    if (autoSend) {
      const userMessage = createUserMessage(trimmedMessage)
      setMessages((prev) => [...prev, userMessage])
      processMessage(trimmedMessage, [], userMessage)
        .then(() => onInitialMessageSent?.())
        .catch((error) => {
          console.error('Failed to auto-send initial message:', error)
          onInitialMessageSent?.() // Clear URL param even on error to prevent retry loops
        })
    } else {
      // Just populate input (existing behavior for ticket page, etc.)
      setInput(trimmedMessage)
    }
  }, [initialMessage, autoSend, uiState.hasAutoSent, uiState.isLoadingThread, userId, client, setInput, uiDispatch, processMessage, onInitialMessageSent])

  const handleSend = useCallback(async () => {
    if (!uiState.input.trim() && attachedFiles.length === 0) {
      return
    }

    if (!userId || !client) {
      return
    }

    const userMessage = createUserMessage(uiState.input)
    if (attachedFiles.length > 0) {
      userMessage.images = attachedFiles
    }

    const currentInput = uiState.input
    const currentFiles = [...attachedFiles]

    // Clear input and files immediately
    setInput("")
    clearFiles()

    // If currently loading, queue the message (don't show in chat yet)
    if (uiState.isLoading || uiState.isRegenerating) {
      const queuedItem = {
        content: currentInput,
        files: currentFiles,
        userMessage,
      }
      messageQueueRef.current.push(queuedItem)
      setQueuedMessagesDisplay(prev => [...prev, { content: currentInput, id: userMessage.id }])
      return
    }

    // Show message in chat and process immediately
    setMessages((prev) => [...prev, userMessage])
    await processMessage(currentInput, currentFiles, userMessage)

    // Check if anything was queued while processing
    if (messageQueueRef.current.length > 0) {
      processQueue()
    }
  }, [uiState.input, uiState.isLoading, uiState.isRegenerating, attachedFiles, userId, client, setInput, clearFiles, processMessage, processQueue])

  const handleStop = useCallback(async () => {
    console.log('User requested stop')
    uiDispatch({ type: 'SET_STOPPING', payload: true })
    shouldInterruptRef.current = true
  }, [uiDispatch])

  const handleRegenerate = useCallback(async () => {
    if (uiState.isLoading || uiState.isRegenerating) return

    const lastUserMessage = [...messages].reverse().find((m) => m.role === "user")
    if (!lastUserMessage) return

    const messagesUpToLastUser = messages.slice(0, messages.findIndex((m) => m.id === lastUserMessage.id) + 1)
    setMessages(messagesUpToLastUser)
    uiDispatch({ type: 'START_REGENERATE' })
    shouldInterruptRef.current = false

    try {
      const assistantMessageId = generateMessageId()
      const { assistantContent } = await processStream(lastUserMessage.content, assistantMessageId)

      if (onThreadUpdate && assistantContent) {
        const firstUserMsg = messagesUpToLastUser.find((m) => m.role === "user")
        const title = customTitle || (firstUserMsg ? truncate(firstUserMsg.content, 60) : "New conversation")
        const messageCount = messagesUpToLastUser.length + 1
        onThreadUpdate(threadId, title, truncate(assistantContent, 100), undefined, messageCount)
      }
    } catch (error) {
      console.error("Error regenerating:", error)
      const errorMessage = createUserMessage(`Error: ${error instanceof Error ? error.message : "Failed to regenerate response"}`)
      errorMessage.role = "assistant"
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      uiDispatch({ type: 'FINISH_REGENERATE' })
    }
  }, [uiState.isLoading, uiState.isRegenerating, messages, processStream, onThreadUpdate, threadId, uiDispatch])

  const handleEditAndRerun = useCallback(async (messageId: string, newContent: string) => {
    console.log('Edit and rerun from message:', messageId, 'new content:', newContent.slice(0, 50))

    if (uiState.isLoading || uiState.isRegenerating) return

    const messageIndex = messages.findIndex((m) => m.id === messageId)
    if (messageIndex === -1) return

    const messagesUpToEdit = messages.slice(0, messageIndex)
    const updatedMessage = {
      ...messages[messageIndex],
      content: newContent,
    }

    setMessages([...messagesUpToEdit, updatedMessage])
    uiDispatch({ type: 'SET_LOADING', payload: true })
    shouldInterruptRef.current = false

    try {
      const assistantMessageId = generateMessageId()
      console.log('Rerunning from edited message with assistantMessageId:', assistantMessageId)
      const { assistantContent } = await processStream(newContent, assistantMessageId)

      if (onThreadUpdate && assistantContent) {
        const firstUserMsg = messagesUpToEdit.find((m) => m.role === "user") || updatedMessage
        const title = customTitle || truncate(firstUserMsg.content, 60) || "New conversation"
        const messageCount = messagesUpToEdit.length + 2
        onThreadUpdate(threadId, title, truncate(assistantContent, 100), undefined, messageCount)
      }
    } catch (error) {
      console.error("Error rerunning from edit:", error)
      const errorMessage = createUserMessage(`Error: ${error instanceof Error ? error.message : "Failed to rerun from edit"}`)
      errorMessage.role = "assistant"
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      uiDispatch({ type: 'SET_LOADING', payload: false })
      uiDispatch({ type: 'SET_STOPPING', payload: false })
    }
  }, [uiState.isLoading, uiState.isRegenerating, messages, processStream, onThreadUpdate, threadId, uiDispatch])

  const handleCopy = async (content: string, messageId: string) => {
    await navigator.clipboard.writeText(content)
    uiDispatch({ type: 'SET_COPIED_ID', payload: messageId })
    setTimeout(() => uiDispatch({ type: 'SET_COPIED_ID', payload: null }), 2000)
  }

  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      if (userId) {
        handleSend()
      }
    }
  }, [userId, handleSend])

  const handleFileButtonClick = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (fileInputRef.current) {
      fileInputRef.current.click()
    }
  }, [])

  // ============================================================================
  // Computed Values
  // ============================================================================

  // Check if this is a new chat (no messages yet)
  const isNewChat = messages.length === 0 && !uiState.isLoadingThread

  // ============================================================================
  // Render
  // ============================================================================

  return (
    <>
      <style>{scrollbarStyles}</style>
      <main className="flex-1 flex flex-col overflow-hidden relative">
        <MessageList
          messages={messages}
          showToolCalls={showToolCalls}
          isRegenerating={uiState.isRegenerating}
          copiedId={uiState.copiedId}
          onCopy={handleCopy}
          onRegenerate={handleRegenerate}
          onEditAndRerun={handleEditAndRerun}
          feedbackComment={feedbackComment}
          showCommentInput={showCommentInput}
          onFeedback={handleFeedback}
          onSubmitComment={handleSubmitComment}
          onCancelComment={handleCancelComment}
          onToggleComment={handleToggleComment}
          setFeedbackComment={setFeedbackComment}
        />

        {isNewChat ? (
          <WelcomeScreen
            input={displayInput}
            onInputChange={setInput}
            onSend={handleSend}
            onKeyDown={handleKeyDown}
            isLoading={uiState.isLoading}
            isStopping={uiState.isStopping}
            onStop={handleStop}
            userId={userId}
            attachedFiles={attachedFiles}
            uploadError={uploadError}
            isDragging={isDragging}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onPaste={handlePaste}
            onRemoveFile={removeFile}
            onFileButtonClick={handleFileButtonClick}
            fileInputRef={fileInputRef}
            onFileSelect={handleFileSelect}
            textareaRef={textareaRef}
            isVoiceListening={isVoiceListening}
            isVoiceSupported={isVoiceSupported}
            onVoiceToggle={toggleVoiceListening}
            voiceError={voiceError}
            agentConfig={agentConfig}
            onAgentConfigChange={onAgentConfigChange}
          />
        ) : (
          <ChatInput
            input={displayInput}
            onInputChange={setInput}
            onSend={handleSend}
            onKeyDown={handleKeyDown}
            isLoading={uiState.isLoading}
            isStopping={uiState.isStopping}
            onStop={handleStop}
            userId={userId}
            attachedFiles={attachedFiles}
            uploadError={uploadError}
            isDragging={isDragging}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onPaste={handlePaste}
            onRemoveFile={removeFile}
            onFileButtonClick={handleFileButtonClick}
            fileInputRef={fileInputRef}
            onFileSelect={handleFileSelect}
            textareaRef={textareaRef}
            isVoiceListening={isVoiceListening}
            isVoiceSupported={isVoiceSupported}
            onVoiceToggle={toggleVoiceListening}
            voiceError={voiceError}
            queuedMessages={queuedMessagesDisplay}
          />
        )}
      </main>
    </>
  )
}
