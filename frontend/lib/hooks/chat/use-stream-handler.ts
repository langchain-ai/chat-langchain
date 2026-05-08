/**
 * Stream Handler Hook
 *
 * This hook manages the streaming of LangGraph agent responses, handling real-time
 * message updates, tool calls, thinking steps, subgraph outputs, and metadata fetching.
 *
 * Key Features:
 * - Real-time streaming of agent responses with progressive token display
 * - Tool call tracking and output capture (both regular tools and subagents)
 * - Thinking step visualization showing agent's reasoning process
 * - Subagent execution tracking with parallel execution support
 * - Usage metadata fetching from LangSmith (tokens and costs)
 * - Public share link generation with retry logic
 * - Stream interruption support
 * - SSR-safe implementation
 *
 * Architecture:
 * - Processes multiple stream modes: values, updates, messages, messages/partial
 * - Filters subgraph events to show only main agent responses
 * - Tracks execution state across streaming events
 * - Implements retry logic for LangSmith API calls (run data may not be immediately available)
 */

import { useCallback } from "react"
import { Client } from "@langchain/langgraph-sdk"
import type {
  Message,
  ToolCall,
  SubgraphOutput,
  UsageMetadata,
  ImageAttachment,
} from "../../types"
import {
  extractTextFromContent,
  ensureMessageExists,
  updateMessageInList,
} from "../../utils/chat"
import type { AgentConfig } from "@/components/layout/agent-settings"
import { shareRun, readRun } from "../../api/langsmith"
import { getModelProvider, getDefaultModel, type ModelOption } from "../../config/deployment-config"

// ============================================================================
// Constants
// ============================================================================

import { LANGGRAPH_API_URL } from "../../constants/api"

/** Retry configuration for LangSmith API calls */
const RETRY_CONFIG = {
  maxRetries: 8, // Increased for slower LangSmith indexing
  baseDelay: 1000, // 1 second initial delay
  initialDelay: 1000, // Wait before first attempt to let server settle
}

/** Error patterns that indicate a retryable failure */
const RETRYABLE_ERROR_PATTERNS = [
  "404",
  "Run not found",
  "Failed to fetch",
  "NetworkError",
  "network",
  "ECONNREFUSED",
  "ETIMEDOUT",
]

/**
 * Check if an error message indicates a retryable failure.
 * Retryable errors include: 404 (run not ingested yet), network failures
 */
function isRetryableError(errorMessage: string): boolean {
  return RETRYABLE_ERROR_PATTERNS.some(pattern => errorMessage.includes(pattern))
}

/**
 * Calculate exponential backoff delay for retry attempt.
 * Returns delays of: 1s, 2s, 4s, 8s, 16s for attempts 0-4
 */
function getBackoffDelay(attempt: number): number {
  return RETRY_CONFIG.baseDelay * Math.pow(2, attempt)
}

// ============================================================================
// Types
// ============================================================================

/**
 * Props for the useStreamHandler hook.
 */
interface UseStreamHandlerProps {
  client: Client | null
  threadId: string
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>
  agentConfig?: AgentConfig
  shouldInterruptRef?: React.MutableRefObject<boolean>
  userId?: string | null
  userEmail?: string | null
  userName?: string | null
}

/**
 * Return type for the useStreamHandler hook.
 */
interface UseStreamHandlerReturn {
  processStream: (
    userContent: string,
    assistantMessageId: string,
    images?: ImageAttachment[]
  ) => Promise<{ assistantContent: string; runId: string | undefined }>
}

// ============================================================================
// Hook Implementation
// ============================================================================

/**
 * Hook to handle streaming responses from LangGraph agents.
 *
 * Manages the complete lifecycle of agent streaming including:
 * - Processing streamed chunks and updating message state
 * - Tracking tool calls and their outputs
 * - Visualizing thinking steps and subagent execution
 * - Fetching usage metadata and generating share links
 *
 * @param client - LangGraph SDK client instance
 * @param threadId - ID of the conversation thread
 * @param setMessages - State setter for messages array
 * @param agentConfig - Optional agent configuration (model, recursion limit, agent type)
 * @param shouldInterruptRef - Optional ref to signal stream interruption
 * @returns Object containing processStream function
 *
 * @example
 * ```tsx
 * const { processStream } = useStreamHandler({
 *   client: langGraphClient,
 *   threadId: "thread-123",
 *   setMessages: setMessages,
 *   agentConfig: { model: "google_genai:gemini-3.1-flash-lite-preview", agentType: "docs_agent" }
 * })
 *
 * await processStream("What is LangChain?", "msg-456")
 * ```
 */
export function useStreamHandler({
  client,
  threadId,
  setMessages,
  agentConfig,
  shouldInterruptRef,
  userId,
  userEmail,
  userName,
}: UseStreamHandlerProps): UseStreamHandlerReturn {
  /**
   * Generates a public LangSmith trace URL.
   *
   * Implements light retry logic because run metadata may not be immediately available.
   *
   * @param runId - LangSmith run ID
   * @param messageId - Message ID to update with trace URL
   */
  const generateShareLink = useCallback(
    async (runId: string, messageId: string) => {
      console.log("[TraceURL] Generating trace URL for runId:", runId, "messageId:", messageId)

      try {
        // Initial delay to let the server settle after stream completion
        console.log("[TraceURL] Waiting 1s before first attempt...")
        await new Promise((resolve) => setTimeout(resolve, RETRY_CONFIG.initialDelay))

        for (let attempt = 0; attempt < RETRY_CONFIG.maxRetries; attempt++) {
          if (attempt > 0) {
            const delay = getBackoffDelay(attempt)
            console.log(`[TraceURL] Retry attempt ${attempt + 1}/${RETRY_CONFIG.maxRetries}, waiting ${delay}ms...`)
            await new Promise((resolve) => setTimeout(resolve, delay))
          }

          try {
            console.log("[TraceURL] Calling shareRun API...")
            const shareUrl = await shareRun(runId)

            if (shareUrl) {
              console.log("[TraceURL] SUCCESS! Trace URL:", shareUrl)
              setMessages((prev) =>
                prev.map((m) => (m.id === messageId ? { ...m, shareUrl } : m))
              )
              return
            }
          } catch (error: any) {
            const errorMessage = error?.message || ""
            console.log("[TraceURL] Error on attempt", attempt + 1, ":", errorMessage)

            if (isRetryableError(errorMessage) && attempt < RETRY_CONFIG.maxRetries - 1) {
              console.log("[TraceURL] Retryable error, will retry...")
              continue
            }

            throw error
          }
        }
      } catch (error: any) {
        const errorMessage = error?.message || ""
        // Only log unexpected errors (retryable errors are expected during normal operation)
        if (!isRetryableError(errorMessage)) {
          console.error("[TraceURL] Error generating trace URL:", error)
        }
      }
    },
    [setMessages]
  )

  /**
   * Fetches usage metadata (tokens and costs) from LangSmith.
   * Implements exponential backoff retry logic as run data may not be immediately available.
   *
   * @param runId - LangSmith run ID
   * @param messageId - Message ID to update with usage metadata
   */
  const fetchUsageMetadata = useCallback(
    async (runId: string, messageId: string) => {
      console.log("[UsageMetadata] Starting fetch for runId:", runId)
      try {
        // Initial delay to let the server settle after stream completion
        console.log("[UsageMetadata] Waiting 1s before first attempt...")
        await new Promise((resolve) => setTimeout(resolve, RETRY_CONFIG.initialDelay))

        for (let attempt = 0; attempt < RETRY_CONFIG.maxRetries; attempt++) {
          if (attempt > 0) {
            const delay = getBackoffDelay(attempt)
            console.log(`[UsageMetadata] Retry attempt ${attempt + 1}/${RETRY_CONFIG.maxRetries}, waiting ${delay}ms...`)
            await new Promise((resolve) => setTimeout(resolve, delay))
          }

          try {
            console.log("[UsageMetadata] Calling readRun API...")
            const run = await readRun(runId)

            if (run) {
              const totalTokens = run.total_tokens || 0
              console.log("[UsageMetadata] Got run, total_tokens:", totalTokens)

              // Only update if we have valid token data
              if (totalTokens > 0) {
                const usageMetadata: UsageMetadata = {
                  input_tokens: run.prompt_tokens || 0,
                  output_tokens: run.completion_tokens || 0,
                  total_tokens: totalTokens,
                  input_cost: (run as any).prompt_cost != null ? Number((run as any).prompt_cost) : 0,
                  output_cost: (run as any).completion_cost != null ? Number((run as any).completion_cost) : 0,
                  total_cost: (run as any).total_cost != null ? Number((run as any).total_cost) : 0,
                }

                console.log("[UsageMetadata] SUCCESS! Setting usage metadata:", usageMetadata)
                setMessages((prev) =>
                  prev.map((m) => (m.id === messageId ? { ...m, usageMetadata } : m))
                )
                return
              } else {
                console.log("[UsageMetadata] Run has no token data yet, retrying...")
              }
            }
          } catch (error: any) {
            const errorMessage = error?.message || ""
            console.log("[UsageMetadata] Error on attempt", attempt + 1, ":", errorMessage)

            if (isRetryableError(errorMessage) && attempt < RETRY_CONFIG.maxRetries - 1) {
              console.log("[UsageMetadata] Retryable error, will retry...")
              continue
            }

            // Only warn for unexpected errors
            if (!isRetryableError(errorMessage)) {
              console.warn("Unable to fetch usage metadata from LangSmith:", errorMessage)
            }
            return
          }
        }
        console.log("[UsageMetadata] Exhausted all retries without success")
      } catch (error: any) {
        const errorMessage = error?.message || ""
        if (!isRetryableError(errorMessage)) {
          console.error("Failed to fetch usage metadata:", error)
        }
      }
    },
    [setMessages]
  )

  /**
   * Processes the stream of agent responses.
   *
   * Main function that handles:
   * - Initiating the stream with LangGraph SDK
   * - Processing various stream event types (values, updates, messages, messages/partial)
   * - Tracking tool calls, thinking steps, and subagent outputs
   * - Updating message state in real-time
   * - Handling stream interruption
   * - Triggering metadata and share link fetching after completion
   *
   * @param userContent - User's message content
   * @param assistantMessageId - ID for the assistant's response message
   * @param images - Optional image attachments
   * @returns Promise with assistant content and run ID
   */
  const processStream = useCallback(
    async (userContent: string, assistantMessageId: string, images?: ImageAttachment[]) => {
      if (!LANGGRAPH_API_URL) {
        throw new Error(
          "Missing LANGGRAPH_API_URL; cannot invoke LangGraph"
        )
      }

      if (!client) {
        throw new Error(
          "Client not initialized; cannot invoke LangGraph. User ID may not be loaded yet."
        )
      }

      // Format message content - use multimodal format if files are present
      let messageContent: any
      if (images && images.length > 0) {
        // Build multimodal message with text and files
        const contentBlocks: any[] = [
          {
            type: "text",
            text: userContent || "Please analyze the attached file(s)."
          }
        ]

        // Process each file
        for (const file of images) {
          // Public CLC does not support HAR analysis; avoid sending large traces to the docs agent.
          if (file.name?.toLowerCase().endsWith(".har")) continue

          const isImage = file.mimeType?.startsWith('image/')

          if (isImage) {
            // Image files: send as base64 image_url
            contentBlocks.push({
              type: "image_url",
              image_url: {
                url: `data:${file.mimeType};base64,${file.base64}`
              }
            })
          } else {
            // Text files: decode base64 and send as text block
            try {
              // Decode base64 to get text content
              const decodedContent = atob(file.base64 || '')
              console.log(`📄 Decoded file ${file.name}:`, {
                mimeType: file.mimeType,
                size: file.size,
                contentLength: decodedContent.length,
                preview: decodedContent.slice(0, 100)
              })
              contentBlocks.push({
                type: "text",
                text: `**File: ${file.name || 'unknown'}**\n\`\`\`\n${decodedContent}\n\`\`\``
              })
            } catch (error) {
              console.error(`Failed to decode file ${file.name}:`, error)
              contentBlocks.push({
                type: "text",
                text: `[Failed to decode file: ${file.name}]`
              })
            }
          }
        }

        messageContent = contentBlocks
      } else {
        // Text-only message
        messageContent = userContent
      }

      // Log the final message being sent
      console.log('📤 Sending message to agent:', {
        hasFiles: images && images.length > 0,
        fileCount: images?.length || 0,
        contentBlocks: Array.isArray(messageContent) ? messageContent.length : 1,
        messagePreview: Array.isArray(messageContent)
          ? messageContent.map(block => `${block.type}: ${block.text?.slice(0, 50) || 'image'}...`)
          : messageContent.slice(0, 100)
      })

      const input = {
        messages: [{ role: "user", content: messageContent }],
      }

      const model = (agentConfig?.model ?? getDefaultModel()) as ModelOption
      const recursionLimit = agentConfig?.recursionLimit ?? 100
      const modelProvider = getModelProvider(model)

      let assistantContent = ""
      let assistantToolCalls: ToolCall[] = []
      let runId: string | undefined = undefined
      let hasSeenNewResponse = false

      const agentType = agentConfig?.agentType ?? "docs_agent"
      const repos = agentConfig?.repos ?? []

      // Trace metadata for LangSmith observability
      const traceMetadata = {
        user_id: userId || "unknown",
        ...(userEmail && userEmail !== userId ? { user_email: userEmail } : {}),
        ...(userName && !userName.startsWith("User") ? { user_name: userName } : {}),
        source_type: "Chat-LangChain",
        graph: agentType,
      }

      const streamResponse = client.runs.stream(threadId, agentType, {
        input,
        config: {
          recursion_limit: recursionLimit,
          tags: ["Chat-LangChain", agentType],
          metadata: traceMetadata,
          configurable: {
            model: model,
            model_provider: modelProvider,
            ...(repos.length > 0 && { repos }),
          },
        } as any,
        streamMode: ["values", "updates", "messages"],
        streamSubgraphs: true,
        ifNotExists: "create",
      })

      // Initialize from existing message data if resuming
      let existingMessage: Message | undefined
      setMessages((prev) => {
        existingMessage = prev.find((m) => m.id === assistantMessageId)
        return prev
      })

      const subgraphOutputs: SubgraphOutput[] = existingMessage?.subgraphOutputs
        ? [...existingMessage.subgraphOutputs]
        : []

      // Restore tool calls from existing message
      if (existingMessage?.toolCalls) {
        assistantToolCalls = [...existingMessage.toolCalls]
      }

      for await (const chunk of streamResponse) {
        // Check if user requested interrupt
        if (shouldInterruptRef?.current) {
          break
        }

        const eventType = chunk.event as string
        const data = chunk.data as any

        // Capture run_id from metadata
        if (!runId) {
          const possibleRunId =
            (chunk as any).metadata?.run_id ||
            (chunk as any).run_id ||
            (chunk as any).data?.run_id

          if (possibleRunId) {
            runId = possibleRunId
          }
        }

        const isSubgraphEvent = eventType.includes("|")
        const eventParts = eventType.split("|")
        const baseEvent = eventParts[0]

        // Track subgraph outputs when they complete or stream
        if (
          (eventType === "updates" ||
            (baseEvent === "updates" && isSubgraphEvent)) &&
          data
        ) {
          // Update tool calls from agent/model messages
          const agentMessages = data.agent?.messages || data.model?.messages
          if (agentMessages && Array.isArray(agentMessages)) {
            agentMessages.forEach((msg: any) => {
              if (msg.tool_calls && Array.isArray(msg.tool_calls)) {
                msg.tool_calls.forEach((toolCall: any) => {
                  const existingToolCall = assistantToolCalls.find(
                    (tc) => tc.id === toolCall.id
                  )
                  if (!existingToolCall) {
                    assistantToolCalls.push({
                      id: toolCall.id,
                      name: toolCall.name,
                      args: toolCall.args,
                    })
                  }

                  // Track task tool calls for subgraph outputs
                  if (toolCall.name === "task") {
                    const subagentName =
                      toolCall.args?.subagent_type || "subagent-task"
                    const existingOutput = subgraphOutputs.find(
                      (o) => o.toolCallId === toolCall.id
                    )

                    if (!existingOutput) {
                      subgraphOutputs.push({
                        name: subagentName,
                        output: "",
                        timestamp: Date.now(),
                        toolCallId: toolCall.id,
                        isStreaming: true,
                        isComplete: false,
                      })
                    }
                  }
                })
              }
            })
          }

          // Process tool messages (both regular tools and subagent responses)
          if (data.tools?.messages && Array.isArray(data.tools.messages)) {
            data.tools.messages.forEach((msg: any) => {
              if (msg.type === "tool" && msg.tool_call_id) {
                // Handle task tools (subagents) separately
                if (msg.name === "task" && msg.content) {
                  const existingOutput = subgraphOutputs.find(
                    (output) => output.toolCallId === msg.tool_call_id
                  )

                  if (existingOutput) {
                    existingOutput.output =
                      typeof msg.content === "string"
                        ? msg.content
                        : JSON.stringify(msg.content)
                    existingOutput.isStreaming = false
                    existingOutput.isComplete = true
                  } else {
                    const toolCall = assistantToolCalls.find(
                      (tc) => tc.id === msg.tool_call_id
                    )
                    const subagentName =
                      toolCall?.args?.subagent_type || "subagent-task"

                    const taskOutput = {
                      name: subagentName,
                      output:
                        typeof msg.content === "string"
                          ? msg.content
                          : JSON.stringify(msg.content),
                      timestamp: Date.now(),
                      toolCallId: msg.tool_call_id,
                      isStreaming: false,
                      isComplete: true,
                    }

                    subgraphOutputs.push(taskOutput)
                  }
                }
                // Handle regular tools (attach output to tool call)
                else {
                  const toolCall = assistantToolCalls.find(
                    (tc) => tc.id === msg.tool_call_id
                  )
                  if (toolCall && msg.content) {
                    toolCall.output =
                      typeof msg.content === "string"
                        ? msg.content
                        : JSON.stringify(msg.content)
                  }
                }
              }
            })
          }
        }

        setMessages((prev) => {
          const existing = prev.find((m) => m.id === assistantMessageId)
          const thinkingSteps = existing?.thinkingSteps || []
          const thinkingStartTime = existing?.thinkingStartTime || Date.now()
          let hasNewSteps = false
          let hasNewSubgraphOutputs = false

          if (subgraphOutputs.length > 0) {
            const existingOutputCount = existing?.subgraphOutputs?.length || 0
            if (subgraphOutputs.length > existingOutputCount) {
              hasNewSubgraphOutputs = true
            }
          }

          // Track node executions from updates (skip 'agent', 'model', 'tools', and middleware nodes)
          if ((eventType === "updates" || baseEvent === "updates") && data) {
            Object.keys(data).forEach((nodeName) => {
              if (
                nodeName === "agent" ||
                nodeName === "model" ||
                nodeName === "tools" ||
                nodeName.includes("Middleware")  // Skip all middleware nodes
              )
                return

              const stepDesc = `Node: ${nodeName}`
              const alreadyExists = thinkingSteps.some((s) => s === stepDesc)
              if (!alreadyExists) {
                thinkingSteps.push(stepDesc)
                hasNewSteps = true
              }
            })
          }

          // Check for AI thinking
          if (
            (eventType === "updates" || baseEvent === "updates") &&
            (data?.agent || data?.model) &&
            !data?.tools
          ) {
            const aiThinkingStep = "Planning next steps..."
            if (!thinkingSteps.some((s) => s === "Planning next steps...")) {
              thinkingSteps.push(aiThinkingStep)
              hasNewSteps = true
            }
          }

        // Detect subagent execution (single or parallel)
        const agentMessages = data?.agent?.messages || data?.model?.messages
        if (agentMessages && Array.isArray(agentMessages)) {
          agentMessages.forEach((msg: any) => {
            if (msg.tool_calls && Array.isArray(msg.tool_calls)) {
              const taskTools = msg.tool_calls.filter((tc: any) => tc.name === "task")

              // Show message for subagents (task tools)
              if (taskTools.length > 0) {
                const subagentNames = taskTools
                  .map((tc: any) => tc.args?.subagent_type || "subagent")
                  .join(", ")

                const parallelStep = taskTools.length > 1
                  ? `Calling ${taskTools.length} subagents in parallel: ${subagentNames}`
                  : `Calling subagent: ${subagentNames}`

                if (!thinkingSteps.includes(parallelStep)) {
                  thinkingSteps.push(parallelStep)
                  hasNewSteps = true
                }
              }
              // Skip regular tools - they'll be shown individually below
            }
          })
        }

        // Track tool executions with clear context
        if (data?.tools?.messages && Array.isArray(data.tools.messages)) {
          data.tools.messages.forEach((msg: any) => {
            if (msg.name && msg.name !== "task") {
              // Find the tool call to get args for context
              const toolCall = assistantToolCalls.find((tc: any) => tc.id === msg.tool_call_id)

              let stepDesc = msg.name

              // Add descriptive context from tool args
              if (toolCall?.args && Object.keys(toolCall.args).length > 0) {
                const args = toolCall.args

                // Codebase tools (old naming)
                if (args.file_path && msg.name === "read_codebase_file") {
                  const path = args.file_path.length > 50 ? "..." + args.file_path.slice(-50) : args.file_path
                  stepDesc = `Reading ${path}`
                } else if (args.pattern && msg.name === "search") {
                  const pattern = args.pattern.length > 40 ? args.pattern.substring(0, 40) + "..." : args.pattern
                  const location = args.path ? ` in ${args.path}` : ""
                  stepDesc = `Searching codebase for "${pattern}"${location}`
                } else if (args.path && msg.name === "list_directory") {
                  stepDesc = `Listing directory ${args.path || "root"}`
                }
                // Public code tools
                else if (args.file_path && msg.name === "read_public_file") {
                  const path = args.file_path.length > 50 ? "..." + args.file_path.slice(-50) : args.file_path
                  stepDesc = `Reading public file: ${path}`
                } else if (args.pattern && msg.name === "search_public_code") {
                  const pattern = args.pattern.length > 40 ? args.pattern.substring(0, 40) + "..." : args.pattern
                  const location = args.path ? ` in ${args.path}` : ""
                  stepDesc = `Searching public code for "${pattern}"${location}`
                } else if (args.path && msg.name === "list_public_directory") {
                  stepDesc = `Listing public directory: ${args.path || "root"}`
                }
                // Full access code tools
                else if (args.file_path && msg.name === "read_all_files") {
                  const path = args.file_path.length > 50 ? "..." + args.file_path.slice(-50) : args.file_path
                  stepDesc = `Reading file (public+private): ${path}`
                } else if (args.pattern && msg.name === "search_all_code") {
                  const pattern = args.pattern.length > 40 ? args.pattern.substring(0, 40) + "..." : args.pattern
                  const location = args.path ? ` in ${args.path}` : ""
                  stepDesc = `Searching all code for "${pattern}"${location}`
                } else if (args.path && msg.name === "list_all_directories") {
                  stepDesc = `Listing directory (public+private): ${args.path || "root"}`
                }
                // Docs tools
                else if (args.query && msg.name === "SearchDocsByLangChain") {
                  const query = args.query.length > 40 ? args.query.substring(0, 40) + "..." : args.query
                  stepDesc = `Searching documentation for "${query}"`
                }
                // Pylon tools
                else if (args.collections && msg.name === "search_support_articles") {
                  stepDesc = `Searching support articles (${args.collections})`
                } else if (args.article_id && msg.name === "get_article_content") {
                  stepDesc = `Fetching support article content`
                }
              }

              // Use tool_call_id to check if already tracked
              const alreadyExists = msg.tool_call_id
                ? thinkingSteps.some((s) => s.includes(msg.tool_call_id))
                : thinkingSteps.includes(stepDesc)

              if (!alreadyExists) {
                thinkingSteps.push(stepDesc)
                hasNewSteps = true
              }
            }
          })
        }

        // Always ensure message exists with thinking state
        if (!existing) {
          return [
            ...prev,
            {
              id: assistantMessageId,
              role: "assistant" as const,
              content: "",
              timestamp: new Date(),
              isThinking: true,
              thinkingSteps: [...thinkingSteps],
              thinkingStartTime,
              subgraphOutputs: subgraphOutputs.length > 0 ? [...subgraphOutputs] : [],
            },
          ]
        } else if (hasNewSteps || hasNewSubgraphOutputs) {
          return prev.map((m) =>
            m.id === assistantMessageId
              ? {
                  ...m,
                  isThinking: true,
                  thinkingSteps: [...thinkingSteps],
                  thinkingStartTime,
                  subgraphOutputs: [...subgraphOutputs],
                }
              : m
          )
        }
        return prev
      })

      // Handle "values" mode - final state with complete output
      // IMPORTANT: Skip subgraph events to avoid showing subagent content in main chat
      if (eventType === "values" && !isSubgraphEvent && data?.messages && Array.isArray(data.messages)) {
        // Find the last assistant message
        const finalAIMessage = [...data.messages].reverse().find((msg: any) =>
          msg.type === "ai" || msg.role === "assistant"
        )

        const finalContent = finalAIMessage?.content ? extractTextFromContent(finalAIMessage.content) : ""
        const hasFinalMessage = !finalAIMessage?.tool_calls || finalAIMessage.tool_calls.length === 0

        // IMPORTANT: Skip subagent responses (they typically start with JSON like '{"answer":')
        const looksLikeSubagentResponse = finalContent.trim().startsWith('{') || finalContent.trim().startsWith('{"answer')

        // Only set content if:
        // 1. We have final content
        // 2. No pending tool calls
        // 3. Haven't set content yet
        // 4. Not a subagent response
        // 5. We've seen NEW streaming content for this request (prevents using old thread history)
        if (finalContent && hasFinalMessage && !looksLikeSubagentResponse && hasSeenNewResponse && !assistantContent) {
          assistantContent = finalContent

          setMessages((prev) => {
            const baseMessage: Message = {
              id: assistantMessageId,
              role: "assistant",
              content: assistantContent,
              timestamp: new Date(),
              isThinking: true,
              subgraphOutputs: [...subgraphOutputs],
            }

            const withMessage = ensureMessageExists(prev, assistantMessageId, baseMessage)
            return updateMessageInList(withMessage, assistantMessageId, {
              content: assistantContent,
              isThinking: true,
              subgraphOutputs: [...subgraphOutputs],
            })
          })
        }
      }

      // Handle streaming messages - show progressive tokens
      // Try both "messages/partial" and "messages" event types
      // IMPORTANT: Skip subgraph events (they have "|" in the event type)
      if ((eventType === "messages/partial" || eventType === "messages") && !isSubgraphEvent && data) {
        // Handle both array and tuple formats
        let aiChunk: any
        if (Array.isArray(data)) {
          aiChunk = data.find((msg: any) => msg.type === "ai" || msg.role === "assistant")
        } else if (data && typeof data === 'object') {
          // Sometimes data is a tuple [message, metadata]
          if (data[0]) aiChunk = data[0]
          else aiChunk = data
        }

        if (aiChunk?.content) {
          const streamedContent = extractTextFromContent(aiChunk.content)

          // IMPORTANT: Skip subagent responses (they typically start with JSON like '{"answer":')
          const looksLikeSubagentResponse = streamedContent.trim().startsWith('{') || streamedContent.trim().startsWith('{"answer')
          if (looksLikeSubagentResponse) {
            continue
          }

          // Only update if we have content and no pending tool calls (check array length)
          const hasPendingToolCalls = aiChunk.tool_calls && Array.isArray(aiChunk.tool_calls) && aiChunk.tool_calls.length > 0
          if (streamedContent && !hasPendingToolCalls) {
            // Accumulate the streamed content
            assistantContent = streamedContent
            hasSeenNewResponse = true // Mark that we've seen new content

            setMessages((prev) => {
              const baseMessage: Message = {
                id: assistantMessageId,
                role: "assistant",
                content: streamedContent,
                timestamp: new Date(),
                isThinking: true,
                subgraphOutputs: [...subgraphOutputs],
              }

              const withMessage = ensureMessageExists(prev, assistantMessageId, baseMessage)
              return updateMessageInList(withMessage, assistantMessageId, {
                content: streamedContent,
                isThinking: true,
                subgraphOutputs: [...subgraphOutputs],
              })
            })
          }
        }
      }

      // Capture tool calls (NOT content - content comes from "values" event)
      // Support both agent (deepagent) and model (create_agent) nodes
      const agentMessages = data?.agent?.messages || data?.model?.messages
      if ((eventType === "updates" || baseEvent === "updates") && agentMessages && Array.isArray(agentMessages)) {
        agentMessages.forEach((msg: any) => {
          if (msg.type === "ai" && msg.tool_calls?.length > 0) {
            assistantToolCalls = msg.tool_calls
          }
        })

        if (assistantToolCalls.length > 0) {
          setMessages((prev) => {
            const baseMessage: Message = {
              id: assistantMessageId,
              role: "assistant",
              content: "",
              timestamp: new Date(),
              toolCalls: assistantToolCalls,
              isThinking: true,
              subgraphOutputs: [...subgraphOutputs],
            }

            const withMessage = ensureMessageExists(prev, assistantMessageId, baseMessage)
            return updateMessageInList(withMessage, assistantMessageId, {
              toolCalls: assistantToolCalls,
              isThinking: true,
              subgraphOutputs: [...subgraphOutputs],
            })
          })
        }
      }
    }

    // Check if stream was interrupted
    const wasInterrupted = shouldInterruptRef?.current || false

    // Mark as complete after stream ends
    setMessages((prev) => {
      const existing = prev.find((m) => m.id === assistantMessageId)
      const thinkingDuration = existing?.thinkingStartTime
        ? Date.now() - existing.thinkingStartTime
        : undefined

      return prev.map((m) =>
        m.id === assistantMessageId
          ? {
              ...m,
              content: wasInterrupted && !assistantContent
                ? "Response stopped. The agent was interrupted while processing your request."
                : assistantContent || "(No response generated)",
              isThinking: false,
              thinkingDuration,
              runId,
              subgraphOutputs: subgraphOutputs.length > 0 ? subgraphOutputs : undefined,
              wasInterrupted,
            }
          : m
      )
    })

    // Fetch usage metadata and generate public share link if we have a runId
    if (runId) {
      fetchUsageMetadata(runId, assistantMessageId)
      generateShareLink(runId, assistantMessageId)
    }

    return { assistantContent, runId }
  }, [client, threadId, setMessages, agentConfig, fetchUsageMetadata, generateShareLink, userId, userEmail, userName])

  return { processStream }
}
