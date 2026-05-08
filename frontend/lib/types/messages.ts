/**
 * Message Types
 *
 * Type definitions for chat messages and related structures.
 */

import type { ToolCall } from "./tools"
import type { SubgraphOutput } from "./tools"
import type { UsageMetadata } from "./metadata"
import type { ImageAttachment } from "./images"

/**
 * Represents a chat message from either user or assistant.
 * Contains metadata for streaming, tool calls, feedback, and tracing.
 */
export interface Message {
  // Core properties
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: Date

  // Image attachments
  images?: ImageAttachment[]

  // Tool execution
  toolCalls?: ToolCall[]
  subgraphOutputs?: SubgraphOutput[]

  // Thinking/streaming state
  isThinking?: boolean
  thinkingSteps?: string[]
  thinkingStartTime?: number
  thinkingDuration?: number

  // LangSmith tracing
  runId?: string
  shareUrl?: string // Public LangSmith trace share URL
  usageMetadata?: UsageMetadata

  // User feedback
  feedback?: "positive" | "negative" | null
  feedbackId?: string
  feedbackComment?: string

  // Interruption tracking
  wasInterrupted?: boolean
}

