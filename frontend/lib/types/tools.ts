/**
 * Tool-Related Types
 *
 * Type definitions for tool calls and subgraph execution.
 */

/**
 * Represents a tool call made by the AI assistant.
 */
export interface ToolCall {
  id: string
  name: string
  args: Record<string, any>
  output?: any
}

/**
 * Represents the output of a subagent/subgraph execution.
 * Used to display parallel task execution in the UI.
 */
export interface SubgraphOutput {
  name: string
  output: string
  timestamp: number
  toolCallId?: string
  isStreaming?: boolean
  isComplete?: boolean
}

