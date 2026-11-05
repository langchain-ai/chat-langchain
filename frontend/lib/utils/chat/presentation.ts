import type { Message } from "../../types/messages.ts"
import type { SubgraphOutput, ToolCall } from "../../types/tools.ts"

/** Keep application payloads and lifecycle decisions outside Macaw's renderers. */
export function messageStatus(message: Message) {
  if (message.status === "failed") return "failed"
  if (message.wasInterrupted || message.status === "interrupted") return "interrupted"
  if (message.isThinking || message.status === "streaming") return "streaming"
  return "completed"
}

export function toolStatus(tool: ToolCall, message: Message) {
  if (tool.status === "succeeded" || tool.status === "failed" || tool.status === "canceled") {
    return tool.status
  }
  if (messageStatus(message) === "interrupted" || messageStatus(message) === "failed") return "canceled"
  // Falsy results are still results. An absent result never implies success.
  if (Object.prototype.hasOwnProperty.call(tool, "output") && tool.output !== undefined) return "succeeded"
  return messageStatus(message) === "streaming" ? "running" : "pending"
}

export function subgraphStatus(subgraph: SubgraphOutput, message: Message) {
  if (subgraph.status === "failed" || subgraph.status === "canceled") return subgraph.status
  if (subgraph.isComplete || subgraph.status === "succeeded") return "succeeded"
  if (messageStatus(message) === "interrupted" || messageStatus(message) === "failed") return "canceled"
  return subgraph.isStreaming && messageStatus(message) === "streaming" ? "running" : "pending"
}

/** Later agent updates must not erase a call's result, state, or position. */
export function mergeToolCalls(current: ToolCall[], incoming: ToolCall[]): ToolCall[] {
  const next = current.map((tool) => ({ ...tool }))
  for (const tool of incoming) {
    const index = next.findIndex((existing) => existing.id === tool.id)
    if (index < 0) next.push({ ...tool, status: tool.status ?? "running" })
    else next[index] = { ...next[index], name: tool.name, args: tool.args }
  }
  return next
}
