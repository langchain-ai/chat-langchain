import type { Message } from "../../types/messages.ts"
import type { ImageAttachment } from "../../types/images.ts"
import type { SubgraphOutput, ToolCall } from "../../types/tools.ts"

type RecordValue = Record<string, unknown>
const record = (value: unknown): RecordValue => value !== null && typeof value === "object" && !Array.isArray(value) ? value as RecordValue : {}
const string = (value: unknown): string | undefined => typeof value === "string" ? value : undefined
const number = (value: unknown): number | undefined => typeof value === "number" && Number.isFinite(value) ? value : undefined
const strings = (value: unknown): string[] => Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : []
const has = (value: RecordValue, key: string) => Object.prototype.hasOwnProperty.call(value, key)

function timestamp(value: unknown): Date {
  const date = typeof value === "string" || typeof value === "number" ? new Date(value) : new Date(0)
  return Number.isNaN(date.getTime()) ? new Date(0) : date
}

function text(value: unknown): string {
  if (typeof value === "string") return value
  if (!Array.isArray(value)) return ""
  return value.flatMap((block) => {
    if (typeof block === "string") return [block]
    const item = record(block)
    return item.type === "text" && typeof item.text === "string" ? [item.text] : []
  }).join("\n\n")
}

function attachments(raw: RecordValue, id: string): ImageAttachment[] {
  const saved: ImageAttachment[] = Array.isArray(raw.images) ? raw.images.flatMap((value, index) => {
    const file = record(value)
    if (!Object.keys(file).length) return []
    return [{
      id: string(file.id) ?? `${id}-file-${index}`,
      mimeType: string(file.mimeType) ?? "application/octet-stream",
      name: string(file.name), url: string(file.url), base64: string(file.base64),
      size: number(file.size), textLength: number(file.textLength),
    }]
  }) : []
  if (Array.isArray(raw.content)) {
    raw.content.forEach((value, index) => {
      const block = record(value)
      if (block.type !== "image_url" && block.type !== "image") return
      const source = record(block.source)
      const url = string(block.image_url) ?? string(record(block.image_url).url) ?? string(block.url)
      const data = url?.match(/^data:(image\/[^;,]+);base64,([\s\S]+)$/)
      const base64 = data?.[2] ?? string(source.data) ?? string(block.data)
      const mimeType = data?.[1] ?? string(source.media_type) ?? string(block.mime_type) ?? "image/png"
      if (!url && !base64) return
      if (saved.some((file) => (url && file.url === url) || (base64 && file.base64 === base64))) return
      saved.push({ id: `${id}-image-${index}`, mimeType, url, base64, name: string(block.name) ?? "Attached image" })
    })
  }
  return saved
}

function toolCalls(value: unknown, messageId: string): ToolCall[] {
  if (!Array.isArray(value)) return []
  return value.flatMap((entry, index) => {
    const tool = record(entry)
    if (!Object.keys(tool).length) return []
    const status = tool.status === "pending" || tool.status === "running" || tool.status === "succeeded" || tool.status === "failed" || tool.status === "canceled" ? tool.status : undefined
    return [{
      id: string(tool.id) ?? `${messageId}-tool-${index}`,
      name: string(tool.name) ?? "Tool",
      args: record(tool.args),
      ...(status ? { status } : {}),
      ...(has(tool, "output") ? { output: tool.output } : {}),
      ...(has(tool, "error") ? { error: tool.error } : {}),
    }]
  })
}

function subgraphs(value: unknown, messageId: string): SubgraphOutput[] {
  if (!Array.isArray(value)) return []
  return value.flatMap((entry, index) => {
    const subgraph = record(entry)
    const name = string(subgraph.name)
    if (!name) return []
    const status = subgraph.status === "running" || subgraph.status === "succeeded" || subgraph.status === "failed" || subgraph.status === "canceled" ? subgraph.status : undefined
    return [{
      name,
      output: string(subgraph.output) ?? "",
      timestamp: number(subgraph.timestamp) ?? index,
      toolCallId: string(subgraph.toolCallId) ?? string(subgraph.tool_call_id) ?? `${messageId}-subagent-${index}`,
      ...(typeof subgraph.isStreaming === "boolean" ? { isStreaming: subgraph.isStreaming } : {}),
      ...(typeof subgraph.isComplete === "boolean" ? { isComplete: subgraph.isComplete } : {}),
      ...(status ? { status } : {}),
      ...(has(subgraph, "error") ? { error: subgraph.error } : {}),
    }]
  })
}

function mergeTools(previous: ToolCall[] = [], incoming: ToolCall[] = []): ToolCall[] {
  const merged = previous.map((tool) => ({ ...tool }))
  for (const tool of incoming) {
    const index = merged.findIndex((current) => current.id === tool.id)
    if (index < 0) merged.push(tool)
    else merged[index] = { ...merged[index], ...tool }
  }
  return merged
}

function mergeSubgraphs(previous: SubgraphOutput[] = [], incoming: SubgraphOutput[] = []): SubgraphOutput[] {
  const merged = previous.map((subgraph) => ({ ...subgraph }))
  for (const subgraph of incoming) {
    const index = merged.findIndex((current) => current.toolCallId === subgraph.toolCallId)
    if (index < 0) merged.push(subgraph)
    else merged[index] = { ...merged[index], ...subgraph }
  }
  return merged
}

/**
 * Restore one visible answer per assistant turn, retaining its preceding tool
 * calls and their explicit result events. Never infer success from missing data.
 * Backend IDs win; fallbacks depend only on the thread and persisted position.
 */
export function restoreThreadMessages(payload: unknown, threadId: string): Message[] {
  if (!Array.isArray(payload)) return []
  const messages: Message[] = []
  let assistant: Message | undefined
  const toolOwners = new Map<string, Message>()

  payload.forEach((value, index) => {
    const raw = record(value)
    const role = raw.type ?? raw.role
    if (role === "tool") {
      const toolCallId = string(raw.tool_call_id)
      const owner = toolCallId ? toolOwners.get(toolCallId) : undefined
      if (!toolCallId || !owner) return
      const tool = owner.toolCalls?.find((call) => call.id === toolCallId)
      if (!tool) return
      tool.status = raw.status === "error" || raw.status === "failed" ? "failed" : "succeeded"
      if (has(raw, "content")) tool.output = raw.content
      if (tool.status === "failed") tool.error = raw.content
      if (tool.name === "task" || raw.name === "task") {
        const output = typeof raw.content === "string" ? raw.content : JSON.stringify(raw.content) ?? ""
        owner.subgraphOutputs = mergeSubgraphs(owner.subgraphOutputs, [{
          name: string(tool.args.subagent_type) ?? string(raw.name) ?? "Subagent",
          toolCallId,
          output,
          timestamp: timestamp(raw.created_at).getTime(),
          isStreaming: false,
          isComplete: true,
          status: tool.status,
          ...(tool.status === "failed" ? { error: raw.content } : {}),
        }])
      }
      return
    }
    if (role !== "human" && role !== "user" && role !== "ai" && role !== "assistant") return
    const isAssistant = role === "ai" || role === "assistant"
    const id = string(raw.id) ?? `history-${threadId}-${index}`
    const metadata = record(raw.response_metadata)
    const status = raw.status === "failed" || raw.status === "interrupted" || raw.status === "streaming" || raw.status === "completed" ? raw.status : undefined
    const interruption = raw.wasInterrupted ?? raw.was_interrupted ?? metadata.was_interrupted
    const wasInterrupted = typeof interruption === "boolean" ? interruption : undefined
    const next: Message = {
      id,
      role: isAssistant ? "assistant" : "user",
      content: text(raw.content),
      timestamp: timestamp(raw.created_at ?? raw.timestamp),
      images: attachments(raw, id),
      toolCalls: toolCalls(raw.tool_calls ?? raw.toolCalls, id),
      subgraphOutputs: subgraphs(raw.subgraphOutputs ?? raw.subgraph_outputs ?? metadata.subgraph_outputs, id),
      thinkingSteps: strings(raw.thinkingSteps ?? raw.thinking_steps ?? metadata.thinking_steps),
      thinkingDuration: number(raw.thinkingDuration ?? metadata.thinking_duration),
      runId: string(raw.run_id ?? raw.runId),
      shareUrl: string(raw.shareUrl ?? metadata.share_url),
      usageMetadata: Object.keys(record(raw.usage_metadata ?? raw.usageMetadata)).length ? record(raw.usage_metadata ?? raw.usageMetadata) : undefined,
      feedback: raw.feedback === "positive" || raw.feedback === "negative" ? raw.feedback : undefined,
      feedbackId: string(raw.feedbackId),
      feedbackComment: string(raw.feedbackComment),
      status: wasInterrupted ? "interrupted" : status,
      wasInterrupted,
      isThinking: status === "streaming",
    }
    if (isAssistant && assistant) {
      const prior = { ...assistant }
      Object.assign(assistant, {
        ...next,
        content: next.content.trim() ? next.content : prior.content,
        images: [...(prior.images ?? []), ...(next.images ?? [])].filter((file, i, files) => files.findIndex((other) => other.id === file.id) === i),
        toolCalls: mergeTools(prior.toolCalls, next.toolCalls),
        subgraphOutputs: mergeSubgraphs(prior.subgraphOutputs, next.subgraphOutputs),
        thinkingSteps: [...new Set([...(prior.thinkingSteps ?? []), ...(next.thinkingSteps ?? [])])],
        thinkingDuration: next.thinkingDuration ?? prior.thinkingDuration,
        runId: next.runId ?? prior.runId,
        shareUrl: next.shareUrl ?? prior.shareUrl,
        usageMetadata: next.usageMetadata ?? prior.usageMetadata,
      })
    } else {
      messages.push(next)
      assistant = isAssistant ? next : undefined
    }
    if (isAssistant && assistant) {
      for (const tool of assistant.toolCalls ?? []) toolOwners.set(tool.id, assistant)
    }
  })

  return messages.filter((message) => Boolean(message.content.trim() || message.images?.length || message.toolCalls?.length || message.subgraphOutputs?.length || message.thinkingSteps?.length || message.status === "failed" || message.status === "interrupted"))
}
