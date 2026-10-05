/**
 * Content Extraction and Formatting Utilities
 *
 * Functions for extracting and formatting message content from various formats.
 */

/**
 * Extract text content from various message content formats.
 * Handles:
 * - String content
 * - Array content with text objects
 * - Mixed content types
 */
export const extractTextFromContent = (content: any): string => {
  if (typeof content === "string") return content

  if (Array.isArray(content)) {
    return content
      .filter((c: any) => typeof c === "string" || c?.type === "text")
      .map((c: any) => (typeof c === "string" ? c : c.text || ""))
      .join("\n\n")
  }

  return ""
}

const getMessageMetadata = (message: any): Record<string, any> => ({
  ...(message?.metadata || {}),
  ...(message?.response_metadata || {}),
  ...(message?.additional_kwargs || {}),
})

const isInternalMessage = (message: any, text: string): boolean => {
  const metadata = getMessageMetadata(message)
  const source = String(metadata.lc_source || metadata.source || "").toLowerCase()
  const messageType = String(message?.type || "").toLowerCase()
  const name = String(message?.name || "").toLowerCase()

  if (
    source.includes("summar") ||
    source.includes("guardrail") ||
    messageType.includes("summary") ||
    messageType.includes("middleware") ||
    name.includes("summar") ||
    name.includes("guardrail") ||
    name.includes("classifier")
  ) {
    return true
  }

  try {
    const parsed = JSON.parse(text)
    return (
      parsed !== null &&
      typeof parsed === "object" &&
      (Object.prototype.hasOwnProperty.call(parsed, "decision") ||
        Object.prototype.hasOwnProperty.call(parsed, "messages"))
    )
  } catch {
    return false
  }
}

export const getAssistantAnswerText = (message: any): string => {
  if (
    !message ||
    (message.type !== "ai" && message.type !== "assistant" && message.role !== "assistant") ||
    (Array.isArray(message.tool_calls) && message.tool_calls.length > 0) ||
    (Array.isArray(message.toolCalls) && message.toolCalls.length > 0)
  ) {
    return ""
  }

  const text = extractTextFromContent(message.content).trim()
  return text && !isInternalMessage(message, text) ? text : ""
}

export const resolveLatestAssistantAnswer = (messages: any[]): string => {
  for (const message of [...messages].reverse()) {
    const text = getAssistantAnswerText(message)
    if (text) return text
  }

  return ""
}
