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

export const extractAcceptedAnswer = (
  event: string,
  data: { messages?: { type?: string; role?: string; content?: unknown; tool_calls?: unknown[] }[] } | undefined
): string | null => {
  if (event !== "values" || !Array.isArray(data?.messages)) return null

  const messages = data.messages
  const humanIndex = messages.findLastIndex(message =>
    message.type === "human" || message.role === "user"
  )
  const assistantIndex = messages.findLastIndex(message =>
    message.type === "ai" || message.role === "assistant"
  )
  if (humanIndex < 0 || assistantIndex <= humanIndex) return null

  const message = messages[assistantIndex]
  if (message.tool_calls?.length) return null
  const content = extractTextFromContent(message.content)
  return content && !content.trim().startsWith("{") ? content : null
}
