interface StreamMessage {
  id?: string
  type?: string
  role?: string
  content?: unknown
  tool_calls?: unknown[]
}

export function getFinalResponseContent(
  messages: StreamMessage[],
  userMessageId: string,
  extractText: (content: unknown) => string
): string | undefined {
  const latestUser = [...messages].reverse().find(
    (message) => message.type === "human" || message.role === "user"
  )
  const finalMessage = messages[messages.length - 1]
  if (
    latestUser?.id !== userMessageId ||
    !finalMessage ||
    (finalMessage.type !== "ai" && finalMessage.role !== "assistant") ||
    finalMessage.tool_calls?.length
  ) {
    return undefined
  }

  const content = extractText(finalMessage.content)
  return content.trim().startsWith("{") ? undefined : content
}
