export function getStreamErrorMessage(error: unknown): string {
  if (error instanceof Error && error.message.trim()) {
    return error.message
  }

  if (typeof error === "object" && error !== null) {
    const message = (error as { message?: unknown }).message
    if (typeof message === "string" && message.trim()) {
      return message
    }
  }

  return "The agent failed while processing your request."
}

export function getAssistantCompletionContent(
  assistantContent: string,
  wasInterrupted: boolean,
  streamError?: unknown
): string {
  if (assistantContent.trim()) {
    return assistantContent
  }

  if (streamError) {
    return `The agent failed while processing your request: ${getStreamErrorMessage(streamError)}`
  }

  if (wasInterrupted) {
    return "Response stopped. The agent was interrupted while processing your request."
  }

  return "(No response generated)"
}
