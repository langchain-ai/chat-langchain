export const FAILED_RESPONSE_MESSAGE = "The response failed. Please try again."

export function repairFailedThreadMessages(values: Record<string, unknown>) {
  const messages = Array.isArray(values.messages) ? values.messages : []
  let lastHumanMessageIndex = -1

  messages.forEach((message: any, index) => {
    if (message?.role === "user" || message?.type === "human") {
      lastHumanMessageIndex = index
    }
  })

  const repairedMessages = messages.slice(0, lastHumanMessageIndex + 1)
  repairedMessages.push({ role: "assistant", content: FAILED_RESPONSE_MESSAGE })

  return { ...values, messages: repairedMessages }
}
