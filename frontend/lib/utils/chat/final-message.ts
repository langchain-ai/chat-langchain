/**
 * Decide whether the last AI message in a "values" event is this run's answer.
 *
 * The first "values" event of a run carries the thread history as it was when
 * the run started, so the last AI message in it (if any) is the previous turn's
 * answer. Any later AI message with a different id is new. This lets a fast
 * answer that never produced "messages" token events still be shown, instead
 * of "(No response generated)" (seen after a stop + immediate resend, Nov 18).
 */
export function lastAiMessageId(messages: unknown[]): string | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i] as { type?: string; role?: string; id?: string } | null
    if (m && (m.type === "ai" || m.role === "assistant")) return m.id ?? null
  }
  return null
}

export function isNewFinalMessage(
  baseline: { captured: boolean; id: string | null },
  message: { id?: string } | null | undefined,
): boolean {
  if (!baseline.captured || !message || !message.id) return false
  return message.id !== baseline.id
}
