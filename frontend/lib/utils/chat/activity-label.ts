/**
 * Label for the single activity block shown while an answer is being worked on.
 * While streaming, the block shows the latest step in place (earlier steps stay
 * inside it, collapsed); once the run settles it becomes "Agent steps (N)".
 */
export function activityLabel(steps: string[], isStreaming: boolean): string {
  if (!isStreaming || steps.length === 0) return `Agent steps (${steps.length})`
  const latest = steps[steps.length - 1].trim().replace(/\.+$/, "")
  return steps.length > 1 ? `${latest}… (step ${steps.length})` : `${latest}…`
}
