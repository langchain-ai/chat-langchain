export const FORUM_URL = "https://forum.langchain.com/"
export const SUPPORT_URL = "https://support.langchain.com/"

const MAX_TITLE = 120

/**
 * Where to send someone after they mark an answer "Not helpful".
 * The forum link opens a new topic prefilled with their question as the title,
 * so they don't have to retype it. Nothing else from the conversation is sent.
 */
export function stillStuckLinks(question?: string): { forum: string; support: string } {
  const title = (question ?? "")
    .split("\n\n---\n")[0]
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, MAX_TITLE)
  if (!title) return { forum: FORUM_URL, support: SUPPORT_URL }
  const forum = `${FORUM_URL}new-topic?title=${encodeURIComponent(title)}`
  return { forum, support: SUPPORT_URL }
}
