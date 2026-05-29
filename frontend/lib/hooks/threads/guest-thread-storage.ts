"use client"

const GUEST_THREAD_IDS_KEY = "chat-langchain-guest-thread-ids"

function readIds(): string[] {
  if (typeof window === "undefined") return []

  try {
    const parsed = JSON.parse(localStorage.getItem(GUEST_THREAD_IDS_KEY) || "[]")
    if (!Array.isArray(parsed)) return []
    return parsed.filter(
      (item): item is string => typeof item === "string" && item.length > 0
    )
  } catch {
    return []
  }
}

function writeIds(threadIds: string[]): void {
  if (typeof window === "undefined") return
  localStorage.setItem(GUEST_THREAD_IDS_KEY, JSON.stringify(Array.from(new Set(threadIds))))
}

export function getStoredGuestThreadIds(): string[] {
  return readIds()
}

export function addStoredGuestThreadId(threadId: string): string[] {
  const threadIds = [threadId, ...readIds().filter((id) => id !== threadId)]
  writeIds(threadIds)
  return threadIds
}

export function removeStoredGuestThreadId(threadId: string): string[] {
  const threadIds = readIds().filter((id) => id !== threadId)
  writeIds(threadIds)
  return threadIds
}
