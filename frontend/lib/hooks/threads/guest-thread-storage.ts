"use client"

const GUEST_THREAD_IDS_KEY = "chat-langchain-guest-thread-ids"

/**
 * Bucket for thread ids recorded before guest threads were owner-scoped. They
 * carry no known actor, so they are offered to whichever guest identity is
 * current and dropped once the backend reports them unreachable.
 */
const LEGACY_OWNER_ID = "__legacy__"

/**
 * Guest identities rotate (a new MDA guest token can carry a new `sub`), and a
 * stale bucket's threads are unreachable forever. Keep a bounded number of them
 * so localStorage cannot grow without limit across rotations.
 */
const MAX_OWNERS = 5
const MAX_THREAD_IDS_PER_OWNER = 200

interface GuestThreadStore {
  version: 2
  owners: Record<string, string[]>
}

function emptyStore(): GuestThreadStore {
  return { version: 2, owners: {} }
}

function sanitizeIds(value: unknown): string[] {
  if (!Array.isArray(value)) return []
  const ids = value.filter(
    (item): item is string => typeof item === "string" && item.length > 0
  )
  return Array.from(new Set(ids)).slice(0, MAX_THREAD_IDS_PER_OWNER)
}

function readStore(): GuestThreadStore {
  if (typeof window === "undefined") return emptyStore()

  let parsed: unknown
  try {
    parsed = JSON.parse(localStorage.getItem(GUEST_THREAD_IDS_KEY) || "null")
  } catch {
    return emptyStore()
  }

  // v1 format: a bare array of thread ids with no owner association.
  if (Array.isArray(parsed)) {
    const legacy = sanitizeIds(parsed)
    return legacy.length
      ? { version: 2, owners: { [LEGACY_OWNER_ID]: legacy } }
      : emptyStore()
  }

  if (!parsed || typeof parsed !== "object") return emptyStore()

  const owners = (parsed as { owners?: unknown }).owners
  if (!owners || typeof owners !== "object" || Array.isArray(owners)) {
    return emptyStore()
  }

  const store = emptyStore()
  for (const [ownerId, ids] of Object.entries(owners as Record<string, unknown>)) {
    if (!ownerId) continue
    const sanitized = sanitizeIds(ids)
    if (sanitized.length) store.owners[ownerId] = sanitized
  }
  return store
}

function writeStore(store: GuestThreadStore): void {
  if (typeof window === "undefined") return

  // Object key order is insertion order, so the oldest buckets sort first and
  // are the ones dropped when the cap is exceeded. The legacy bucket is kept
  // until its ids are individually resolved or pruned.
  const entries = Object.entries(store.owners).filter(([, ids]) => ids.length > 0)
  const legacy = entries.filter(([ownerId]) => ownerId === LEGACY_OWNER_ID)
  const scoped = entries.filter(([ownerId]) => ownerId !== LEGACY_OWNER_ID)
  const kept = [...legacy, ...scoped.slice(-MAX_OWNERS)]

  localStorage.setItem(
    GUEST_THREAD_IDS_KEY,
    JSON.stringify({ version: 2, owners: Object.fromEntries(kept) })
  )
}

function visibleIds(store: GuestThreadStore, ownerId: string | null | undefined): string[] {
  const owned = ownerId ? store.owners[ownerId] ?? [] : []
  const legacy = store.owners[LEGACY_OWNER_ID] ?? []
  return Array.from(new Set([...owned, ...legacy]))
}

/**
 * Thread ids worth attempting for this guest identity: the ones it created,
 * plus any unattributed ids left by an older build.
 */
export function getStoredGuestThreadIds(ownerId: string | null | undefined): string[] {
  return visibleIds(readStore(), ownerId)
}

export function addStoredGuestThreadId(
  ownerId: string | null | undefined,
  threadId: string
): string[] {
  if (!ownerId || !threadId) return getStoredGuestThreadIds(ownerId)

  const store = readStore()
  const existing = store.owners[ownerId] ?? []
  store.owners[ownerId] = [threadId, ...existing.filter((id) => id !== threadId)].slice(
    0,
    MAX_THREAD_IDS_PER_OWNER
  )
  writeStore(store)
  return visibleIds(store, ownerId)
}

export function removeStoredGuestThreadId(
  ownerId: string | null | undefined,
  threadId: string
): string[] {
  return pruneStoredGuestThreadIds(ownerId, [threadId])
}

/**
 * Forget thread ids the backend has confirmed this guest cannot reach, so they
 * are not re-requested on every load. Clears them from the current owner and
 * from the unattributed legacy bucket.
 */
export function pruneStoredGuestThreadIds(
  ownerId: string | null | undefined,
  threadIds: string[]
): string[] {
  const doomed = new Set(threadIds)
  if (!doomed.size) return getStoredGuestThreadIds(ownerId)

  const store = readStore()
  for (const bucket of [ownerId, LEGACY_OWNER_ID]) {
    if (!bucket) continue
    const ids = store.owners[bucket]
    if (!ids) continue
    const kept = ids.filter((id) => !doomed.has(id))
    if (kept.length) store.owners[bucket] = kept
    else delete store.owners[bucket]
  }
  writeStore(store)
  return visibleIds(store, ownerId)
}
