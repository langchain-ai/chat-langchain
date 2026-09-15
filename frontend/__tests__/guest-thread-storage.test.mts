import assert from "node:assert/strict"
import test, { beforeEach } from "node:test"
import {
  addStoredGuestThreadId,
  getStoredGuestThreadIds,
  pruneStoredGuestThreadIds,
  removeStoredGuestThreadId,
} from "../lib/hooks/threads/guest-thread-storage.ts"

const KEY = "chat-langchain-guest-thread-ids"

class MemoryStorage {
  private readonly values = new Map<string, string>()

  getItem(key: string): string | null {
    return this.values.get(key) ?? null
  }

  setItem(key: string, value: string): void {
    this.values.set(key, value)
  }

  clear(): void {
    this.values.clear()
  }
}

const storage = new MemoryStorage()
;(globalThis as any).window = globalThis
;(globalThis as any).localStorage = storage

beforeEach(() => {
  storage.clear()
})

test("thread ids are scoped to the guest actor that created them", () => {
  addStoredGuestThreadId("guest:a", "thread-1")
  addStoredGuestThreadId("guest:b", "thread-2")

  assert.deepEqual(getStoredGuestThreadIds("guest:a"), ["thread-1"])
  assert.deepEqual(getStoredGuestThreadIds("guest:b"), ["thread-2"])
})

test("an unknown owner sees no other actor's threads", () => {
  addStoredGuestThreadId("guest:a", "thread-1")

  assert.deepEqual(getStoredGuestThreadIds("guest:rotated"), [])
  assert.deepEqual(getStoredGuestThreadIds(null), [])
})

test("legacy flat-array ids are offered to whichever identity is current", () => {
  storage.setItem(KEY, JSON.stringify(["legacy-1", "legacy-2"]))

  assert.deepEqual(getStoredGuestThreadIds("guest:a"), ["legacy-1", "legacy-2"])
  assert.deepEqual(getStoredGuestThreadIds("guest:b"), ["legacy-1", "legacy-2"])
})

test("pruning forgets unreachable ids from owner and legacy buckets", () => {
  storage.setItem(KEY, JSON.stringify(["legacy-1"]))
  addStoredGuestThreadId("guest:a", "thread-1")
  addStoredGuestThreadId("guest:a", "thread-2")

  const remaining = pruneStoredGuestThreadIds("guest:a", ["thread-1", "legacy-1"])

  assert.deepEqual(remaining, ["thread-2"])
  assert.deepEqual(getStoredGuestThreadIds("guest:a"), ["thread-2"])
})

test("pruning nothing leaves the store untouched", () => {
  addStoredGuestThreadId("guest:a", "thread-1")

  assert.deepEqual(pruneStoredGuestThreadIds("guest:a", []), ["thread-1"])
})

test("removal drops a single id", () => {
  addStoredGuestThreadId("guest:a", "thread-1")
  addStoredGuestThreadId("guest:a", "thread-2")

  assert.deepEqual(removeStoredGuestThreadId("guest:a", "thread-1"), ["thread-2"])
})

test("re-adding an id keeps it unique and most-recent-first", () => {
  addStoredGuestThreadId("guest:a", "thread-1")
  addStoredGuestThreadId("guest:a", "thread-2")
  addStoredGuestThreadId("guest:a", "thread-1")

  assert.deepEqual(getStoredGuestThreadIds("guest:a"), ["thread-1", "thread-2"])
})

test("owner buckets are bounded so rotations cannot grow storage forever", () => {
  for (let i = 0; i < 8; i += 1) {
    addStoredGuestThreadId(`guest:${i}`, `thread-${i}`)
  }

  const owners = JSON.parse(storage.getItem(KEY) || "{}").owners
  assert.equal(Object.keys(owners).length, 5)
  assert.deepEqual(getStoredGuestThreadIds("guest:0"), [])
  assert.deepEqual(getStoredGuestThreadIds("guest:7"), ["thread-7"])
})

test("stored ids per owner are capped", () => {
  for (let i = 0; i < 250; i += 1) {
    addStoredGuestThreadId("guest:a", `thread-${i}`)
  }

  assert.equal(getStoredGuestThreadIds("guest:a").length, 200)
})

test("corrupt storage degrades to empty rather than throwing", () => {
  storage.setItem(KEY, "{not json")
  assert.deepEqual(getStoredGuestThreadIds("guest:a"), [])

  storage.setItem(KEY, JSON.stringify({ version: 2, owners: "nope" }))
  assert.deepEqual(getStoredGuestThreadIds("guest:a"), [])
})

test("a write with no known owner does not strand the id under a bogus bucket", () => {
  assert.deepEqual(addStoredGuestThreadId(null, "thread-1"), [])
  assert.equal(storage.getItem(KEY), null)
})
