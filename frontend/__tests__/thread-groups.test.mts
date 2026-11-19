import assert from "node:assert/strict"
import test from "node:test"
import { groupThreads } from "../lib/utils/chat/thread-groups.ts"

const at = (y: number, m: number, d: number, h: number, min = 0) => new Date(y, m, d, h, min).toISOString()
const titles = (threads: { updated_at: string }[], now: Date) =>
  groupThreads(threads, now).map((g) => [g.title, g.threads.length])

test("last night's chat is Yesterday, not Today", () => {
  const now = new Date(2026, 10, 19, 9, 30)
  assert.deepEqual(titles([{ updated_at: at(2026, 10, 18, 21, 31) }], now), [["Yesterday", 1]])
})

test("this morning's chat is Today", () => {
  const now = new Date(2026, 10, 19, 9, 30)
  assert.deepEqual(titles([{ updated_at: at(2026, 10, 19, 0, 5) }], now), [["Today", 1]])
})

test("two days ago is Previous 7 Days even if under 48h", () => {
  const now = new Date(2026, 10, 19, 9, 30)
  assert.deepEqual(titles([{ updated_at: at(2026, 10, 17, 22, 0) }], now), [["Previous 7 Days", 1]])
})

test("older than a week is Older; order of groups is kept", () => {
  const now = new Date(2026, 10, 19, 9, 30)
  assert.deepEqual(
    titles([{ updated_at: at(2026, 9, 1, 12) }, { updated_at: at(2026, 10, 19, 8) }], now),
    [["Today", 1], ["Older", 1]],
  )
})
