import assert from "node:assert/strict"
import test from "node:test"
import {
  persistStreamFailure,
  STREAM_FAILURE_MESSAGE,
} from "../lib/hooks/chat/stream-failure.ts"


test("persistStreamFailure checkpoints a non-empty assistant message", async () => {
  const updates: unknown[] = []
  const messages = [{ id: "assistant-1", role: "assistant", content: "", timestamp: new Date() }]
  const client = {
    threads: {
      updateState: async (_threadId: string, options: unknown) => {
        updates.push(options)
        return { configurable: {} }
      },
    },
  }
  await persistStreamFailure(client as never, "thread-1", "assistant-1")

  assert.deepEqual(updates, [
    {
      values: {
        messages: [{ id: "assistant-1", role: "assistant", content: STREAM_FAILURE_MESSAGE }],
      },
    },
  ])
})
