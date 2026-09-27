import assert from "node:assert/strict"
import test from "node:test"

import {
  appendRunFailureMessage,
  RUN_FAILURE_MESSAGE,
} from "../lib/hooks/chat/run-failure.ts"

test("persists a terminal AI message without replacing thread history", async () => {
  let update: unknown
  const client = {
    threads: {
      updateState: async (_threadId: string, options: unknown) => {
        update = options
        return { configurable: {} }
      },
    },
  }

  await appendRunFailureMessage(client, "thread-1", "assistant-1", RUN_FAILURE_MESSAGE)

  assert.deepEqual(update, {
    values: {
      messages: [{
        id: "assistant-1",
        type: "ai",
        role: "assistant",
        content: RUN_FAILURE_MESSAGE,
      }],
    },
  })
})
