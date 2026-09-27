import assert from "node:assert/strict"
import test from "node:test"
import {
  FAILED_RESPONSE_MESSAGE,
  repairFailedThreadMessages,
} from "../lib/utils/chat/failed-thread-state.ts"

test("repairs a failed turn without replaying its tool history", () => {
  const values = {
    messages: [
      { role: "user", content: "Earlier question" },
      { role: "assistant", content: "Earlier answer" },
      { role: "user", content: "Failed question" },
      { type: "ai", tool_calls: [{ name: "check_links" }] },
      { type: "tool", content: "repeated check_links output" },
    ],
  }

  const repaired = repairFailedThreadMessages(values)
  const messages = repaired.messages as Array<Record<string, unknown>>

  assert.deepEqual(messages, [
    { role: "user", content: "Earlier question" },
    { role: "assistant", content: "Earlier answer" },
    { role: "user", content: "Failed question" },
    { role: "assistant", content: FAILED_RESPONSE_MESSAGE },
  ])
})
