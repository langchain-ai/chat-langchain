import assert from "node:assert/strict"
import test from "node:test"
import {
  getAssistantCompletionContent,
  getStreamErrorMessage,
} from "../lib/hooks/chat/stream-error.ts"

test("recursion failures become a terminal assistant error when no output streamed", () => {
  const recursionError = new Error("Recursion limit of 100 reached")

  assert.equal(
    getAssistantCompletionContent("", false, recursionError),
    "The agent failed while processing your request: Recursion limit of 100 reached"
  )
})

test("non-empty assistant output is preserved when a stream later fails", () => {
  assert.equal(
    getAssistantCompletionContent(
      "The answer is ready.",
      false,
      new Error("network disconnected")
    ),
    "The answer is ready."
  )
})

test("unknown stream errors use a non-empty terminal message", () => {
  assert.equal(
    getStreamErrorMessage({}),
    "The agent failed while processing your request."
  )
})
