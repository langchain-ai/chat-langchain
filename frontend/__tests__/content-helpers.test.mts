import assert from "node:assert/strict"
import test from "node:test"
import {
  getAssistantAnswerText,
  resolveLatestAssistantAnswer,
} from "../lib/utils/chat/content-helpers.ts"

test("resolves the latest non-empty assistant answer", () => {
  assert.equal(
    resolveLatestAssistantAnswer([
      { type: "ai", content: "Earlier answer" },
      { type: "ai", content: [] },
      { type: "ai", content: [{ type: "text", text: "Latest answer" }] },
    ]),
    "Latest answer"
  )
})

test("skips middleware payloads and tool-call messages", () => {
  assert.equal(
    resolveLatestAssistantAnswer([
      { type: "ai", content: "Genuine answer" },
      { type: "ai", content: '{"decision":"ALLOWED"}' },
      { type: "ai", content: "Summary", metadata: { lc_source: "summarization" } },
      { type: "ai", content: "Tool call", tool_calls: [{ name: "search" }] },
    ]),
    "Genuine answer"
  )
})

test("rejects classifier and summary-shaped JSON without metadata", () => {
  assert.equal(getAssistantAnswerText({ type: "ai", content: '{"messages": []}' }), "")
  assert.equal(getAssistantAnswerText({ type: "ai", content: '{"decision":"ALLOWED"}' }), "")
})
