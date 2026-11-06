import assert from "node:assert/strict"
import test from "node:test"
import { isDisplayableMessage } from "../lib/utils/chat/message-helpers.ts"

test("thread history excludes internal summaries while retaining user and assistant turns", () => {
  const messages = [
    { type: "human", content: "Current User Goal: learn about agents", additional_kwargs: { lc_source: "summarization" } },
    { type: "human", content: "How do agents work?" },
    { type: "ai", content: "Agents use tools." },
    { role: "user", content: "How do I stream them?", additional_kwargs: {} },
    { role: "assistant", content: "Use stream()." },
    { type: "tool", content: "internal tool result" },
  ]

  assert.deepEqual(messages.filter(isDisplayableMessage).map(msg => msg.content), [
    "How do agents work?",
    "Agents use tools.",
    "How do I stream them?",
    "Use stream().",
  ])
})

test("streamed state selects the assistant response without exposing the summary", () => {
  const summary = { type: "human", content: "Current User Goal", additional_kwargs: { lc_source: "summarization" } }
  const answer = { role: "assistant", content: "Use stream()." }
  const messages = [summary, { role: "user", content: "How do I stream?" }, answer]

  assert.equal(isDisplayableMessage(summary), false)
  assert.equal(messages.filter(isDisplayableMessage).at(-1), answer)
  assert.equal(isDisplayableMessage({ ...summary, type: "ai" }), false)
})
