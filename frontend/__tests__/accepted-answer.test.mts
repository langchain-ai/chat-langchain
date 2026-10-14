import assert from "node:assert/strict"
import test from "node:test"
import { extractAcceptedAnswer } from "../lib/utils/chat/content-helpers.ts"

const human = { type: "human", content: "Explain the program" }
const rejected = { type: "ai", content: "Title\n" + "*   ".repeat(5000) }

test("ignores failed-attempt tokens and uses the accepted graph-state answer", () => {
  assert.equal(extractAcceptedAnswer("messages", { messages: [human, rejected] }), null)
  assert.equal(extractAcceptedAnswer("messages/partial", { messages: [human, rejected] }), null)
  assert.equal(extractAcceptedAnswer("updates", { messages: [human, rejected] }), null)
  assert.equal(extractAcceptedAnswer("values|subagent", { messages: [human, rejected] }), null)
  assert.equal(extractAcceptedAnswer("values", {
    messages: [human, { type: "ai", content: "This line adds documents." }],
  }), "This line adds documents.")
})

test("shows recovery even when there were no successful model tokens", () => {
  const recovery = "I couldn't generate an answer, please rephrase."
  assert.equal(extractAcceptedAnswer("values", {
    messages: [human, { type: "ai", content: recovery }],
  }), recovery)
})

test("does not reuse a prior turn or display pending tool calls", () => {
  assert.equal(extractAcceptedAnswer("values", {
    messages: [human, { type: "ai", content: "Old answer" }, human],
  }), null)
  assert.equal(extractAcceptedAnswer("values", {
    messages: [human, { type: "ai", content: "Searching", tool_calls: [{}] }],
  }), null)
})

test("accepts text blocks and role-based serialized messages", () => {
  assert.equal(extractAcceptedAnswer("values", {
    messages: [
      { role: "user", content: "Explain" },
      { role: "assistant", content: [{ type: "text", text: "Accepted answer" }] },
    ],
  }), "Accepted answer")
})
