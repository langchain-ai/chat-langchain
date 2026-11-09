import assert from "node:assert/strict"
import test from "node:test"
import { restoreThreadMessages } from "../lib/utils/chat/history.ts"

test("restores one final answer while retaining ordered tool calls and actual results", () => {
  const payload = [
    { type: "human", id: "user", content: "Search the docs" },
    { type: "ai", id: "planning", content: "Searching", tool_calls: [{ id: "search", name: "search", args: { query: "LangGraph" } }] },
    { type: "tool", tool_call_id: "search", content: false },
    { type: "ai", id: "final", content: "Here is the answer", response_metadata: { thinking_duration: 42 } },
  ]
  const result = restoreThreadMessages(payload, "thread")
  assert.deepEqual(result.map((message) => message.id), ["user", "final"])
  assert.equal(result[1]?.content, "Here is the answer")
  assert.deepEqual(result[1]?.toolCalls, [{ id: "search", name: "search", args: { query: "LangGraph" }, output: false, status: "succeeded" }])
  assert.equal(result[1]?.thinkingDuration, 42)
  assert.equal("output" in payload[1]!.tool_calls![0]!, false)
})

test("keeps image-only messages and deterministic IDs for multimodal history", () => {
  const payload = [{ role: "user", content: [{ type: "image_url", image_url: { url: "data:image/png;base64,aGVsbG8=" } }] }]
  const first = restoreThreadMessages(payload, "thread")
  const second = restoreThreadMessages(payload, "thread")
  assert.equal(first.length, 1)
  assert.equal(first[0]?.content, "")
  assert.equal(first[0]?.images?.[0]?.base64, "aGVsbG8=")
  assert.equal(first[0]?.images?.[0]?.mimeType, "image/png")
  assert.deepEqual(first, second)
})

test("does not fabricate a successful tool result when the backend has none", () => {
  const [message] = restoreThreadMessages([{ type: "ai", content: "", tool_calls: [{ id: "pending", name: "search", args: {} }] }], "thread")
  assert.equal(message?.toolCalls?.[0]?.status, undefined)
  assert.equal(message?.toolCalls?.[0]?.output, undefined)
  assert.equal(message?.content, "")
})

test("restores failed subagent execution and preserves explicit step metadata", () => {
  const result = restoreThreadMessages([
    { type: "human", content: "Research" },
    { type: "ai", tool_calls: [{ id: "task", name: "task", args: { subagent_type: "docs researcher" } }], response_metadata: { thinking_steps: ["Look up documentation"] } },
    { type: "tool", tool_call_id: "task", status: "error", content: "Search unavailable" },
    { type: "assistant", id: "answer", content: "Try again", status: "failed" },
  ], "thread")
  assert.equal(result[1]?.status, "failed")
  assert.deepEqual(result[1]?.thinkingSteps, ["Look up documentation"])
  assert.equal(result[1]?.toolCalls?.[0]?.status, "failed")
  assert.equal(result[1]?.subgraphOutputs?.[0]?.name, "docs researcher")
  assert.equal(result[1]?.subgraphOutputs?.[0]?.status, "failed")
  assert.equal(result[1]?.subgraphOutputs?.[0]?.output, "Search unavailable")
})

test("separate user turns retain their own tools and interrupted state", () => {
  const result = restoreThreadMessages([
    { type: "human", id: "u1", content: "First" },
    { type: "ai", id: "a1", content: "Partial", was_interrupted: true, tool_calls: [{ id: "one", name: "search", args: {} }] },
    { type: "human", id: "u2", content: "Second" },
    { type: "ai", id: "a2", content: "Second answer" },
  ], "thread")
  assert.deepEqual(result.map((message) => message.id), ["u1", "a1", "u2", "a2"])
  assert.equal(result[1]?.status, "interrupted")
  assert.equal(result[3]?.toolCalls?.length, 0)
})

test("tolerates missing or unrelated backend data without displaying hidden reasoning", () => {
  assert.deepEqual(restoreThreadMessages(null, "thread"), [])
  const result = restoreThreadMessages([null, { type: "system", content: "Private system text" }, { type: "ai", content: [{ type: "reasoning", text: "Private reasoning" }, { type: "text", text: "Visible answer" }] }], "thread")
  assert.equal(result.length, 1)
  assert.equal(result[0]?.content, "Visible answer")
  assert.deepEqual(result[0]?.thinkingSteps, [])
})

test("omits backend context summaries from restored conversation turns", () => {
  const result = restoreThreadMessages([
    { type: "human", id: "summary", content: "Current User Goal: learn agents", additional_kwargs: { lc_source: "summarization" } },
    { type: "human", id: "question", content: "How do agents work?" },
    { type: "ai", id: "answer", content: "Agents loop over tools." },
  ], "thread")
  assert.deepEqual(result.map((message) => message.id), ["question", "answer"])
})
