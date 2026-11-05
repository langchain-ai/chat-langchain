import assert from "node:assert/strict"
import test from "node:test"
import { mergeToolCalls, messageStatus, subgraphStatus, toolStatus } from "../lib/utils/chat/presentation.ts"
import type { Message } from "../lib/types/messages.ts"

const message: Message = { id: "reply-1", role: "assistant", content: "", timestamp: new Date(0) }
const tool = { id: "call-1", name: "search", args: { query: "LangGraph" } }

test("missing tool output stays unresolved after a response settles", () => {
  assert.equal(toolStatus(tool, message), "pending")
  assert.equal(toolStatus(tool, { ...message, isThinking: true }), "running")
  for (const output of [false, 0, "", null]) {
    assert.equal(toolStatus({ ...tool, output }, message), "succeeded")
  }
})

test("interruption and failure settle unfinished work without overriding completed calls", () => {
  assert.equal(messageStatus({ ...message, isThinking: true, wasInterrupted: true }), "interrupted")
  assert.equal(messageStatus({ ...message, isThinking: true, status: "failed" }), "failed")
  assert.equal(toolStatus(tool, { ...message, wasInterrupted: true }), "canceled")
  assert.equal(toolStatus({ ...tool, status: "succeeded" }, { ...message, status: "failed" }), "succeeded")
  assert.equal(toolStatus({ ...tool, status: "failed" }, message), "failed")
})

test("subagent output is only complete with an explicit completion signal", () => {
  const subgraph = { name: "research", output: "Partial result", timestamp: 1, toolCallId: "call-1" }
  assert.equal(subgraphStatus(subgraph, message), "pending")
  assert.equal(subgraphStatus({ ...subgraph, isStreaming: true }, { ...message, isThinking: true }), "running")
  assert.equal(subgraphStatus({ ...subgraph, isComplete: true }, message), "succeeded")
  assert.equal(subgraphStatus({ ...subgraph, isStreaming: true }, { ...message, wasInterrupted: true }), "canceled")
  assert.equal(subgraphStatus({ ...subgraph, status: "failed", isComplete: true }, message), "failed")
})

test("repeated tool-call events preserve results, stable IDs, and execution order", () => {
  const completed = { ...tool, output: false, status: "succeeded" as const }
  const next = mergeToolCalls([completed], [tool, { ...tool, id: "call-2" }])
  assert.deepEqual(next.map((call) => call.id), ["call-1", "call-2"])
  assert.equal(next[0]?.output, false)
  assert.equal(next[0]?.status, "succeeded")
  assert.equal(next[1]?.status, "running")
  assert.notEqual(next[0], completed)
})
