import assert from "node:assert/strict"
import test from "node:test"
import { failActiveRun, mergeToolCalls, messageStatus, settleInterruptedMessages, subgraphStatus, toolStatus } from "../lib/utils/chat/presentation.ts"
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

test("a transport rejection after stop settles the active reply and preserves partial results", () => {
  const pending: Message = {
    ...message,
    content: "Here is the partial answer",
    status: "streaming",
    isThinking: true,
    thinkingStartTime: 100,
    toolCalls: [
      { ...tool, status: "succeeded", output: false },
      { ...tool, id: "running", status: "running", output: "Partial output" },
      { ...tool, id: "failed", status: "failed", error: "Search unavailable" },
      { ...tool, id: "canceled", status: "canceled" },
    ],
    subgraphOutputs: [
      { name: "research", output: "Partial research", timestamp: 1, isStreaming: true },
      { name: "completed", output: "Complete research", timestamp: 2, isComplete: true },
    ],
  }
  const [result] = settleInterruptedMessages([pending], pending.id, 600)
  assert.equal(result?.content, pending.content)
  assert.equal(result?.status, "interrupted")
  assert.equal(result?.isThinking, false)
  assert.equal(result?.wasInterrupted, true)
  assert.equal(result?.thinkingDuration, 500)
  assert.deepEqual(result?.toolCalls?.map((call) => call.status), ["succeeded", "canceled", "failed", "canceled"])
  assert.equal(result?.toolCalls?.[1]?.output, "Partial output")
  assert.deepEqual(result?.subgraphOutputs?.map((subgraph) => subgraph.status), ["canceled", "succeeded"])
  assert.equal(result?.subgraphOutputs?.[0]?.isStreaming, false)
  assert.equal(pending.isThinking, true)
})

test("stop before the first stream event still produces a settled assistant message", () => {
  const [result] = settleInterruptedMessages([], "new-reply", 100)
  assert.equal(result?.id, "new-reply")
  assert.equal(result?.status, "interrupted")
  assert.match(result?.content ?? "", /Response stopped/)
})

test("a failed run produces one failed reply that keeps partial content", () => {
  const user: Message = { id: "user-1", role: "user", content: "Search the docs", timestamp: new Date(0) }
  const pending: Message = { ...message, content: "Partial answer", status: "streaming", isThinking: true, thinkingStartTime: 100, toolCalls: [{ ...tool, status: "running" }] }
  const result = failActiveRun([user, pending], pending.id, "Error: Search unavailable", 600)
  assert.deepEqual(result.map((item) => item.id), ["user-1", "reply-1"])
  assert.equal(result[1]?.status, "failed")
  assert.equal(result[1]?.isThinking, false)
  assert.equal(result[1]?.content, "Partial answer\n\nError: Search unavailable")
  assert.equal(result[1]?.thinkingDuration, 500)
  assert.equal(toolStatus(result[1]!.toolCalls![0]!, result[1]!), "canceled")
})

test("a run that fails before its first event appends a single failed reply", () => {
  const [result] = failActiveRun([], "new-reply", "Error: Failed to connect to the agent", 100)
  assert.equal(result?.id, "new-reply")
  assert.equal(result?.status, "failed")
  assert.equal(result?.content, "Error: Failed to connect to the agent")
})
