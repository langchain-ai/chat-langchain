import assert from "node:assert/strict"
import { randomUUID } from "node:crypto"
import { readFileSync } from "node:fs"
import test from "node:test"
import { runInNewContext } from "node:vm"
import ts from "typescript"
import { extractTextFromContent } from "../lib/utils/chat/content-helpers.ts"
import { getFinalResponseContent } from "../lib/utils/chat/final-response.ts"

const source = readFileSync(
  new URL("../lib/hooks/chat/use-stream-handler.ts", import.meta.url), "utf8"
)
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS },
}).outputText

async function processEvents(makeEvents: (userId: string) => unknown[]) {
  const exports: Record<string, any> = {}
  runInNewContext(compiled, {
    exports,
    require: (name: string) => {
      if (name === "react") return { useCallback: (callback: unknown) => callback }
      if (name.endsWith("/final-response")) return { getFinalResponseContent }
      if (name.endsWith("/utils/chat")) return {
        extractTextFromContent,
        ensureMessageExists: (messages: any[], id: string, message: any) =>
          messages.some((item) => item.id === id) ? messages : [...messages, message],
        updateMessageInList: (messages: any[], id: string, update: any) =>
          messages.map((item) => item.id === id ? { ...item, ...update } : item),
      }
      if (name.endsWith("/api/langsmith")) return {}
      if (name.endsWith("/constants/api")) return { LANGGRAPH_API_URL: "https://test.invalid" }
      throw new Error(`Unexpected import: ${name}`)
    },
    crypto: { randomUUID },
    console: { log() {}, error() {} },
  })
  let messages: any[] = []
  const client = {
    runs: {
      stream: async function* (_thread: string, _agent: string, options: any) {
        for (const event of makeEvents(options.input.messages[0].id)) yield event
      },
    },
  }
  const { processStream } = exports.useStreamHandler({
    client,
    threadId: "thread",
    auth: {},
    setMessages: (update: (previous: any[]) => any[]) => { messages = update(messages) },
  })
  const result = await processStream("Follow-up", "assistant")
  return { result, message: messages.find((message) => message.id === "assistant") }
}

function values(userId: string, content: unknown) {
  return {
    event: "values",
    data: { messages: [
      { type: "human", id: userId, content: "Follow-up" },
      { type: "ai", id: "final", content },
    ] },
  }
}

for (const sanitized of ["Read the docs.", "", [{ type: "text", text: "Read the docs." }]]) {
  test(`completed chat replaces streamed URLs with sanitized content: ${JSON.stringify(sanitized)}`, async () => {
    const { result, message } = await processEvents((userId) => [
      values("previous-turn", "Previous answer"),
      { event: "messages", data: [{ type: "ai", content: "https://invalid.example" }] },
      values(userId, "https://invalid.example"),
      values(userId, sanitized),
    ])
    const expected = extractTextFromContent(sanitized)
    assert.equal(result.assistantContent, expected)
    assert.equal(message.content, expected)
    assert.equal(message.isThinking, false)
    assert.ok(!message.content.includes("https://invalid.example"))
  })
}

test("current-turn values work without token streaming", async () => {
  const { result, message } = await processEvents((userId) => [values(userId, "Final answer")])
  assert.equal(result.assistantContent, "Final answer")
  assert.equal(message.content, "Final answer")
})

test("previous history, tool calls and subagent responses do not overwrite the final answer", async () => {
  const { result, message } = await processEvents((userId) => [
    values(userId, "Final answer"),
    values("previous-turn", "Previous answer"),
    { ...values(userId, "Subgraph answer"), event: "values|subgraph" },
    values(userId, '{"answer":"Subagent answer"}'),
    { event: "values", data: { messages: [
      { type: "human", id: userId },
      { type: "ai", content: "Planning", tool_calls: [{ name: "search" }] },
    ] } },
    { event: "messages", data: [{ type: "ai", content: "Late unsanitized tokens" }] },
  ])
  assert.equal(result.assistantContent, "Final answer")
  assert.equal(message.content, "Final answer")
})

test("history-only values do not become a new response", async () => {
  const { result, message } = await processEvents(() => [values("previous-turn", "Previous answer")])
  assert.equal(result.assistantContent, "")
  assert.equal(message.content, "(No response generated)")
})
