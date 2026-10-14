import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import test from "node:test"
import { runInNewContext } from "node:vm"
import ts from "typescript"

const source = readFileSync(new URL("../lib/hooks/chat/use-stream-handler.ts", import.meta.url), "utf8")
const contentHelpers = readFileSync(new URL("../lib/utils/chat/content-helpers.ts", import.meta.url), "utf8")

function loadModule(source: string, dependencies: Record<string, unknown> = {}) {
  const exports: Record<string, any> = {}
  const { outputText } = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
  })
  runInNewContext(outputText, {
    exports,
    require: (name: string) => {
      assert.ok(name in dependencies, `Unexpected dependency: ${name}`)
      return dependencies[name]
    },
    console: { log() {}, warn() {}, error() {} },
  })
  return exports
}

const helpers = loadModule(contentHelpers)
const { useStreamHandler } = loadModule(source, {
  react: { useCallback: (callback: unknown) => callback },
  "../../utils/chat": {
    ...helpers,
    ensureMessageExists: (messages: any[], id: string, message: unknown) =>
      messages.some((item) => item.id === id) ? messages : [...messages, message],
    updateMessageInList: (messages: any[], id: string, update: object) =>
      messages.map((message) => message.id === id ? { ...message, ...update } : message),
  },
  "../../api/langsmith": {},
  "../../constants/api": { LANGGRAPH_API_URL: "https://example.com" },
})

const badUrl = "https://docs.langchain.com/missing"
const human = { type: "human", id: "answer-input", content: "Follow up" }
const ai = (content: unknown, extra: object = {}) => ({ type: "ai", content, ...extra })
const tokens = (content: string) => ({ event: "messages", data: [ai(content)] })
const values = (...messages: unknown[]) => ({ event: "values", data: { messages } })

async function processEvents(events: unknown[]) {
  let messages: any[] = []
  let submittedInput: any
  const client = {
    runs: {
      async *stream(_threadId: string, _agent: string, options: any) {
        submittedInput = options.input
        yield* events
      },
    },
  }
  const { processStream } = useStreamHandler({
    client,
    threadId: "thread",
    setMessages: (update: (messages: any[]) => any[]) => { messages = update(messages) },
  })
  const result = await processStream("Follow up", "answer")
  assert.equal(submittedInput.messages[0].id, human.id)
  return { result, displayed: messages.find((message) => message.id === "answer") }
}

test("replaces streamed raw links with the sanitized final response", async () => {
  const { result, displayed } = await processEvents([
    tokens(`Read [Missing](${badUrl}).`),
    values(human, ai(`Read [Missing](${badUrl}).`)),
    values(human, ai("Read Missing.")),
  ])
  assert.equal(result.assistantContent, "Read Missing.")
  assert.equal(displayed.content, "Read Missing.")
  assert.equal(displayed.isThinking, false)
  assert.ok(!displayed.content.includes(badUrl))
})

test("clears streamed text when sanitization removes the entire response", async () => {
  const { result, displayed } = await processEvents([
    tokens(badUrl), values(human, ai(badUrl)), values(human, ai("")),
  ])
  assert.equal(result.assistantContent, "")
  assert.equal(displayed.content, "(No response generated)")
  assert.ok(!displayed.content.includes(badUrl))
})

test("accepts current-turn final values even without streamed text", async () => {
  const { result, displayed } = await processEvents([values(human, ai("Checked response"))])
  assert.equal(result.assistantContent, "Checked response")
  assert.equal(displayed.content, "Checked response")
})

test("does not replace the response with old history or intermediate messages", async () => {
  const { result, displayed } = await processEvents([
    values({ ...human, id: "old-input" }, ai("Old answer")),
    tokens("Current answer"),
    values({ ...human, id: "old-input" }, ai("Old answer")),
    values(ai("Old answer"), human),
    values(human, ai("Tool planning", { tool_calls: [{ id: "call", name: "search" }] })),
    values(human, ai("Intermediate"), { type: "tool", content: "Tool result" }),
  ])
  assert.equal(result.assistantContent, "Current answer")
  assert.equal(displayed.content, "Current answer")
})

test("ignores subgraph and JSON subagent responses", async () => {
  const { result, displayed } = await processEvents([
    tokens("Main answer"),
    { ...values(human, ai("Subgraph answer")), event: "values|subgraph" },
    { ...tokens("Subgraph tokens"), event: "messages|subgraph" },
    values(human, ai('{"answer":"Subagent answer"}')),
  ])
  assert.equal(result.assistantContent, "Main answer")
  assert.equal(displayed.content, "Main answer")
})

test("keeps final values authoritative even if raw tokens arrive later", async () => {
  const { result, displayed } = await processEvents([
    values(human, ai("Sanitized")), tokens(badUrl),
  ])
  assert.equal(result.assistantContent, "Sanitized")
  assert.equal(displayed.content, "Sanitized")
})
