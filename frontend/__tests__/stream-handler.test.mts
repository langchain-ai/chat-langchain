import assert from "node:assert/strict"
import { existsSync } from "node:fs"
import { registerHooks } from "node:module"
import test from "node:test"
import type { Message } from "../lib/types/messages.ts"

const hooks = registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier === "react") {
      return { url: "data:text/javascript,export const useCallback = (callback) => callback", shortCircuit: true }
    }
    if (specifier.startsWith("@/") || specifier.startsWith(".")) {
      const base = specifier.startsWith("@/")
        ? new URL(`../${specifier.slice(2)}`, import.meta.url)
        : new URL(specifier, context.parentURL)
      for (const suffix of [".ts", "/index.ts"]) {
        const url = new URL(`${base.href}${suffix}`)
        if (existsSync(url)) return nextResolve(url.href, context)
      }
    }
    return nextResolve(specifier, context)
  },
})
process.env.NEXT_PUBLIC_LANGGRAPH_API_URL = "http://127.0.0.1:2024"
const { useStreamHandler } = await import("../lib/hooks/chat/use-stream-handler.ts")
hooks.deregister()

type Chunk = { event: string; data: unknown }
const ai = (content: string, id = "answer") => ({ type: "ai", id, content, tool_calls: [] })
const values = (content: string, id = "answer"): Chunk => ({ event: "values", data: { messages: [ai(content, id)] } })
const tokens = (content: string): Chunk => ({ event: "messages", data: [ai(content), {}] })

async function stream(chunks: Chunk[], interruptAt = -1) {
  let messages: Message[] = []
  const states: Message[][] = []
  const shouldInterruptRef = { current: false }
  const client = {
    runs: {
      async *stream() {
        for (const [index, chunk] of chunks.entries()) {
          if (index === interruptAt) shouldInterruptRef.current = true
          yield chunk
        }
      },
    },
  }
  const { processStream } = useStreamHandler({
    client: client as never,
    threadId: "thread",
    setMessages: (update) => {
      messages = typeof update === "function" ? update(messages) : update
      states.push(messages)
    },
    shouldInterruptRef,
    auth: {} as never,
  })
  const result = await processStream("Question", "reply")
  return { message: messages.find((message) => message.id === "reply")!, result, states }
}

test("same-id validated values replace streamed links through completion", async () => {
  const draft = "Read https://docs.launchchain.com/x for details."
  const corrected = "See the documentation for details."
  const { message, result, states } = await stream([
    values("Previous answer", "previous"),
    tokens(draft),
    values(corrected),
  ])
  assert.ok(states.some((messages) => messages.some((message) => message.content === draft)))
  assert.equal(message.content, corrected)
  assert.doesNotMatch(message.content, /launchchain/)
  assert.equal(message.status, "completed")
  assert.equal(result.assistantContent, corrected)
})

test("subgraph, subagent JSON, and tool-call values preserve the streamed answer", async () => {
  for (const chunk of [
    { ...values("Subgraph answer"), event: "values|research" },
    values('{"answer":"Subagent answer"}'),
    { event: "values", data: { messages: [{ ...ai("Calling tool"), tool_calls: [{ id: "call", name: "search", args: {} }] }] } },
  ]) {
    const { message } = await stream([values("Previous", "previous"), tokens("Main answer"), chunk])
    assert.equal(message.content, "Main answer")
  }
})

test("baseline previous-turn values do not become the new answer", async () => {
  const { message } = await stream([values("Previous", "previous"), values("Previous", "previous")])
  assert.equal(message.content, "(No response generated)")
})

test("a new final answer without token events still appears", async () => {
  const { message } = await stream([values("Previous", "previous"), values("New answer")])
  assert.equal(message.content, "New answer")
})

test("interruption keeps partial content and ignores subsequent values", async () => {
  const { message, result } = await stream([
    values("Previous", "previous"), tokens("Partial answer"), values("Final answer"),
  ], 2)
  assert.equal(message.content, "Partial answer")
  assert.equal(message.status, "interrupted")
  assert.equal(message.wasInterrupted, true)
  assert.equal(result.assistantContent, "Partial answer")
})
