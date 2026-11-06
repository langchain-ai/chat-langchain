import assert from "node:assert/strict"
import test from "node:test"
import {
  appendWithinInputBudget,
  chatUIReducer,
  claimInitialPrompt,
  getAttachedTextLength,
  getAttachmentMimeType,
  limitComposerInput,
  type ChatUIState,
} from "../lib/utils/chat/composer-state.ts"

const idle: ChatUIState = {
  input: "A draft I am still writing",
  copiedId: null,
  isLoading: false,
  isLoadingThread: false,
  isRegenerating: false,
  isStopping: false,
  hasAutoSent: false,
  errorMessage: null,
}

test("queued and URL-triggered runs preserve the current composer draft", () => {
  const running = chatUIReducer(idle, { type: "START_SEND" })
  assert.equal(running.input, idle.input)
  assert.equal(running.isLoading, true)
  const finished = chatUIReducer(running, { type: "FINISH_SEND" })
  const nextQueuedRun = chatUIReducer(finished, { type: "START_SEND" })
  assert.equal(nextQueuedRun.input, idle.input)
})

test("submitting a draft explicitly clears it before the run starts", () => {
  const submitted = chatUIReducer(idle, { type: "SET_INPUT", payload: "" })
  assert.equal(chatUIReducer(submitted, { type: "START_SEND" }).input, "")
})

test("cancellation ends the stopping state without losing a newly typed draft", () => {
  for (const type of ["FINISH_SEND", "FINISH_REGENERATE"] as const) {
    const finished = chatUIReducer({ ...idle, isLoading: type === "FINISH_SEND", isRegenerating: type === "FINISH_REGENERATE", isStopping: true }, { type })
    assert.equal(finished.isStopping, false)
    assert.equal(finished.input, idle.input)
  }
})

test("initial prompts wait for auth/history and claim only once across effect replay", () => {
  const claims = new Set<string>()
  assert.equal(claimInitialPrompt(claims, "one", "  Build an agent  ", false), null)
  assert.equal(claimInitialPrompt(claims, "one", "  Build an agent  ", true), "Build an agent")
  // StrictMode may rerun the effect with the same stale reducer state.
  assert.equal(claimInitialPrompt(claims, "one", "  Build an agent  ", true), null)
  assert.equal(claimInitialPrompt(claims, "two", "Build an agent", true), "Build an agent")
})

test("blank initial prompts do not consume a thread's one-time claim", () => {
  const claims = new Set<string>()
  assert.equal(claimInitialPrompt(claims, "one", " \n ", true), null)
  assert.equal(claimInitialPrompt(claims, "one", "A question", true), "A question")
})

test("the 50,000 character budget includes text files but excludes image bytes", () => {
  const attachments = [
    { mimeType: "text/plain", textLength: 10_000 },
    { mimeType: "image/png", textLength: 100_000 },
  ]
  assert.equal(getAttachedTextLength(attachments), 10_000)
  const exact = limitComposerInput("a".repeat(40_000), getAttachedTextLength(attachments), 50_000)
  assert.equal(exact.overLimit, false)
  const tooLong = limitComposerInput("a".repeat(40_001), getAttachedTextLength(attachments), 50_000)
  assert.equal(tooLong.overLimit, true)
  assert.equal(tooLong.value.length, 40_000)
})

test("concurrent upload completions are admitted against current attachments", () => {
  const first = { mimeType: "text/plain", textLength: 30_000 }
  const second = { mimeType: "text/plain", textLength: 25_000 }
  const files = appendWithinInputBudget([], first, 0, 50_000)!
  assert.equal(appendWithinInputBudget(files, second, 0, 50_000), null)
  assert.deepEqual(files, [first])
})

test("typing while a file converts can reject that completed upload", () => {
  const upload = { mimeType: "text/plain", textLength: 20_000 }
  assert.equal(appendWithinInputBudget([], upload, 30_001, 50_000), null)
  assert.deepEqual(appendWithinInputBudget([], upload, 30_000, 50_000), [upload])
})

test("removing a text attachment restores its input budget", () => {
  const upload = { mimeType: "text/plain", textLength: 30_000 }
  assert.equal(appendWithinInputBudget([upload], upload, 0, 50_000), null)
  assert.deepEqual(appendWithinInputBudget([], upload, 0, 50_000), [upload])
})

test("image files with absent browser MIME metadata remain image attachments", () => {
  assert.equal(getAttachmentMimeType({ type: "", name: "SCREENSHOT.PNG" }), "image/png")
  assert.equal(getAttachmentMimeType({ type: "application/octet-stream", name: "photo.jpg" }), "image/jpeg")
  assert.equal(getAttachmentMimeType({ type: "", name: "main.ts" }), "text/plain")
  assert.equal(getAttachmentMimeType({ type: "application/json", name: "data.json" }), "application/json")
})
