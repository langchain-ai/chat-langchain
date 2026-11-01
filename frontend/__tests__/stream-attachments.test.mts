import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import test from "node:test"
import { runInNewContext } from "node:vm"
import ts from "typescript"

const source = readFileSync(
  new URL("../lib/hooks/chat/use-stream-handler.ts", import.meta.url),
  "utf8"
)
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS },
})

async function sendAttachment(base64: string, mimeType = "text/plain") {
  const exports = {} as { useStreamHandler: Function }
  const errors: unknown[][] = []
  runInNewContext(outputText, {
    exports,
    require: (name: string) => {
      if (name === "react") return { useCallback: (callback: Function) => callback }
      if (name === "../../constants/api") {
        return { LANGGRAPH_API_URL: "https://example.test" }
      }
      return {}
    },
    atob,
    Uint8Array,
    TextDecoder,
    console: { log() {}, error: (...args: unknown[]) => errors.push(args) },
  })

  let input: unknown
  const { processStream } = exports.useStreamHandler({
    client: {
      runs: {
        stream(_threadId: string, _agent: string, options: { input: unknown }) {
          input = options.input
          return (async function* () {})()
        },
      },
    },
    threadId: "thread-1",
    setMessages: (update: Function) => update([]),
    auth: {},
  })
  await processStream("Review this file", "assistant-1", [
    { name: "article.md", base64, mimeType },
  ])
  return { input: JSON.parse(JSON.stringify(input)), errors }
}

for (const text of ["中文文章：café, naïve, résumé 😀", "Plain ASCII text\nconst value = 1"]) {
  test(`preserves attachment text in the agent message: ${text}`, async () => {
    const { input, errors } = await sendAttachment(Buffer.from(text, "utf8").toString("base64"))
    assert.deepEqual(input, {
      messages: [{ role: "user", content: [
        { type: "text", text: "Review this file" },
        { type: "text", text: `**File: article.md**\n\`\`\`\n${text}\n\`\`\`` },
      ] }],
    })
    assert.equal(errors.length, 0)
  })
}

test("keeps the decode failure message and logs the error", async () => {
  const { input, errors } = await sendAttachment("!invalid base64!")
  assert.equal(input.messages[0].content[1].text, "[Failed to decode file: article.md]")
  assert.equal(errors.length, 1)
  assert.equal(errors[0][0], "Failed to decode file article.md:")
})

test("preserves image attachments as data URLs", async () => {
  const { input, errors } = await sendAttachment("aW1hZ2U=", "image/png")
  assert.deepEqual(input.messages[0].content[1], {
    type: "image_url",
    image_url: { url: "data:image/png;base64,aW1hZ2U=" },
  })
  assert.equal(errors.length, 0)
})
