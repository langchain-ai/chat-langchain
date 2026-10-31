import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import test from "node:test"
import vm from "node:vm"
import ts from "typescript"

const source = readFileSync(
  new URL("../lib/hooks/chat/use-stream-handler.ts", import.meta.url),
  "utf8"
)
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS },
})
const exports: Record<string, any> = {}
vm.runInNewContext(outputText, {
  exports,
  require(name: string) {
    if (name === "react") return { useCallback: (callback: unknown) => callback }
    if (name === "../../constants/api") {
      return { LANGGRAPH_API_URL: "https://example.langgraph.app" }
    }
    return {}
  },
  atob,
  Uint8Array,
  TextDecoder,
  console: { log() {}, error() {} },
})

async function submittedContent(files: Record<string, unknown>[]) {
  let content: unknown
  const { processStream } = exports.useStreamHandler({
    client: {
      runs: {
        async *stream(_threadId: string, _agent: string, options: any) {
          content = options.input.messages[0].content
        },
      },
    },
    threadId: "thread-id",
    setMessages: (update: (messages: unknown[]) => unknown[]) => update([]),
    auth: {},
  })
  await processStream("请阅读附件", "assistant-id", files)
  return JSON.parse(JSON.stringify(content))
}

for (const text of ["LangChain 文档\n你好，世界！", "Emoji: 🦜🔗 😀", "Plain ASCII text"]) {
  test(`preserves UTF-8 attachment text: ${text}`, async () => {
    assert.deepEqual(await submittedContent([{
      name: "langchain.md",
      mimeType: "text/markdown",
      base64: Buffer.from(text, "utf8").toString("base64"),
    }]), [
      { type: "text", text: "请阅读附件" },
      { type: "text", text: `**File: langchain.md**\n\`\`\`\n${text}\n\`\`\`` },
    ])
  })
}

test("skips HAR attachments and preserves image blocks", async () => {
  assert.deepEqual(await submittedContent([
    { name: "trace.HAR", mimeType: "application/json", base64: "invalid!" },
    { name: "image.png", mimeType: "image/png", base64: "aW1hZ2U=" },
  ]), [
    { type: "text", text: "请阅读附件" },
    { type: "image_url", image_url: { url: "data:image/png;base64,aW1hZ2U=" } },
  ])
})

test("preserves the error block for invalid base64", async () => {
  assert.deepEqual(await submittedContent([
    { name: "broken.txt", mimeType: "text/plain", base64: "invalid!" },
  ]), [
    { type: "text", text: "请阅读附件" },
    { type: "text", text: "[Failed to decode file: broken.txt]" },
  ])
})
