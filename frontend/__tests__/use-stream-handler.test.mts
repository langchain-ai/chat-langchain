import assert from "node:assert/strict"
import test from "node:test"
import { decodeBase64Text } from "../lib/utils/chat/decode-base64-text.ts"

test("decodes UTF-8 attachment text without changing Unicode characters", () => {
  const original = "# 你好，世界 — LangChain"
  const base64 = Buffer.from(original, "utf8").toString("base64")

  assert.equal(decodeBase64Text(base64), original)
})

test("decodes ASCII attachment text unchanged", () => {
  const original = "# Hello, world!"
  const base64 = Buffer.from(original, "utf8").toString("base64")

  assert.equal(decodeBase64Text(base64), original)
})
