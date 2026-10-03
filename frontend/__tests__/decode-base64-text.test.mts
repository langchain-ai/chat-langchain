import assert from "node:assert/strict"
import test from "node:test"
import { decodeBase64Text } from "../lib/utils/decode-base64-text.ts"

test("decodes UTF-8 text without mojibake", () => {
  const original = "你好，LangChain 👋"
  const base64 = Buffer.from(original, "utf8").toString("base64")

  assert.equal(decodeBase64Text(base64, "text/markdown; charset=utf-8"), original)
})

test("decodes plain ASCII text", () => {
  const original = "plain ASCII text"
  const base64 = Buffer.from(original, "ascii").toString("base64")

  assert.equal(decodeBase64Text(base64), original)
})
