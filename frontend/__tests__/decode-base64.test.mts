import assert from "node:assert/strict"
import test from "node:test"
import { decodeBase64Text } from "../lib/utils/decode-base64.ts"

function encodeBase64(value: string): string {
  return Buffer.from(value, "utf8").toString("base64")
}

test("decodes UTF-8 text attachments without mojibake", () => {
  const content = "你好，世界！🚀"

  assert.equal(decodeBase64Text(encodeBase64(content)), content)
})

test("decodes ASCII text attachments unchanged", () => {
  const content = "# Hello\n\nThis is a plain ASCII document."

  assert.equal(decodeBase64Text(encodeBase64(content)), content)
})
