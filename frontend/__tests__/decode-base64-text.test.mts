import assert from "node:assert/strict"
import test from "node:test"
import { decodeBase64Text } from "../lib/hooks/chat/decode-base64-text.ts"

function encodeBase64(value: string): string {
  return btoa(String.fromCharCode(...new TextEncoder().encode(value)))
}

test("decodes UTF-8 text attachments without mojibake", () => {
  const content = "你好, café 👋"

  assert.equal(decodeBase64Text(encodeBase64(content)), content)
})

test("keeps ASCII text attachments unchanged", () => {
  const content = "Plain ASCII attachment"

  assert.equal(decodeBase64Text(encodeBase64(content)), content)
})
